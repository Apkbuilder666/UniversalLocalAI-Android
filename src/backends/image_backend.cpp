#include "localai/backend.hpp"
#include "localai/log.hpp"

#include <stable-diffusion.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>

namespace localai {
namespace {

std::mutex generationMutex;

double number(const InferenceRequest& request, std::string_view key, double fallback) {
    const auto found = request.numeric.find(std::string(key));
    return found == request.numeric.end() ? fallback : found->second;
}

void imageProgress(int step, int steps, float, void* userData) {
    const auto* request = static_cast<const InferenceRequest*>(userData);
    if (request && request->onProgress && steps > 0) request->onProgress(step / static_cast<double>(steps));
}

Result<bool> writePpm(const std::filesystem::path& path, const sd_image_t& image) {
    if (!image.data || image.width == 0 || image.height == 0 || image.channel < 3) {
        return Result<bool>::failure("image.output", "Generated image buffer is invalid");
    }
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return Result<bool>::failure("image.output", "Unable to create image output file");
    output << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    if (image.channel == 3) {
        output.write(reinterpret_cast<const char*>(image.data),
                     static_cast<std::streamsize>(image.width * image.height * 3));
    } else {
        for (std::uint64_t pixel = 0; pixel < static_cast<std::uint64_t>(image.width) * image.height; ++pixel) {
            output.write(reinterpret_cast<const char*>(image.data + pixel * image.channel), 3);
        }
    }
    return output ? Result<bool>::success(true)
                  : Result<bool>::failure("image.output", "Failed while writing image data");
}

class ImageBackend final : public IModelBackend {
public:
    ~ImageBackend() override { unload(); }
    bool canLoad(const ModelDescriptor& model) const override {
        return (model.format == ModelFormat::Gguf || model.format == ModelFormat::SafeTensors) &&
               std::ranges::find(model.capabilities, Capability::ImageGeneration) != model.capabilities.end();
    }

    Result<BackendInfo> load(const ModelDescriptor& model, const LoadOptions& options) override {
        unload();
        sd_ctx_params_t parameters;
        sd_ctx_params_init(&parameters);
        modelPath_ = model.path.string();
#ifdef LOCALAI_IMAGE_ACCELERATOR_ENABLED
        backend_ = options.deviceId == "cpu" ? "cpu" : "";
#else
        backend_ = "cpu";
#endif
        parameters.model_path = modelPath_.c_str();
        parameters.n_threads = options.threads > 0 ? options.threads : static_cast<int>(std::thread::hardware_concurrency());
        parameters.enable_mmap = options.mmap;
        parameters.auto_fit = true;
        parameters.backend = backend_.empty() ? nullptr : backend_.c_str();
        const auto started = std::chrono::steady_clock::now();
        context_ = new_sd_ctx(&parameters);
        if (!context_ && backend_.empty()) {
            Logger::instance().write(LogLevel::Warning, LogCategory::Backend,
                "Still-image accelerator initialization failed; retrying on CPU");
            backend_ = "cpu";
            parameters.backend = backend_.c_str();
            context_ = new_sd_ctx(&parameters);
        }
        if (!context_) return Result<BackendInfo>::failure("image.load", "stable-diffusion.cpp rejected the model or components");
        stats_.loadMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return Result<BackendInfo>::success(backendInfo());
    }

    Result<InferenceOutput> infer(const InferenceRequest& request,
                                  const CancellationFlag& cancellation) override {
        std::scoped_lock globalLock(generationMutex);
        if (!context_) return Result<InferenceOutput>::failure("image.state", "No image-generation model is loaded");
        if (request.prompt.empty()) return Result<InferenceOutput>::failure("image.prompt", "A prompt is required");
        if (cancellation && cancellation->load()) return Result<InferenceOutput>::failure("task.cancelled", "Task cancelled");
        sd_img_gen_params_t parameters;
        sd_img_gen_params_init(&parameters);
        parameters.prompt = request.prompt.c_str();
        parameters.negative_prompt = request.negativePrompt.c_str();
        parameters.width = std::clamp(static_cast<int>(number(request, "width", 512)), 64, 4096);
        parameters.height = std::clamp(static_cast<int>(number(request, "height", 512)), 64, 4096);
        const bool hasInitialImage = request.imageWidth > 0 && request.imageHeight > 0 &&
            (request.imageChannels == 3 || request.imageChannels == 4) &&
            request.imagePixels.size() == static_cast<std::size_t>(request.imageWidth * request.imageHeight * request.imageChannels);
        if (!request.imagePixels.empty() && !hasInitialImage)
            return Result<InferenceOutput>::failure("image.initial", "Initial image must be a complete RGB or RGBA still image");
        if (hasInitialImage) {
            parameters.init_image = {static_cast<std::uint32_t>(request.imageWidth),
                                     static_cast<std::uint32_t>(request.imageHeight),
                                     static_cast<std::uint32_t>(request.imageChannels),
                                     const_cast<std::uint8_t*>(request.imagePixels.data())};
            parameters.strength = static_cast<float>(std::clamp(number(request, "strength", 0.75), 0.0, 1.0));
        }
        if (!request.maskPixels.empty()) {
            if (!hasInitialImage || request.maskChannels != 1 || request.maskWidth != request.imageWidth ||
                request.maskHeight != request.imageHeight ||
                request.maskPixels.size() != static_cast<std::size_t>(request.maskWidth * request.maskHeight)) {
                return Result<InferenceOutput>::failure("image.mask", "Inpainting mask must be one channel and match the initial image dimensions");
            }
            parameters.mask_image = {static_cast<std::uint32_t>(request.maskWidth),
                                     static_cast<std::uint32_t>(request.maskHeight), 1,
                                     const_cast<std::uint8_t*>(request.maskPixels.data())};
        }
        parameters.seed = static_cast<std::int64_t>(number(request, "seed", 42));
        parameters.batch_count = std::clamp(static_cast<int>(number(request, "batch", 1)), 1, 8);
        parameters.sample_params.sample_steps = std::clamp(static_cast<int>(number(request, "steps", 20)), 1, 200);
        parameters.sample_params.guidance.txt_cfg = static_cast<float>(number(request, "cfg", 7.0));
        const auto sampler = request.text.find("sampler");
        if (sampler != request.text.end()) {
            const auto method = str_to_sample_method(sampler->second.c_str());
            if (method != SAMPLE_METHOD_COUNT) parameters.sample_params.sample_method = method;
        }
        const auto scheduler = request.text.find("scheduler");
        if (scheduler != request.text.end()) {
            const auto value = str_to_scheduler(scheduler->second.c_str());
            if (value != SCHEDULER_COUNT) parameters.sample_params.scheduler = value;
        }
        if (parameters.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            parameters.sample_params.sample_method = sd_get_default_sample_method(context_);
        }
        if (parameters.sample_params.scheduler == SCHEDULER_COUNT) {
            parameters.sample_params.scheduler = sd_get_default_scheduler(context_, parameters.sample_params.sample_method);
        }
        sd_set_progress_callback(imageProgress, const_cast<InferenceRequest*>(&request));
        sd_image_t* images{};
        int count{};
        const auto started = std::chrono::steady_clock::now();
        const bool success = generate_image(context_, &parameters, &images, &count);
        const auto finished = std::chrono::steady_clock::now();
        sd_set_progress_callback(nullptr, nullptr);
        if (cancellation && cancellation->load()) {
            if (images) free_sd_images(images, count);
            return Result<InferenceOutput>::failure("task.cancelled", "Image generation cancelled");
        }
        if (!success || !images || count < 1) {
            if (images) free_sd_images(images, count);
            return Result<InferenceOutput>::failure("image.generate", "stable-diffusion.cpp did not produce an image");
        }
        InferenceOutput output;
        output.imageWidth = static_cast<int>(images[0].width);
        output.imageHeight = static_cast<int>(images[0].height);
        output.imageChannels = static_cast<int>(images[0].channel);
        const auto bytes = static_cast<std::size_t>(images[0].width) * images[0].height * images[0].channel;
        output.imagePixels.assign(images[0].data, images[0].data + bytes);
        if (!request.outputPath.empty()) {
            auto written = writePpm(request.outputPath, images[0]);
            if (!written) {
                free_sd_images(images, count);
                return Result<InferenceOutput>::failure(written.error().code, written.error().message);
            }
            output.artifactPath = request.outputPath;
        }
        free_sd_images(images, count);
        output.performance.totalMilliseconds = std::chrono::duration<double, std::milli>(finished - started).count();
        output.performance.outputUnits = parameters.sample_params.sample_steps;
        output.performance.outputUnitsPerSecond = output.performance.totalMilliseconds > 0.0
            ? parameters.sample_params.sample_steps / (output.performance.totalMilliseconds / 1000.0) : 0.0;
        output.execution = {{"engine", "stable-diffusion.cpp"},
                            {"provider", backend_.empty() ? "GGML automatic device selection" : "CPU"},
                            {"mode", !request.maskPixels.empty() ? "inpainting" : hasInitialImage ? "image-to-image" : "text-to-image"},
                            {"verification", backend_.empty() ? "Provider selected; exact graph placement is not exposed by this API"
                                                               : "CPU backend explicitly selected"}};
        stats_ = output.performance;
        return Result<InferenceOutput>::success(std::move(output));
    }

    void cancel() override { if (context_) sd_cancel_generation(context_, SD_CANCEL_ALL); }
    void unload() override {
        if (context_) free_sd_ctx(context_);
        context_ = nullptr;
    }
    BackendInfo backendInfo() const override {
        return {"stable-diffusion.cpp", "Still-image generation", "stable-diffusion.cpp", backend_.empty() ? "GGML automatic" : "CPU",
                backend_.empty() ? "Compiled GGML devices" : "CPU", {ModelFormat::Gguf, ModelFormat::SafeTensors},
                {Capability::ImageGeneration, Capability::ImageToImage, Capability::Inpainting}, true, true,
                context_ != nullptr, context_ ? "Image model initialized" : "Runtime compiled; no model initialized"};
    }
    PerformanceStats statistics() const override { return stats_; }

private:
    sd_ctx_t* context_{};
    std::string modelPath_;
    std::string backend_;
    PerformanceStats stats_;
};

}

void registerImageBackend() {
    BackendRegistry::instance().registerBackend("stable-diffusion.cpp", [] { return std::make_unique<ImageBackend>(); });
}

} // namespace localai

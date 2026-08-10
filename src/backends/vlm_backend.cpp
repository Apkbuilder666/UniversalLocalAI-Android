#include "localai/backend.hpp"

#include "localai/log.hpp"

#include <llama.h>
#include <mtmd-helper.h>
#include <mtmd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <mutex>
#include <sstream>
#include <thread>

namespace localai {
namespace {

double option(const InferenceRequest& request, std::string_view key, double fallback) {
    const auto found = request.numeric.find(std::string(key));
    return found == request.numeric.end() ? fallback : found->second;
}

bool hasStillImageExtension(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
           extension == ".bmp" || extension == ".webp";
}

std::vector<std::string> stopStrings(const InferenceRequest& request) {
    std::vector<std::string> values;
    const auto found = request.text.find("stop_sequences");
    if (found == request.text.end()) return values;
    std::size_t begin{};
    while (begin <= found->second.size()) {
        const auto end = found->second.find('|', begin);
        auto value = found->second.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!value.empty()) values.push_back(std::move(value));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return values;
}

std::size_t safeStreamLength(const std::string& generated, const std::vector<std::string>& stops) {
    std::size_t held{};
    for (const auto& stop : stops) {
        const auto maximum = std::min(stop.size() - 1, generated.size());
        for (std::size_t length = maximum; length > held; --length) {
            if (generated.compare(generated.size() - length, length, stop, 0, length) == 0) {
                held = length;
                break;
            }
        }
    }
    return generated.size() - held;
}

std::string tokenText(const llama_vocab* vocabulary, llama_token token) {
    std::string result(64, '\0');
    int written = llama_token_to_piece(vocabulary, token, result.data(), static_cast<int32_t>(result.size()), 0, true);
    if (written < 0) {
        result.resize(static_cast<std::size_t>(-written));
        written = llama_token_to_piece(vocabulary, token, result.data(), static_cast<int32_t>(result.size()), 0, true);
    }
    if (written < 0) return {};
    result.resize(static_cast<std::size_t>(written));
    return result;
}

ggml_type kvType(std::string_view value) {
    if (value == "f32") return GGML_TYPE_F32;
    if (value == "q8_0") return GGML_TYPE_Q8_0;
    if (value == "q4_0") return GGML_TYPE_Q4_0;
    return GGML_TYPE_F16;
}

std::string visionPrompt(const llama_model* model, const InferenceRequest& request) {
    std::string user = mtmd_default_marker();
    user += request.prompt.empty() ? "Describe this image." : request.prompt;
    const auto system = request.text.find("system");
    std::vector<llama_chat_message> messages;
    if (system != request.text.end() && !system->second.empty()) messages.push_back({"system", system->second.c_str()});
    messages.push_back({"user", user.c_str()});
    const char* modelTemplate = llama_model_chat_template(model, nullptr);
    if (!modelTemplate) return user;
    const int required = llama_chat_apply_template(modelTemplate, messages.data(), messages.size(), true, nullptr, 0);
    if (required <= 0 || required > 64 * 1024 * 1024) return user;
    std::string formatted(static_cast<std::size_t>(required), '\0');
    const int written = llama_chat_apply_template(modelTemplate, messages.data(), messages.size(), true,
                                                   formatted.data(), required);
    if (written < 0) return user;
    formatted.resize(static_cast<std::size_t>(written));
    return formatted;
}

class VlmBackend final : public IModelBackend {
public:
    ~VlmBackend() override { unload(); }

    bool canLoad(const ModelDescriptor& model) const override {
        return model.format == ModelFormat::Gguf &&
               std::ranges::find(model.capabilities, Capability::ImageUnderstanding) != model.capabilities.end() &&
               model.metadata.contains("mmproj.path");
    }

    Result<BackendInfo> load(const ModelDescriptor& descriptor, const LoadOptions& options) override {
        unload();
        const auto projector = descriptor.metadata.find("mmproj.path");
        if (projector == descriptor.metadata.end() || !std::filesystem::is_regular_file(projector->second)) {
            return Result<BackendInfo>::failure("vlm.projector", "A matching multimodal projector GGUF is required beside the text model");
        }
        const auto started = std::chrono::steady_clock::now();
        auto modelParameters = llama_model_default_params();
        modelParameters.n_gpu_layers = options.deviceId == "cpu" ? 0 : options.gpuLayers;
        modelParameters.load_mode = options.mlock
            ? (options.mmap ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MLOCK)
            : (options.mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE);
        model_ = llama_model_load_from_file(descriptor.path.string().c_str(), modelParameters);
        if (!model_ && modelParameters.n_gpu_layers != 0) {
            Logger::instance().write(LogLevel::Warning, LogCategory::Backend,
                "VLM accelerator model load failed; retrying on CPU");
            auto fallback = options;
            fallback.deviceId = "cpu";
            fallback.gpuLayers = 0;
            return load(descriptor, fallback);
        }
        if (!model_) return Result<BackendInfo>::failure("vlm.model", "llama.cpp rejected the VLM text model");

        auto contextParameters = llama_context_default_params();
        contextParameters.n_ctx = static_cast<std::uint32_t>(std::max(1024, options.contextSize));
        contextParameters.n_batch = static_cast<std::uint32_t>(std::max(128, options.batchSize));
        contextParameters.n_threads = options.threads > 0 ? options.threads : static_cast<int>(std::thread::hardware_concurrency());
        contextParameters.n_threads_batch = contextParameters.n_threads;
        contextParameters.type_k = kvType(options.kvCacheType);
        contextParameters.type_v = kvType(options.kvCacheType);
        context_ = llama_init_from_model(model_, contextParameters);
        if (!context_) {
            if (options.deviceId != "cpu") {
                unload();
                Logger::instance().write(LogLevel::Warning, LogCategory::Memory,
                    "VLM accelerator context allocation failed; retrying on CPU");
                auto fallback = options;
                fallback.deviceId = "cpu";
                fallback.gpuLayers = 0;
                return load(descriptor, fallback);
            }
            unload();
            return Result<BackendInfo>::failure("vlm.context", "llama.cpp could not allocate the VLM context");
        }

        auto mediaParameters = mtmd_context_params_default();
        mediaParameters.use_gpu = options.deviceId != "cpu" && llama_supports_gpu_offload();
        mediaParameters.n_threads = contextParameters.n_threads;
        mediaParameters.batch_max_tokens = static_cast<int32_t>(contextParameters.n_batch);
        media_ = mtmd_init_from_file(projector->second.c_str(), model_, mediaParameters);
        if (!media_ || !mtmd_support_vision(media_)) {
            if (options.deviceId != "cpu") {
                unload();
                Logger::instance().write(LogLevel::Warning, LogCategory::Backend,
                    "VLM projector accelerator initialization failed; retrying on CPU");
                auto fallback = options;
                fallback.deviceId = "cpu";
                fallback.gpuLayers = 0;
                return load(descriptor, fallback);
            }
            unload();
            return Result<BackendInfo>::failure("vlm.projector", "libmtmd rejected the projector or it does not expose still-image input");
        }
        options_ = options;
        projectorPath_ = projector->second;
        stats_.loadMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        info_ = backendInfo();
        info_.initialized = true;
        info_.verification = "Text model and still-image projector initialized by llama.cpp libmtmd";
        Logger::instance().write(LogLevel::Info, LogCategory::Backend,
                                 "Initialized still-image VLM with projector " + projectorPath_.filename().string());
        return Result<BackendInfo>::success(info_);
    }

    Result<InferenceOutput> infer(const InferenceRequest& request,
                                  const CancellationFlag& cancellation) override {
        std::scoped_lock guard(mutex_);
        if (!model_ || !context_ || !media_) return Result<InferenceOutput>::failure("vlm.state", "No VLM is loaded");
        if (request.kind != TaskKind::ImageUnderstanding) {
            return Result<InferenceOutput>::failure("vlm.task", "The VLM backend accepts still-image understanding requests only");
        }
        const bool hasPixels = request.imageWidth > 0 && request.imageHeight > 0 && request.imageChannels == 3 &&
            request.imagePixels.size() == static_cast<std::size_t>(request.imageWidth * request.imageHeight * 3);
        const bool hasFile = !request.inputPath.empty() && std::filesystem::is_regular_file(request.inputPath) &&
                             hasStillImageExtension(request.inputPath);
        if (!hasPixels && !hasFile) return Result<InferenceOutput>::failure(
            "vlm.image", "A decoded RGB still image or supported still-image file is required");
        cancelled_.store(false);
        llama_memory_clear(llama_get_memory(context_), true);
        const auto started = std::chrono::steady_clock::now();

        mtmd_bitmap* bitmap{};
        if (hasPixels) {
            bitmap = mtmd_bitmap_init(static_cast<std::uint32_t>(request.imageWidth),
                                      static_cast<std::uint32_t>(request.imageHeight),
                                      request.imagePixels.data());
        } else {
            bitmap = mtmd_helper_bitmap_init_from_file(media_, request.inputPath.string().c_str(), false).bitmap;
        }
        if (!bitmap) return Result<InferenceOutput>::failure("vlm.image", "libmtmd could not create the image bitmap");
        const auto bitmapWidth = mtmd_bitmap_get_nx(bitmap);
        const auto bitmapHeight = mtmd_bitmap_get_ny(bitmap);
        const auto prompt = visionPrompt(model_, request);
        mtmd_input_text text{prompt.data(), prompt.size(), true, true};
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const mtmd_bitmap* bitmaps[] = {bitmap};
        const int tokenized = mtmd_tokenize(media_, chunks, &text, bitmaps, 1);
        if (tokenized != 0) {
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bitmap);
            return Result<InferenceOutput>::failure("vlm.tokenize", "libmtmd could not combine the prompt and still image");
        }
        const auto inputTokens = mtmd_helper_get_n_tokens(chunks);
        if (inputTokens >= llama_n_ctx(context_)) {
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bitmap);
            return Result<InferenceOutput>::failure("vlm.context", "Image and prompt exceed the allocated context window");
        }
        llama_pos past{};
        const int evaluated = mtmd_helper_eval_chunks(media_, context_, chunks, 0, 0,
                                                       std::max(128, options_.batchSize), true, &past);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        if (evaluated != 0) return Result<InferenceOutput>::failure("vlm.evaluate", "libmtmd failed while encoding the still image");
        if (cancelled_.load() || (cancellation && cancellation->load())) {
            return Result<InferenceOutput>::failure("task.cancelled", "Task cancelled");
        }

        const auto* vocabulary = llama_model_get_vocab(model_);
        auto* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
            llama_vocab_n_tokens(vocabulary), static_cast<int>(option(request, "repeat_last_n", 64)),
            static_cast<float>(option(request, "repeat_penalty", 1.1)), 0.0F, 0.0F));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(static_cast<int>(option(request, "top_k", 40))));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(static_cast<float>(option(request, "top_p", 0.95)), 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(static_cast<float>(option(request, "min_p", 0.05)), 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(static_cast<float>(option(request, "temperature", 0.2))));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(static_cast<std::uint32_t>(option(request, "seed", LLAMA_DEFAULT_SEED))));

        InferenceOutput output;
        const auto stops = stopStrings(request);
        std::string generatedText;
        std::size_t streamed{};
        bool stopped{};
        auto firstOutput = std::chrono::steady_clock::time_point{};
        const int limit = std::min(static_cast<int>(option(request, "max_tokens", 256)),
                                   static_cast<int>(llama_n_ctx(context_) - past));
        int generated{};
        llama_batch batch = llama_batch_init(1, 0, 1);
        for (; generated < limit; ++generated) {
            if (cancelled_.load() || (cancellation && cancellation->load())) break;
            const llama_token token = llama_sampler_sample(sampler, context_, -1);
            if (llama_vocab_is_eog(vocabulary, token)) break;
            const auto piece = tokenText(vocabulary, token);
            if (firstOutput == std::chrono::steady_clock::time_point{}) firstOutput = std::chrono::steady_clock::now();
            generatedText += piece;
            std::size_t stopPosition = std::string::npos;
            for (const auto& stop : stops) {
                const auto position = generatedText.find(stop);
                if (position < stopPosition) stopPosition = position;
            }
            if (stopPosition != std::string::npos) {
                if (request.onText && stopPosition > streamed)
                    request.onText(std::string_view(generatedText).substr(streamed, stopPosition - streamed));
                output.text = generatedText.substr(0, stopPosition);
                stopped = true;
                ++generated;
                break;
            }
            const auto safeLength = stops.empty() ? generatedText.size() : safeStreamLength(generatedText, stops);
            if (request.onText && safeLength > streamed)
                request.onText(std::string_view(generatedText).substr(streamed, safeLength - streamed));
            streamed = safeLength;
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = past++;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;
            if (llama_decode(context_, batch) != 0) {
                llama_batch_free(batch);
                llama_sampler_free(sampler);
                return Result<InferenceOutput>::failure("vlm.decode", "llama.cpp failed while generating the VLM response");
            }
        }
        if (!stopped) {
            output.text = std::move(generatedText);
            if (request.onText && output.text.size() > streamed)
                request.onText(std::string_view(output.text).substr(streamed));
        }
        llama_batch_free(batch);
        llama_sampler_free(sampler);

        const auto finished = std::chrono::steady_clock::now();
        output.performance.loadMilliseconds = stats_.loadMilliseconds;
        output.performance.inputUnits = inputTokens;
        output.performance.outputUnits = generated;
        output.performance.totalMilliseconds = std::chrono::duration<double, std::milli>(finished - started).count();
        if (firstOutput != std::chrono::steady_clock::time_point{}) {
            output.performance.firstOutputMilliseconds = std::chrono::duration<double, std::milli>(firstOutput - started).count();
            const auto seconds = std::max(0.000001, std::chrono::duration<double>(finished - firstOutput).count());
            output.performance.outputUnitsPerSecond = generated / seconds;
        }
        output.execution = {{"engine", "llama.cpp libmtmd"},
                            {"provider", options_.deviceId == "cpu" || !llama_supports_gpu_offload() ? "CPU" : "GGML backend selection"},
                            {"projector", projectorPath_.filename().string()},
                            {"image", std::to_string(bitmapWidth) + "x" + std::to_string(bitmapHeight) + " RGB"},
                            {"kv_cache_type", options_.kvCacheType},
                            {"kv_cache_location", options_.deviceId == "cpu" || !llama_supports_gpu_offload()
                                ? "CPU" : "Runtime managed; exact placement unavailable"},
                            {"verification", "libmtmd encoded one decoded still image and llama.cpp generated this response"}};
        stats_ = output.performance;
        return Result<InferenceOutput>::success(std::move(output));
    }

    void cancel() override { cancelled_.store(true); }

    void unload() override {
        std::scoped_lock guard(mutex_);
        if (media_) mtmd_free(media_);
        if (context_) llama_free(context_);
        if (model_) llama_model_free(model_);
        media_ = nullptr;
        context_ = nullptr;
        model_ = nullptr;
        projectorPath_.clear();
        info_.initialized = false;
    }

    BackendInfo backendInfo() const override {
        return {"llama.cpp-vlm", "llama.cpp Still-image VLM", "llama.cpp libmtmd",
                llama_supports_gpu_offload() ? "CPU with compiled GGML accelerators" : "CPU",
                llama_supports_gpu_offload() ? "GGML devices" : "CPU", {ModelFormat::Gguf},
                {Capability::ImageUnderstanding}, true, true, media_ != nullptr,
                media_ ? info_.verification : "libmtmd compiled; no model and projector initialized"};
    }

    PerformanceStats statistics() const override { return stats_; }

private:
    llama_model* model_{};
    llama_context* context_{};
    mtmd_context* media_{};
    LoadOptions options_;
    std::filesystem::path projectorPath_;
    BackendInfo info_;
    PerformanceStats stats_;
    std::atomic_bool cancelled_{};
    mutable std::mutex mutex_;
};

}

void registerVlmBackend() {
    BackendRegistry::instance().registerBackend("llama.cpp-vlm", [] { return std::make_unique<VlmBackend>(); });
}

} // namespace localai

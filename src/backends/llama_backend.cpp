#include "localai/backend.hpp"

#include "localai/log.hpp"

#include <ggml-backend.h>
#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <sstream>
#include <thread>

namespace localai {
namespace {

double numeric(const InferenceRequest& request, std::string_view key, double fallback) {
    const auto found = request.numeric.find(std::string(key));
    return found == request.numeric.end() ? fallback : found->second;
}

std::string formatPrompt(const llama_model* model, const InferenceRequest& request) {
    const auto system = request.text.find("system");
    std::vector<llama_chat_message> messages;
    if (system != request.text.end() && !system->second.empty()) {
        messages.push_back({"system", system->second.c_str()});
    }
    messages.push_back({"user", request.prompt.c_str()});
    const char* modelTemplate = llama_model_chat_template(model, nullptr);
    if (!modelTemplate) return request.prompt;
    int required = llama_chat_apply_template(modelTemplate, messages.data(), messages.size(), true, nullptr, 0);
    if (required <= 0 || required > 64 * 1024 * 1024) return request.prompt;
    std::string output(static_cast<std::size_t>(required), '\0');
    const int written = llama_chat_apply_template(modelTemplate, messages.data(), messages.size(), true,
                                                   output.data(), required);
    if (written < 0) return request.prompt;
    output.resize(static_cast<std::size_t>(written));
    return output;
}

std::string tokenPiece(const llama_vocab* vocab, llama_token token) {
    std::string piece(64, '\0');
    int written = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
    if (written < 0) {
        piece.resize(static_cast<std::size_t>(-written));
        written = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
    }
    if (written < 0) return {};
    piece.resize(static_cast<std::size_t>(written));
    return piece;
}

ggml_type kvType(std::string_view value) {
    if (value == "f32") return GGML_TYPE_F32;
    if (value == "q8_0") return GGML_TYPE_Q8_0;
    if (value == "q4_0") return GGML_TYPE_Q4_0;
    return GGML_TYPE_F16;
}

std::string deviceSummary() {
    static std::once_flag loadFlag;
    std::call_once(loadFlag, [] { ggml_backend_load_all(); });
    std::ostringstream output;
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        auto* device = ggml_backend_dev_get(index);
        if (index) output << ", ";
        output << ggml_backend_dev_description(device);
    }
    return output.str().empty() ? "CPU" : output.str();
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

class LlamaBackend final : public IModelBackend {
public:
    ~LlamaBackend() override { unload(); }

    bool canLoad(const ModelDescriptor& model) const override {
        return model.format == ModelFormat::Gguf &&
               std::ranges::find(model.capabilities, Capability::TextGeneration) != model.capabilities.end();
    }

    Result<BackendInfo> load(const ModelDescriptor& model, const LoadOptions& options) override {
        unload();
        const auto started = std::chrono::steady_clock::now();
        auto parameters = llama_model_default_params();
        parameters.n_gpu_layers = options.deviceId == "cpu" ? 0 : options.gpuLayers;
        parameters.load_mode = options.mlock
            ? (options.mmap ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MLOCK)
            : (options.mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE);
        model_ = llama_model_load_from_file(model.path.string().c_str(), parameters);
        bool cpuFallback{};
        if (!model_ && parameters.n_gpu_layers != 0) {
            Logger::instance().write(LogLevel::Warning, LogCategory::Backend,
                "llama.cpp accelerator load failed; retrying the GGUF model on CPU");
            parameters.n_gpu_layers = 0;
            model_ = llama_model_load_from_file(model.path.string().c_str(), parameters);
            cpuFallback = model_ != nullptr;
        }
        if (!model_) {
            return Result<BackendInfo>::failure("llama.load", "llama.cpp rejected the GGUF model");
        }
        auto contextParameters = llama_context_default_params();
        contextParameters.n_ctx = static_cast<std::uint32_t>(std::max(128, options.contextSize));
        contextParameters.n_batch = static_cast<std::uint32_t>(std::max(32, options.batchSize));
        contextParameters.n_threads = options.threads > 0 ? options.threads : static_cast<int>(std::thread::hardware_concurrency());
        contextParameters.n_threads_batch = contextParameters.n_threads;
        contextParameters.type_k = kvType(options.kvCacheType);
        contextParameters.type_v = kvType(options.kvCacheType);
        context_ = llama_init_from_model(model_, contextParameters);
        if (!context_ && parameters.n_gpu_layers != 0) {
            llama_model_free(model_);
            model_ = nullptr;
            Logger::instance().write(LogLevel::Warning, LogCategory::Memory,
                "llama.cpp accelerator context allocation failed; retrying the GGUF model on CPU");
            parameters.n_gpu_layers = 0;
            model_ = llama_model_load_from_file(model.path.string().c_str(), parameters);
            if (model_) context_ = llama_init_from_model(model_, contextParameters);
            cpuFallback = context_ != nullptr;
        }
        if (!context_) {
            if (model_) llama_model_free(model_);
            model_ = nullptr;
            return Result<BackendInfo>::failure("llama.context", "llama.cpp could not allocate the requested context");
        }
        options_ = options;
        if (cpuFallback) {
            options_.deviceId = "cpu";
            options_.gpuLayers = 0;
        }
        offloadAvailable_ = llama_supports_gpu_offload();
        loadedPath_ = model.path;
        stats_.loadMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        info_ = backendInfo();
        info_.initialized = true;
        info_.device = deviceSummary();
        info_.verification = parameters.n_gpu_layers == 0 || !offloadAvailable_
                                 ? (cpuFallback ? "Accelerator initialization failed; model loaded on CPU fallback"
                                                : "Model loaded without an available GPU layer-offload path")
                                 : "llama.cpp device placement active; exact layer placement is reported after inference";
        Logger::instance().write(LogLevel::Info, LogCategory::Backend,
                                 "Loaded " + model.displayName + " through llama.cpp on " + info_.device);
        return Result<BackendInfo>::success(info_);
    }

    Result<InferenceOutput> infer(const InferenceRequest& request,
                                  const CancellationFlag& cancellation) override {
        std::scoped_lock lock(inferenceMutex_);
        if (!model_ || !context_) return Result<InferenceOutput>::failure("llama.state", "No GGUF model is loaded");
        cancelled_.store(false);
        llama_memory_clear(llama_get_memory(context_), true);
        const auto started = std::chrono::steady_clock::now();
        const auto prompt = formatPrompt(model_, request);
        const auto* vocab = llama_model_get_vocab(model_);
        int tokenCount = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
        if (tokenCount >= 0) return Result<InferenceOutput>::failure("llama.tokenize", "Tokenizer returned an invalid size");
        tokenCount = -tokenCount;
        if (tokenCount >= static_cast<int>(llama_n_ctx(context_))) {
            return Result<InferenceOutput>::failure("llama.context", "Prompt exceeds the allocated context window");
        }
        std::vector<llama_token> tokens(static_cast<std::size_t>(tokenCount));
        if (llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(), tokenCount,
                           true, true) < 0) {
            return Result<InferenceOutput>::failure("llama.tokenize", "Prompt tokenization failed");
        }

        auto samplerParameters = llama_sampler_chain_default_params();
        auto* sampler = llama_sampler_chain_init(samplerParameters);
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
            llama_vocab_n_tokens(vocab), static_cast<int>(numeric(request, "repeat_last_n", 64)),
            static_cast<float>(numeric(request, "repeat_penalty", 1.1)), 0.0F, 0.0F));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(static_cast<int>(numeric(request, "top_k", 40))));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(static_cast<float>(numeric(request, "top_p", 0.95)), 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(static_cast<float>(numeric(request, "min_p", 0.05)), 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(static_cast<float>(numeric(request, "temperature", 0.8))));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(static_cast<std::uint32_t>(numeric(request, "seed", LLAMA_DEFAULT_SEED))));

        InferenceOutput output;
        const auto stops = stopStrings(request);
        std::string generatedText;
        std::size_t streamed{};
        bool stopped{};
        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        int generated{};
        const int maximum = std::min(static_cast<int>(numeric(request, "max_tokens", 512)),
                                     static_cast<int>(llama_n_ctx(context_)) - tokenCount);
        auto firstOutput = std::chrono::steady_clock::time_point{};
        for (int position = 0; generated < maximum;) {
            if (cancelled_.load() || (cancellation && cancellation->load())) break;
            if (llama_decode(context_, batch) != 0) {
                llama_sampler_free(sampler);
                return Result<InferenceOutput>::failure("llama.decode", "llama.cpp failed while evaluating tokens");
            }
            position += batch.n_tokens;
            auto token = llama_sampler_sample(sampler, context_, -1);
            if (llama_vocab_is_eog(vocab, token)) break;
            auto piece = tokenPiece(vocab, token);
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
            ++generated;
            batch = llama_batch_get_one(&token, 1);
        }
        if (!stopped) {
            output.text = std::move(generatedText);
            if (request.onText && output.text.size() > streamed)
                request.onText(std::string_view(output.text).substr(streamed));
        }
        llama_sampler_free(sampler);
        const auto finished = std::chrono::steady_clock::now();
        output.performance.loadMilliseconds = stats_.loadMilliseconds;
        output.performance.inputUnits = tokens.size();
        output.performance.outputUnits = static_cast<std::uint64_t>(generated);
        output.performance.totalMilliseconds = std::chrono::duration<double, std::milli>(finished - started).count();
        if (firstOutput != std::chrono::steady_clock::time_point{}) {
            output.performance.firstOutputMilliseconds =
                std::chrono::duration<double, std::milli>(firstOutput - started).count();
        }
        const double generationSeconds = std::max(0.000001,
            std::chrono::duration<double>(finished - (firstOutput == std::chrono::steady_clock::time_point{} ? started : firstOutput)).count());
        output.performance.outputUnitsPerSecond = generated / generationSeconds;
        output.execution["engine"] = "llama.cpp";
        output.execution["device"] = info_.device;
        output.execution["provider"] = options_.gpuLayers == 0 || !offloadAvailable_ ? "CPU" : "GGML backend selection";
        output.execution["gpu_layers_requested"] = std::to_string(options_.gpuLayers);
        output.execution["context_allocated"] = std::to_string(llama_n_ctx(context_));
        output.execution["kv_cache_type"] = options_.kvCacheType;
        output.execution["kv_cache_location"] = options_.gpuLayers == 0 || !offloadAvailable_
            ? "CPU" : "Runtime managed; exact placement unavailable";
        output.execution["verification"] = options_.gpuLayers == 0 || !offloadAvailable_
            ? "No GPU layer-offload path was selected"
            : "Provider selected; exact per-operation hardware placement is not exposed by this llama.cpp API";
        stats_ = output.performance;
        return Result<InferenceOutput>::success(std::move(output));
    }

    void cancel() override { cancelled_.store(true); }

    void unload() override {
        std::scoped_lock lock(inferenceMutex_);
        if (context_) llama_free(context_);
        if (model_) llama_model_free(model_);
        context_ = nullptr;
        model_ = nullptr;
        loadedPath_.clear();
        info_.initialized = false;
    }

    BackendInfo backendInfo() const override {
        BackendInfo info;
        info.id = "llama.cpp";
        info.name = "llama.cpp";
        info.runtime = "llama.cpp / GGML";
        info.provider = llama_supports_gpu_offload() ? "CPU with compiled GGML accelerators" : "CPU";
        info.device = deviceSummary();
        info.formats = {ModelFormat::Gguf};
        info.capabilities = {Capability::TextGeneration};
        info.compiled = true;
        info.available = true;
        info.initialized = model_ != nullptr;
        info.verification = info.initialized ? info_.verification : "Runtime loaded; no model initialized";
        return info;
    }

    PerformanceStats statistics() const override { return stats_; }

private:
    llama_model* model_{};
    llama_context* context_{};
    LoadOptions options_;
    std::filesystem::path loadedPath_;
    BackendInfo info_;
    PerformanceStats stats_;
    std::atomic_bool cancelled_{};
    bool offloadAvailable_{};
    mutable std::mutex inferenceMutex_;
};

}

void registerLlamaBackend() {
    BackendRegistry::instance().registerBackend("llama.cpp", [] { return std::make_unique<LlamaBackend>(); });
}

} // namespace localai

#include "localai/backend.hpp"

#include "localai/audio.hpp"
#include "localai/log.hpp"

#include <whisper.h>

#include <chrono>
#include <mutex>
#include <thread>

namespace localai {
namespace {

struct WhisperCallbacks {
    const InferenceRequest* request{};
    CancellationFlag cancellation;
    std::atomic_bool* localCancellation{};
};

bool abortWhisper(void* userData) {
    const auto& state = *static_cast<WhisperCallbacks*>(userData);
    return state.localCancellation->load() || (state.cancellation && state.cancellation->load());
}

void whisperSegments(whisper_context* context, whisper_state*, int newSegments, void* userData) {
    const auto& state = *static_cast<WhisperCallbacks*>(userData);
    if (!state.request->onText) return;
    const int count = whisper_full_n_segments(context);
    for (int index = std::max(0, count - newSegments); index < count; ++index) {
        state.request->onText(whisper_full_get_segment_text(context, index));
    }
}

void whisperProgress(whisper_context*, whisper_state*, int progress, void* userData) {
    const auto& state = *static_cast<WhisperCallbacks*>(userData);
    if (state.request->onProgress) state.request->onProgress(progress / 100.0);
}

class WhisperBackend final : public IModelBackend {
public:
    ~WhisperBackend() override { unload(); }
    bool canLoad(const ModelDescriptor& model) const override { return model.format == ModelFormat::WhisperGgml; }

    Result<BackendInfo> load(const ModelDescriptor& model, const LoadOptions& options) override {
        unload();
        const auto started = std::chrono::steady_clock::now();
        auto parameters = whisper_context_default_params();
#ifdef LOCALAI_WHISPER_ACCELERATOR_ENABLED
        parameters.use_gpu = options.deviceId != "cpu";
#else
        parameters.use_gpu = false;
#endif
        context_ = whisper_init_from_file_with_params(model.path.string().c_str(), parameters);
        if (!context_ && parameters.use_gpu) {
            Logger::instance().write(LogLevel::Warning, LogCategory::Backend,
                "whisper.cpp accelerator initialization failed; retrying on CPU");
            parameters.use_gpu = false;
            context_ = whisper_init_from_file_with_params(model.path.string().c_str(), parameters);
        }
        if (!context_) return Result<BackendInfo>::failure("whisper.load", "whisper.cpp rejected the model file");
        threads_ = options.threads > 0 ? options.threads : static_cast<int>(std::thread::hardware_concurrency());
        gpuRequested_ = parameters.use_gpu;
        stats_.loadMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return Result<BackendInfo>::success(backendInfo());
    }

    Result<InferenceOutput> infer(const InferenceRequest& request,
                                  const CancellationFlag& cancellation) override {
        std::scoped_lock lock(mutex_);
        if (!context_) return Result<InferenceOutput>::failure("whisper.state", "No whisper.cpp model is loaded");
        std::vector<float> samples = request.tensor;
        if (samples.empty() && !request.inputPath.empty()) {
            auto audio = readWaveFile(request.inputPath, WHISPER_SAMPLE_RATE);
            if (!audio) return Result<InferenceOutput>::failure(audio.error().code, audio.error().message);
            samples = std::move(audio.value().samples);
        }
        if (samples.empty()) return Result<InferenceOutput>::failure("whisper.audio", "16 kHz mono samples or a WAVE file are required");
        cancelled_.store(false);
        WhisperCallbacks callbacks{&request, cancellation, &cancelled_};
        auto parameters = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        parameters.n_threads = std::max(1, threads_);
        parameters.print_progress = false;
        parameters.print_realtime = false;
        parameters.print_timestamps = false;
        parameters.translate = request.text.contains("translate") && request.text.at("translate") == "true";
        const auto language = request.text.find("language");
        parameters.language = language == request.text.end() ? "auto" : language->second.c_str();
        parameters.new_segment_callback = whisperSegments;
        parameters.new_segment_callback_user_data = &callbacks;
        parameters.progress_callback = whisperProgress;
        parameters.progress_callback_user_data = &callbacks;
        parameters.abort_callback = abortWhisper;
        parameters.abort_callback_user_data = &callbacks;
        const auto started = std::chrono::steady_clock::now();
        const int status = whisper_full(context_, parameters, samples.data(), static_cast<int>(samples.size()));
        const auto finished = std::chrono::steady_clock::now();
        if (cancelled_.load() || (cancellation && cancellation->load())) {
            return Result<InferenceOutput>::failure("task.cancelled", "Transcription cancelled");
        }
        if (status != 0) return Result<InferenceOutput>::failure("whisper.run", "whisper.cpp transcription failed");
        InferenceOutput output;
        for (int index = 0; index < whisper_full_n_segments(context_); ++index) {
            output.text += whisper_full_get_segment_text(context_, index);
        }
        const double elapsed = std::chrono::duration<double>(finished - started).count();
        const double audioSeconds = samples.size() / static_cast<double>(WHISPER_SAMPLE_RATE);
        output.performance.totalMilliseconds = elapsed * 1000.0;
        output.performance.inputUnits = samples.size();
        output.performance.outputUnitsPerSecond = audioSeconds > 0.0 ? elapsed / audioSeconds : 0.0;
        output.execution = {{"engine", "whisper.cpp"},
                            {"provider", gpuRequested_ ? "GGML automatic device selection" : "CPU"},
                            {"verification", gpuRequested_ ? "GPU requested; exact graph placement is not exposed by whisper.cpp"
                                                            : "CPU execution requested"},
                            {"real_time_factor", std::to_string(output.performance.outputUnitsPerSecond)}};
        stats_ = output.performance;
        return Result<InferenceOutput>::success(std::move(output));
    }

    void cancel() override { cancelled_.store(true); }
    void unload() override {
        std::scoped_lock lock(mutex_);
        if (context_) whisper_free(context_);
        context_ = nullptr;
    }
    BackendInfo backendInfo() const override {
        return {"whisper.cpp", "whisper.cpp", "whisper.cpp / GGML", gpuRequested_ ? "GGML automatic" : "CPU",
                gpuRequested_ ? "Compiled GGML devices" : "CPU", {ModelFormat::WhisperGgml},
                {Capability::SpeechToText}, true, true, context_ != nullptr,
                context_ ? "Model initialized" : "Runtime compiled; no model initialized"};
    }
    PerformanceStats statistics() const override { return stats_; }

private:
    whisper_context* context_{};
    int threads_{1};
    bool gpuRequested_{};
    std::atomic_bool cancelled_{};
    PerformanceStats stats_;
    mutable std::mutex mutex_;
};

}

void registerWhisperBackend() {
    BackendRegistry::instance().registerBackend("whisper.cpp", [] { return std::make_unique<WhisperBackend>(); });
}

} // namespace localai

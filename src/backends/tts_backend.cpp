#include "localai/backend.hpp"

#include "localai/audio.hpp"

#include <sherpa-onnx/c-api/c-api.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

namespace localai {
namespace {

std::filesystem::path firstExisting(const std::filesystem::path& directory,
                                    std::initializer_list<const char*> names) {
    for (const auto* name : names) {
        auto candidate = directory / name;
        if (std::filesystem::is_regular_file(candidate)) return candidate;
    }
    return {};
}

struct TtsProgressState {
    const InferenceRequest* request{};
    CancellationFlag external;
    std::atomic_bool* local{};
};

int32_t ttsProgress(const float*, int32_t, float progress, void* userData) {
    const auto& state = *static_cast<TtsProgressState*>(userData);
    if (state.request->onProgress) state.request->onProgress(progress);
    return state.local->load() || (state.external && state.external->load()) ? 0 : 1;
}

class TtsBackend final : public IModelBackend {
public:
    ~TtsBackend() override { unload(); }
    bool canLoad(const ModelDescriptor& model) const override { return model.format == ModelFormat::AudioBundle; }

    Result<BackendInfo> load(const ModelDescriptor& model, const LoadOptions& options) override {
        unload();
        directory_ = model.path;
        modelFile_ = firstExisting(directory_, {"model.onnx", "model.int8.onnx", "model.fp16.onnx"});
        tokens_ = firstExisting(directory_, {"tokens.txt"});
        voices_ = firstExisting(directory_, {"voices.bin"});
        lexicon_ = firstExisting(directory_, {"lexicon.txt", "lexicon-us-en.txt"});
        dataDirectory_ = directory_ / "espeak-ng-data";
        if (modelFile_.empty() || tokens_.empty()) {
            return Result<BackendInfo>::failure("tts.bundle", "TTS bundle requires a supported ONNX model and tokens.txt");
        }
        std::string family = directory_.filename().string();
        std::ranges::transform(family, family.begin(), [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
#if defined(__ANDROID__)
        provider_ = options.deviceId == "cpu" ? "cpu" : "nnapi";
#else
        provider_ = options.deviceId == "auto" ? "cpu" : options.deviceId;
#endif
        SherpaOnnxOfflineTtsConfig config{};
        const auto modelPath = modelFile_.string();
        const auto tokensPath = tokens_.string();
        const auto voicesPath = voices_.string();
        const auto dataPath = std::filesystem::is_directory(dataDirectory_) ? dataDirectory_.string() : std::string{};
        const auto lexiconPath = lexicon_.string();
        if (family.find("kitten") != std::string::npos) {
            family_ = "KittenTTS";
            config.model.kitten.model = modelPath.c_str();
            config.model.kitten.voices = voicesPath.c_str();
            config.model.kitten.tokens = tokensPath.c_str();
            config.model.kitten.data_dir = dataPath.empty() ? nullptr : dataPath.c_str();
        } else if (!voices_.empty()) {
            family_ = "Kokoro";
            config.model.kokoro.model = modelPath.c_str();
            config.model.kokoro.voices = voicesPath.c_str();
            config.model.kokoro.tokens = tokensPath.c_str();
            config.model.kokoro.data_dir = dataPath.empty() ? nullptr : dataPath.c_str();
            config.model.kokoro.lexicon = lexiconPath.empty() ? nullptr : lexiconPath.c_str();
        } else {
            family_ = "VITS";
            config.model.vits.model = modelPath.c_str();
            config.model.vits.tokens = tokensPath.c_str();
            config.model.vits.data_dir = dataPath.empty() ? nullptr : dataPath.c_str();
            config.model.vits.lexicon = lexiconPath.empty() ? nullptr : lexiconPath.c_str();
        }
        config.model.num_threads = options.threads > 0 ? options.threads : static_cast<int>(std::thread::hardware_concurrency());
        config.model.provider = provider_.c_str();
        config.model.debug = 0;
        config.max_num_sentences = 2;
        const auto started = std::chrono::steady_clock::now();
        tts_ = SherpaOnnxCreateOfflineTts(&config);
        if (!tts_ && provider_ != "cpu") {
            provider_ = "cpu";
            config.model.provider = provider_.c_str();
            tts_ = SherpaOnnxCreateOfflineTts(&config);
        }
        if (!tts_) return Result<BackendInfo>::failure("tts.load", "sherpa-onnx rejected the audio model bundle");
        stats_.loadMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return Result<BackendInfo>::success(backendInfo());
    }

    Result<InferenceOutput> infer(const InferenceRequest& request,
                                  const CancellationFlag& cancellation) override {
        std::scoped_lock lock(mutex_);
        if (!tts_) return Result<InferenceOutput>::failure("tts.state", "No TTS model is loaded");
        if (request.prompt.empty()) return Result<InferenceOutput>::failure("tts.text", "Text is required for speech synthesis");
        cancelled_.store(false);
        TtsProgressState state{&request, cancellation, &cancelled_};
        SherpaOnnxGenerationConfig parameters{};
        parameters.sid = static_cast<int32_t>(request.numeric.contains("speaker") ? request.numeric.at("speaker") : 0);
        parameters.speed = static_cast<float>(request.numeric.contains("speed") ? request.numeric.at("speed") : 1.0);
        parameters.silence_scale = static_cast<float>(request.numeric.contains("silence") ? request.numeric.at("silence") : 0.2);
        const auto started = std::chrono::steady_clock::now();
        const auto* audio = SherpaOnnxOfflineTtsGenerateWithConfig(
            tts_, request.prompt.c_str(), &parameters, ttsProgress, &state);
        const auto finished = std::chrono::steady_clock::now();
        if (cancelled_.load() || (cancellation && cancellation->load())) {
            if (audio) SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
            return Result<InferenceOutput>::failure("task.cancelled", "Speech synthesis cancelled");
        }
        if (!audio || !audio->samples || audio->n <= 0 || audio->sample_rate <= 0) {
            if (audio) SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
            return Result<InferenceOutput>::failure("tts.generate", "sherpa-onnx did not produce audio");
        }
        InferenceOutput output;
        output.audioSamples.assign(audio->samples, audio->samples + audio->n);
        output.audioSampleRate = audio->sample_rate;
        SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
        if (!request.outputPath.empty()) {
            auto saved = writeWaveFile(request.outputPath, output.audioSamples, output.audioSampleRate);
            if (!saved) return Result<InferenceOutput>::failure(saved.error().code, saved.error().message);
            output.artifactPath = request.outputPath;
        }
        output.performance.totalMilliseconds = std::chrono::duration<double, std::milli>(finished - started).count();
        output.performance.outputUnits = output.audioSamples.size();
        output.execution = {{"engine", "sherpa-onnx"}, {"model_family", family_}, {"provider", provider_},
                            {"verification", provider_ == "nnapi"
                                ? "NNAPI provider generated samples; exact GPU/NPU operation placement unavailable"
                                : "sherpa-onnx generated samples with the CPU provider"}};
        stats_ = output.performance;
        return Result<InferenceOutput>::success(std::move(output));
    }

    void cancel() override { cancelled_.store(true); }
    void unload() override {
        std::scoped_lock lock(mutex_);
        if (tts_) SherpaOnnxDestroyOfflineTts(tts_);
        tts_ = nullptr;
    }
    BackendInfo backendInfo() const override {
        return {"sherpa-onnx-tts", "Offline text-to-speech", "sherpa-onnx", provider_, provider_,
                {ModelFormat::AudioBundle}, {Capability::TextToSpeech}, true, true, tts_ != nullptr,
                tts_ ? family_ + " model initialized" : "Runtime compiled; no model initialized"};
    }
    PerformanceStats statistics() const override { return stats_; }

private:
    const SherpaOnnxOfflineTts* tts_{};
    std::filesystem::path directory_;
    std::filesystem::path modelFile_;
    std::filesystem::path tokens_;
    std::filesystem::path voices_;
    std::filesystem::path lexicon_;
    std::filesystem::path dataDirectory_;
    std::string family_;
    std::string provider_{"cpu"};
    std::atomic_bool cancelled_{};
    PerformanceStats stats_;
    mutable std::mutex mutex_;
};

}

void registerTtsBackend() {
    BackendRegistry::instance().registerBackend("sherpa-onnx-tts", [] { return std::make_unique<TtsBackend>(); });
}

} // namespace localai

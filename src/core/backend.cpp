#include "localai/backend.hpp"

#include <algorithm>
#include <stdexcept>

namespace localai {

std::string toString(ModelFormat value) {
    switch (value) {
    case ModelFormat::Gguf: return "GGUF";
    case ModelFormat::SafeTensors: return "SafeTensors";
    case ModelFormat::Onnx: return "ONNX";
    case ModelFormat::LiteRt: return "LiteRT/TFLite";
    case ModelFormat::TorchScript: return "TorchScript";
    case ModelFormat::WhisperGgml: return "whisper.cpp model";
    case ModelFormat::AudioBundle: return "Audio model bundle";
    default: return "Unknown";
    }
}

std::string toString(Capability value) {
    switch (value) {
    case Capability::TextGeneration: return "Text generation";
    case Capability::TextEmbedding: return "Text embedding";
    case Capability::Reranking: return "Reranking";
    case Capability::ImageUnderstanding: return "Image understanding";
    case Capability::ImageClassification: return "Image classification";
    case Capability::ObjectDetection: return "Object detection";
    case Capability::ImageGeneration: return "Image generation";
    case Capability::ImageToImage: return "Image-to-image";
    case Capability::Inpainting: return "Inpainting";
    case Capability::SpeechToText: return "Speech-to-text";
    case Capability::TextToSpeech: return "Text-to-speech";
    case Capability::AudioClassification: return "Audio classification";
    }
    return "Unknown";
}

std::string toString(AcceleratorType value) {
    switch (value) {
    case AcceleratorType::Cpu: return "CPU";
    case AcceleratorType::Gpu: return "GPU";
    case AcceleratorType::Npu: return "NPU";
    }
    return "Unknown";
}

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry registry;
    return registry;
}

void BackendRegistry::registerBackend(std::string id, BackendFactory factory) {
    if (id.empty() || !factory) {
        throw std::invalid_argument("Backend registration requires an identifier and factory");
    }
    std::scoped_lock lock(mutex_);
    factories_.insert_or_assign(std::move(id), std::move(factory));
}

std::vector<std::string> BackendRegistry::ids() const {
    std::scoped_lock lock(mutex_);
    std::vector<std::string> result;
    result.reserve(factories_.size());
    for (const auto& [id, factory] : factories_) {
        static_cast<void>(factory);
        result.push_back(id);
    }
    std::ranges::sort(result);
    return result;
}

std::unique_ptr<IModelBackend> BackendRegistry::create(const std::string& id) const {
    std::scoped_lock lock(mutex_);
    const auto found = factories_.find(id);
    return found == factories_.end() ? nullptr : found->second();
}

std::vector<BackendInfo> BackendRegistry::probe() const {
    std::vector<BackendInfo> result;
    for (const auto& id : ids()) {
        if (auto backend = create(id)) {
            result.push_back(backend->backendInfo());
        }
    }
    return result;
}

std::vector<std::string> BackendRegistry::compatible(const ModelDescriptor& model) const {
    std::vector<std::string> result;
    for (const auto& id : ids()) {
        if (auto backend = create(id); backend && backend->canLoad(model)) {
            result.push_back(id);
        }
    }
    return result;
}

void registerCompiledBackends() {
    static std::once_flag once;
    std::call_once(once, [] {
#ifdef LOCALAI_HAS_LLAMA
        extern void registerLlamaBackend();
        extern void registerVlmBackend();
        registerLlamaBackend();
        registerVlmBackend();
#endif
#ifdef LOCALAI_HAS_ONNX
        extern void registerOnnxBackend();
        registerOnnxBackend();
#endif
#ifdef LOCALAI_HAS_LITERT
        extern void registerLiteRtBackend();
        registerLiteRtBackend();
#endif
#ifdef LOCALAI_HAS_LIBTORCH
        extern void registerLibTorchBackend();
        registerLibTorchBackend();
#endif
#ifdef LOCALAI_HAS_OPENVINO
        extern void registerOpenVinoProvider();
        registerOpenVinoProvider();
#endif
#ifdef LOCALAI_HAS_IMAGE_GENERATION
        extern void registerImageBackend();
        registerImageBackend();
#endif
#ifdef LOCALAI_HAS_WHISPER
        extern void registerWhisperBackend();
        registerWhisperBackend();
#endif
#ifdef LOCALAI_HAS_TTS
        extern void registerTtsBackend();
        registerTtsBackend();
#endif
    });
}

} // namespace localai

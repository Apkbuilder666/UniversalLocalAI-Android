#include "localai/backend.hpp"
#include "localai/model.hpp"

#include <atomic>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>

namespace {

bool require(bool condition, std::string_view message) {
    if (!condition) std::cerr << "FAILED: " << message << '\n';
    return condition;
}

localai::LoadOptions cpuOptions() {
    localai::LoadOptions options;
    options.deviceId = "cpu";
    options.gpuLayers = 0;
    options.contextSize = 2048;
    options.batchSize = 512;
    options.threads = static_cast<int>(std::min(4U, std::max(1U, std::thread::hardware_concurrency())));
    return options;
}

bool textSmoke(const std::filesystem::path& modelPath) {
    auto descriptor = localai::ModelInspector::inspect(modelPath);
    if (!require(static_cast<bool>(descriptor), "Stories GGUF inspection")) return false;
    auto backend = localai::BackendRegistry::instance().create("llama.cpp");
    if (!require(backend != nullptr, "llama.cpp backend registration")) return false;
    auto loaded = backend->load(descriptor.value(), cpuOptions());
    if (!require(static_cast<bool>(loaded), "Stories GGUF load")) return false;
    localai::InferenceRequest request;
    request.prompt = "Once upon a time";
    request.numeric = {{"max_tokens", 8}, {"temperature", 0.0}};
    auto output = backend->infer(request, std::make_shared<std::atomic_bool>(false));
    return require(static_cast<bool>(output) && output.value().performance.outputUnits > 0,
                   "Stories GGUF real generation");
}

bool visionSmoke(const std::filesystem::path& modelPath) {
    auto descriptor = localai::ModelInspector::inspect(modelPath);
    if (!require(static_cast<bool>(descriptor), "SmolVLM GGUF inspection")) return false;
    auto backend = localai::BackendRegistry::instance().create("llama.cpp-vlm");
    if (!require(backend != nullptr && backend->canLoad(descriptor.value()), "SmolVLM projector pairing")) return false;
    auto loaded = backend->load(descriptor.value(), cpuOptions());
    if (!require(static_cast<bool>(loaded), "SmolVLM and projector load")) return false;
    localai::InferenceRequest request;
    request.kind = localai::TaskKind::ImageUnderstanding;
    request.prompt = "What is the dominant color? Answer briefly.";
    request.imageWidth = 64;
    request.imageHeight = 64;
    request.imageChannels = 3;
    request.imagePixels.resize(64 * 64 * 3);
    for (std::size_t index = 0; index < request.imagePixels.size(); index += 3) {
        request.imagePixels[index] = 220;
        request.imagePixels[index + 1] = 20;
        request.imagePixels[index + 2] = 20;
    }
    request.numeric = {{"max_tokens", 12}, {"temperature", 0.0}};
    auto output = backend->infer(request, std::make_shared<std::atomic_bool>(false));
    return require(static_cast<bool>(output) && !output.value().text.empty(), "SmolVLM real still-image inference");
}

bool whisperSmoke(const std::filesystem::path& modelPath) {
    auto descriptor = localai::ModelInspector::inspect(modelPath);
    if (!require(static_cast<bool>(descriptor), "Whisper model inspection")) return false;
    auto backend = localai::BackendRegistry::instance().create("whisper.cpp");
    if (!require(backend != nullptr, "whisper.cpp backend registration")) return false;
    if (!require(static_cast<bool>(backend->load(descriptor.value(), cpuOptions())), "Whisper model load")) return false;
    localai::InferenceRequest request;
    request.kind = localai::TaskKind::SpeechToText;
    request.tensor.assign(16000 * 3, 0.0F);
    request.text["language"] = "en";
    auto output = backend->infer(request, std::make_shared<std::atomic_bool>(false));
    return require(static_cast<bool>(output) && output.value().performance.totalMilliseconds > 0.0,
                   "Whisper real speech-to-text inference");
}

#ifdef LOCALAI_HAS_ONNX
bool onnxImageSmoke(const std::filesystem::path& modelPath) {
    std::cerr << "ONNX model smoke started\n";
    auto descriptor = localai::ModelInspector::inspect(modelPath);
    if (!require(static_cast<bool>(descriptor), "SqueezeNet ONNX inspection")) return false;
    auto backend = localai::BackendRegistry::instance().create("onnxruntime");
    if (!require(backend != nullptr && backend->canLoad(descriptor.value()), "ONNX Runtime registration")) return false;
    auto loaded = backend->load(descriptor.value(), cpuOptions());
    std::cerr << "ONNX model loaded\n";
    if (!require(static_cast<bool>(loaded), "SqueezeNet ONNX load")) return false;
    localai::InferenceRequest request;
    request.kind = localai::TaskKind::ImageClassification;
    request.imageWidth = 224;
    request.imageHeight = 224;
    request.imageChannels = 3;
    request.imagePixels.assign(224 * 224 * 3, 128);
    auto output = backend->infer(request, std::make_shared<std::atomic_bool>(false));
    std::cerr << "ONNX model invocation returned\n";
    if (!output) std::cerr << "ONNX inference error: " << output.error().message << '\n';
    return require(static_cast<bool>(output) && output.value().tensor.size() == 1000 && !output.value().text.empty(),
                   "SqueezeNet real ONNX classification");
}
#endif

#ifdef LOCALAI_HAS_LITERT
bool liteRtImageSmoke(const std::filesystem::path& modelPath) {
    auto descriptor = localai::ModelInspector::inspect(modelPath);
    if (!require(static_cast<bool>(descriptor), "MobileNet TFLite inspection")) return false;
    auto backend = localai::BackendRegistry::instance().create("litert");
    if (!require(backend != nullptr && backend->canLoad(descriptor.value()), "LiteRT registration")) return false;
    auto loaded = backend->load(descriptor.value(), cpuOptions());
    if (!loaded) std::cerr << "LiteRT load error: " << loaded.error().message << '\n';
    if (!require(static_cast<bool>(loaded), "MobileNet LiteRT load")) return false;
    localai::InferenceRequest request;
    request.kind = localai::TaskKind::ImageClassification;
    request.imageWidth = 224;
    request.imageHeight = 224;
    request.imageChannels = 3;
    request.imagePixels.assign(224 * 224 * 3, 128);
    auto output = backend->infer(request, std::make_shared<std::atomic_bool>(false));
    if (!output) std::cerr << "LiteRT inference error: " << output.error().message << '\n';
    return require(static_cast<bool>(output) && output.value().tensor.size() >= 1000 && !output.value().text.empty(),
                   "MobileNet real LiteRT classification");
}
#endif

}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "A model directory argument is required\n";
        return 2;
    }
    localai::registerCompiledBackends();
    const std::filesystem::path directory(argv[1]);
    bool passed = true;
#ifdef LOCALAI_HAS_ONNX
    passed = onnxImageSmoke(directory / "squeezenet1.1-7.onnx") && passed;
    std::cerr << "ONNX model smoke completed\n";
#endif
#ifdef LOCALAI_HAS_LITERT
    passed = liteRtImageSmoke(directory / "mobilenet_v1_1.0_224.tflite") && passed;
    std::cerr << "LiteRT model smoke completed\n";
#endif
    passed = whisperSmoke(directory / "ggml-tiny.en-q5_1.bin") && passed;
    std::cerr << "Whisper model smoke completed\n";
    passed = textSmoke(directory / "stories15M-q4_0.gguf") && passed;
    std::cerr << "Text model smoke completed\n";
    passed = visionSmoke(directory / "SmolVLM-256M-Instruct-Q8_0.gguf") && passed;
    std::cerr << "VLM model smoke completed\n";
    return passed ? 0 : 1;
}

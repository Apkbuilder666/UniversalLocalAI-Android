#include "localai/backend.hpp"
#include "localai/hardware.hpp"
#include "localai/model.hpp"

#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ranges>
#include <string_view>
#include <thread>

namespace {

void usage() {
    std::cout << "Universal Local AI CLI\n"
                 "  localai-cli hardware\n"
                 "  localai-cli inspect <model-or-bundle>\n"
                 "  localai-cli run <model-or-bundle> <input> [output]\n"
                 "  localai-cli vision <model.gguf> <still-image> [prompt]\n"
                 "  localai-cli benchmark <model-or-bundle> <input>\n";
}

int inspect(const std::filesystem::path& path) {
    auto result = localai::ModelInspector::inspect(path);
    if (!result) {
        std::cerr << result.error().message << '\n';
        return 2;
    }
    const auto& model = result.value();
    std::cout << "Model: " << model.displayName << "\nFormat: " << localai::toString(model.format)
              << "\nSize: " << model.fileBytes << " bytes\nCapabilities:";
    if (model.capabilities.empty()) std::cout << " model-specific adapter required";
    for (const auto capability : model.capabilities) std::cout << "\n  " << localai::toString(capability);
    std::cout << "\nMetadata:\n";
    for (const auto& [key, value] : model.metadata) std::cout << "  " << key << ": " << value << '\n';
    localai::registerCompiledBackends();
    const auto compatible = localai::BackendRegistry::instance().compatible(model);
    std::cout << "Compatible compiled backends:";
    if (compatible.empty()) std::cout << " none";
    for (const auto& id : compatible) std::cout << "\n  " << id;
    std::cout << '\n';
    return 0;
}

int hardware() {
    const auto snapshot = localai::HardwareDiscovery::discover();
    std::cout << "CPU\n  " << snapshot.cpu.model << "\n  Architecture: " << snapshot.cpu.architecture
              << "\n  Logical cores: " << snapshot.cpu.logicalCores << "\nMemory\n  Total: "
              << snapshot.memory.totalBytes << " bytes\n  Available: " << snapshot.memory.availableBytes << " bytes\nDevices\n";
    for (const auto& device : snapshot.devices) {
        std::cout << "  " << localai::toString(device.type) << " | " << device.name << " | "
                  << device.runtime << " | " << device.status << '\n';
    }
    return 0;
}

int run(const std::filesystem::path& path, const std::string& input,
        const std::filesystem::path& outputPath, bool benchmarkOnly, bool visionMode = false) {
    auto inspected = localai::ModelInspector::inspect(path);
    if (!inspected) {
        std::cerr << inspected.error().message << '\n';
        return 2;
    }
    localai::registerCompiledBackends();
    const auto compatible = localai::BackendRegistry::instance().compatible(inspected.value());
    if (compatible.empty()) {
        std::cerr << "No compiled backend accepts this model\n";
        return 3;
    }
    auto backendId = compatible.front();
    if (visionMode) {
        const auto found = std::ranges::find(compatible, std::string("llama.cpp-vlm"));
        if (found == compatible.end()) {
            std::cerr << "The selected model has no compiled still-image VLM adapter\n";
            return 3;
        }
        backendId = *found;
    }
    auto backend = localai::BackendRegistry::instance().create(backendId);
    localai::LoadOptions options;
    options.threads = static_cast<int>(std::thread::hardware_concurrency());
    auto loaded = backend->load(inspected.value(), options);
    if (!loaded) {
        std::cerr << loaded.error().message << '\n';
        return 4;
    }
    localai::InferenceRequest request;
    request.prompt = input;
    request.outputPath = outputPath;
    request.onText = benchmarkOnly ? std::function<void(std::string_view)>{}
                                   : [](std::string_view value) { std::cout << value << std::flush; };
    if (visionMode) {
        request.kind = localai::TaskKind::ImageUnderstanding;
        request.inputPath = input;
        request.prompt = outputPath.empty() ? "Describe this image accurately." : outputPath.string();
        request.outputPath.clear();
        request.numeric = {{"max_tokens", 256}, {"temperature", 0.2}, {"top_k", 40}, {"top_p", 0.95}};
    } else if (inspected.value().format == localai::ModelFormat::WhisperGgml) {
        request.kind = localai::TaskKind::SpeechToText;
        request.inputPath = input;
        request.prompt.clear();
    } else if (inspected.value().format == localai::ModelFormat::AudioBundle) {
        request.kind = localai::TaskKind::TextToSpeech;
    } else if (std::ranges::find(inspected.value().capabilities, localai::Capability::ImageGeneration) !=
               inspected.value().capabilities.end()) {
        request.kind = localai::TaskKind::ImageGeneration;
        request.numeric = {{"width", 512}, {"height", 512}, {"steps", 20}, {"cfg", 7}, {"seed", 42}};
    } else {
        request.kind = localai::TaskKind::Text;
        request.numeric = {{"max_tokens", 256}, {"temperature", 0.8}, {"top_k", 40}, {"top_p", 0.95}};
    }
    auto result = backend->infer(request, std::make_shared<std::atomic_bool>(false));
    if (!result) {
        std::cerr << result.error().message << '\n';
        return 5;
    }
    if (!benchmarkOnly && !result.value().text.empty() && !request.onText) std::cout << result.value().text;
    if (!benchmarkOnly) std::cout << '\n';
    std::cout << std::fixed << std::setprecision(2)
              << "Total: " << result.value().performance.totalMilliseconds << " ms\n"
              << "Rate: " << result.value().performance.outputUnitsPerSecond << " units/s\n";
    for (const auto& [key, value] : result.value().execution) std::cout << key << ": " << value << '\n';
    return 0;
}

}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    const std::string_view command = argv[1];
    if (command == "hardware") return hardware();
    if (command == "inspect" && argc == 3) return inspect(argv[2]);
    if ((command == "run" || command == "benchmark") && argc >= 4) {
        return run(argv[2], argv[3], argc >= 5 ? std::filesystem::path(argv[4]) : std::filesystem::path{},
                   command == "benchmark");
    }
    if (command == "vision" && argc >= 4) {
        return run(argv[2], argv[3], argc >= 5 ? std::filesystem::path(argv[4]) : std::filesystem::path{},
                   false, true);
    }
    usage();
    return 1;
}

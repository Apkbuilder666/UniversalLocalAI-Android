#include "localai/backend.hpp"
#include "localai/benchmark.hpp"
#include "localai/hardware.hpp"
#include "localai/model.hpp"
#include "localai/scheduler.hpp"
#include "localai/settings.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

namespace {

int failures{};

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

template <typename T>
void write(std::ostream& output, T value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void writeString(std::ostream& output, std::string_view value) {
    write(output, static_cast<std::uint64_t>(value.size()));
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
}

std::filesystem::path createGguf(const std::filesystem::path& directory) {
    const auto path = directory / "test-model.gguf";
    std::ofstream output(path, std::ios::binary);
    output.write("GGUF", 4);
    write(output, std::uint32_t{3});
    write(output, std::uint64_t{0});
    write(output, std::uint64_t{2});
    writeString(output, "general.architecture");
    write(output, std::uint32_t{8});
    writeString(output, "llama");
    writeString(output, "general.name");
    write(output, std::uint32_t{8});
    writeString(output, "Unit Test Model");
    return path;
}

void modelInspection(const std::filesystem::path& directory) {
    auto result = localai::ModelInspector::inspect(createGguf(directory));
    check(static_cast<bool>(result), "valid GGUF should inspect successfully");
    if (result) {
        check(result.value().format == localai::ModelFormat::Gguf, "GGUF format should be detected");
        check(result.value().displayName == "Unit Test Model", "GGUF name should be read from metadata");
        check(result.value().capabilities == std::vector{localai::Capability::TextGeneration},
              "LLM GGUF should classify as text generation");
        check(result.value().estimatedRamBytes.has_value() && *result.value().estimatedRamBytes >= result.value().fileBytes,
              "model memory estimate should cover file-backed weights");
    }
    std::ofstream invalid(directory / "invalid.onnx", std::ios::binary);
    invalid << "x";
    invalid.close();
    auto invalidResult = localai::ModelInspector::inspect(directory / "invalid.onnx");
    check(static_cast<bool>(invalidResult), "non-empty ONNX container should remain inspectable without fabricated metadata");
    if (invalidResult) check(invalidResult.value().capabilities.empty(), "unknown ONNX architecture should have no guessed capability");

    const auto incompleteBundle = directory / "incomplete-tts";
    std::filesystem::create_directories(incompleteBundle);
    std::ofstream(incompleteBundle / "tokens.txt") << "token";
    check(!localai::ModelInspector::inspect(incompleteBundle),
          "TTS bundle without an executable ONNX model should be rejected");
    const auto completeBundle = directory / "complete-tts";
    std::filesystem::create_directories(completeBundle);
    std::ofstream(completeBundle / "tokens.txt") << "token";
    std::ofstream(completeBundle / "model.onnx", std::ios::binary) << "model";
    const auto bundleResult = localai::ModelInspector::inspect(completeBundle);
    check(bundleResult && bundleResult.value().format == localai::ModelFormat::AudioBundle,
          "TTS bundle with model.onnx and tokens.txt should be recognized");
}

void catalogPersistence(const std::filesystem::path& directory) {
    const auto database = directory / "catalog.dat";
    const auto modelPath = createGguf(directory);
    localai::ModelCatalog catalog(database);
    auto imported = catalog.importModel(modelPath);
    check(static_cast<bool>(imported) && catalog.models().size() == 1, "catalog should import a valid model");
    localai::ModelCatalog restored(database);
    check(static_cast<bool>(restored.load()) && restored.models().size() == 1, "catalog should restore persisted paths");
    if (imported) check(restored.remove(imported.value().id) && restored.models().empty(), "catalog should remove an entry without deleting model data");
}

class TestBackend final : public localai::IModelBackend {
public:
    bool canLoad(const localai::ModelDescriptor& model) const override { return model.format == localai::ModelFormat::Onnx; }
    localai::Result<localai::BackendInfo> load(const localai::ModelDescriptor&, const localai::LoadOptions&) override {
        initialized_ = true;
        return localai::Result<localai::BackendInfo>::success(backendInfo());
    }
    localai::Result<localai::InferenceOutput> infer(const localai::InferenceRequest&,
                                                    const localai::CancellationFlag&) override {
        return localai::Result<localai::InferenceOutput>::failure("test.only", "Not used by this registry test");
    }
    void cancel() override { cancelled_ = true; }
    void unload() override { initialized_ = false; }
    localai::BackendInfo backendInfo() const override {
        return {"test-backend", "Test backend", "Test runtime", "CPU", "CPU", {localai::ModelFormat::Onnx}, {},
                true, true, initialized_, "Test registration"};
    }
    localai::PerformanceStats statistics() const override { return {}; }
private:
    bool initialized_{};
    bool cancelled_{};
};

void backendRegistry() {
    auto& registry = localai::BackendRegistry::instance();
    registry.registerBackend("test-backend", [] { return std::make_unique<TestBackend>(); });
    localai::ModelDescriptor onnx;
    onnx.format = localai::ModelFormat::Onnx;
    const auto compatible = registry.compatible(onnx);
    check(std::ranges::find(compatible, std::string("test-backend")) != compatible.end(),
          "backend registry should select a compatible format adapter");
    check(registry.create("missing-backend") == nullptr, "backend registry should reject unknown identifiers");
}

void settingsPersistence(const std::filesystem::path& directory) {
    const auto path = directory / "settings.dat";
    localai::Settings settings(path);
    settings.set("theme", "dark");
    settings.set("context", "4096");
    check(static_cast<bool>(settings.save()), "settings should save");
    localai::Settings restored(path);
    check(static_cast<bool>(restored.load()), "settings should load");
    check(restored.get("theme") == "dark", "string setting should persist");
    check(restored.getInt("context", 0) == 4096, "integer setting should persist");
}

void benchmarkPersistence(const std::filesystem::path& directory) {
    const auto path = directory / "benchmarks.dat";
    localai::BenchmarkStore store(path);
    localai::PerformanceStats cpu;
    cpu.outputUnitsPerSecond = 8.0;
    localai::PerformanceStats accelerated;
    accelerated.outputUnitsPerSecond = 21.0;
    store.record({"model-a", "cpu-backend", "CPU", "CPU", cpu});
    store.record({"model-a", "gpu-backend", "Vulkan", "GPU", accelerated});
    check(static_cast<bool>(store.save()), "benchmarks should save");
    localai::BenchmarkStore restored(path);
    check(static_cast<bool>(restored.load()) && restored.forModel("model-a").size() == 2,
          "benchmarks should load by model");
    const auto fastest = restored.fastestBackend("model-a", {"cpu-backend", "gpu-backend"});
    check(fastest && *fastest == "gpu-backend", "Auto selection should use the fastest measured compatible backend");
}

void schedulerCancellation() {
    localai::InferenceScheduler scheduler(1);
    auto task = scheduler.submit([](const localai::CancellationFlag& cancellation) {
        while (!cancellation->load()) std::this_thread::yield();
        return localai::Result<localai::InferenceOutput>::failure("task.cancelled", "cancelled");
    });
    task.cancel();
    auto result = task.future.get();
    check(!result && result.error().code == "task.cancelled", "scheduled task should observe cancellation");
    scheduler.shutdown();
}

void hardwareDiscovery() {
    const auto hardware = localai::HardwareDiscovery::discover();
    check(hardware.cpu.logicalCores > 0, "logical CPU count should be positive");
    check(!hardware.cpu.architecture.empty(), "architecture should be reported");
    check(!hardware.devices.empty() && hardware.devices.front().available, "CPU fallback should always be available");
}

}

int main() {
    const auto directory = std::filesystem::temp_directory_path() /
        ("universal-local-ai-tests-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    modelInspection(directory);
    catalogPersistence(directory);
    settingsPersistence(directory);
    benchmarkPersistence(directory);
    schedulerCancellation();
    backendRegistry();
    hardwareDiscovery();
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    if (failures) std::cerr << failures << " test checks failed\n";
    return failures ? 1 : 0;
}

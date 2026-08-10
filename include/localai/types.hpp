#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace localai {

enum class ModelFormat { Unknown, Gguf, SafeTensors, Onnx, LiteRt, TorchScript, WhisperGgml, AudioBundle };
enum class Capability {
    TextGeneration,
    TextEmbedding,
    Reranking,
    ImageUnderstanding,
    ImageClassification,
    ObjectDetection,
    ImageGeneration,
    ImageToImage,
    Inpainting,
    SpeechToText,
    TextToSpeech,
    AudioClassification
};
enum class AcceleratorType { Cpu, Gpu, Npu };
enum class TaskKind {
    Text,
    Tensor,
    ImageUnderstanding,
    ImageClassification,
    ImageGeneration,
    SpeechToText,
    TextToSpeech
};

struct Error {
    std::string code;
    std::string message;
};

template <typename T>
class Result {
public:
    static Result success(T value) { return Result(std::move(value)); }
    static Result failure(std::string code, std::string message) {
        return Result(Error{std::move(code), std::move(message)});
    }
    explicit operator bool() const { return std::holds_alternative<T>(data_); }
    const T& value() const { return std::get<T>(data_); }
    T& value() { return std::get<T>(data_); }
    const Error& error() const { return std::get<Error>(data_); }

private:
    explicit Result(T value) : data_(std::move(value)) {}
    explicit Result(Error error) : data_(std::move(error)) {}
    std::variant<T, Error> data_;
};

struct ModelDescriptor {
    std::filesystem::path path;
    std::string id;
    std::string displayName;
    ModelFormat format{ModelFormat::Unknown};
    std::vector<Capability> capabilities;
    std::map<std::string, std::string> metadata;
    std::uintmax_t fileBytes{};
    std::optional<std::uint64_t> estimatedRamBytes;
    std::optional<std::uint64_t> estimatedVramBytes;
};

struct ExecutionDevice {
    std::string id;
    std::string name;
    AcceleratorType type{AcceleratorType::Cpu};
    std::string vendor;
    std::string runtime;
    std::optional<std::uint64_t> availableMemoryBytes;
    std::vector<ModelFormat> formats;
    std::vector<std::string> precisions;
    bool available{};
    bool initialized{};
    std::string status;
};

struct LoadOptions {
    std::string deviceId{"auto"};
    int threads{};
    int contextSize{4096};
    int batchSize{512};
    int gpuLayers{-1};
    std::string kvCacheType{"f16"};
    bool mmap{true};
    bool mlock{};
};

struct InferenceRequest {
    TaskKind kind{TaskKind::Text};
    std::string prompt;
    std::string negativePrompt;
    std::filesystem::path inputPath;
    std::filesystem::path outputPath;
    std::vector<std::uint8_t> imagePixels;
    int imageWidth{};
    int imageHeight{};
    int imageChannels{};
    std::vector<std::uint8_t> maskPixels;
    int maskWidth{};
    int maskHeight{};
    int maskChannels{};
    std::vector<float> tensor;
    std::vector<std::int64_t> tensorShape;
    std::map<std::string, double> numeric;
    std::map<std::string, std::string> text;
    std::function<void(std::string_view)> onText;
    std::function<void(double)> onProgress;
};

struct PerformanceStats {
    double loadMilliseconds{};
    double firstOutputMilliseconds{};
    double totalMilliseconds{};
    double promptUnitsPerSecond{};
    double outputUnitsPerSecond{};
    std::uint64_t inputUnits{};
    std::uint64_t outputUnits{};
};

struct InferenceOutput {
    std::string text;
    std::filesystem::path artifactPath;
    std::vector<float> tensor;
    std::vector<std::int64_t> tensorShape;
    std::vector<float> audioSamples;
    int audioSampleRate{};
    std::vector<std::uint8_t> imagePixels;
    int imageWidth{};
    int imageHeight{};
    int imageChannels{};
    std::map<std::string, std::string> execution;
    PerformanceStats performance;
};

struct BackendInfo {
    std::string id;
    std::string name;
    std::string runtime;
    std::string provider;
    std::string device;
    std::vector<ModelFormat> formats;
    std::vector<Capability> capabilities;
    bool compiled{};
    bool available{};
    bool initialized{};
    std::string verification;
};

using CancellationFlag = std::shared_ptr<std::atomic_bool>;

std::string toString(ModelFormat value);
std::string toString(Capability value);
std::string toString(AcceleratorType value);

} // namespace localai

#include "localai/model.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

namespace localai {
namespace {

class BinaryReader {
public:
    explicit BinaryReader(const std::filesystem::path& path) : input_(path, std::ios::binary) {}
    bool valid() const { return static_cast<bool>(input_); }

    template <typename T>
    bool read(T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        input_.read(reinterpret_cast<char*>(&value), sizeof(T));
        return static_cast<bool>(input_);
    }

    bool readString(std::string& value) {
        std::uint64_t size{};
        if (!read(size) || size > 64ULL * 1024ULL * 1024ULL) return false;
        value.resize(static_cast<std::size_t>(size));
        input_.read(value.data(), static_cast<std::streamsize>(size));
        return static_cast<bool>(input_);
    }

    bool skip(std::uint64_t bytes) {
        if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) return false;
        input_.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
        return static_cast<bool>(input_);
    }

private:
    std::ifstream input_;
};

bool skipGgufValue(BinaryReader& reader, std::uint32_t type, std::string* rendered = nullptr,
                   std::uint32_t depth = 0) {
    if (depth > 4) return false;
    auto scalar = [&](auto tag) {
        using Value = decltype(tag);
        Value value{};
        if (!reader.read(value)) return false;
        if (rendered) {
            if constexpr (std::is_same_v<Value, std::uint8_t>) *rendered = std::to_string(value);
            else if constexpr (std::is_same_v<Value, std::int8_t>) *rendered = std::to_string(value);
            else *rendered = std::to_string(value);
        }
        return true;
    };
    switch (type) {
    case 0: return scalar(std::uint8_t{});
    case 1: return scalar(std::int8_t{});
    case 2: return scalar(std::uint16_t{});
    case 3: return scalar(std::int16_t{});
    case 4: return scalar(std::uint32_t{});
    case 5: return scalar(std::int32_t{});
    case 6: return scalar(float{});
    case 7: {
        std::uint8_t value{};
        if (!reader.read(value)) return false;
        if (rendered) *rendered = value ? "true" : "false";
        return true;
    }
    case 8: {
        std::string value;
        if (!reader.readString(value)) return false;
        if (rendered) *rendered = std::move(value);
        return true;
    }
    case 9: {
        std::uint32_t elementType{};
        std::uint64_t count{};
        if (!reader.read(elementType) || !reader.read(count) || count > 100000000ULL) return false;
        for (std::uint64_t index = 0; index < count; ++index) {
            if (!skipGgufValue(reader, elementType, nullptr, depth + 1)) return false;
        }
        if (rendered) *rendered = "array[" + std::to_string(count) + "]";
        return true;
    }
    case 10: return scalar(std::uint64_t{});
    case 11: return scalar(std::int64_t{});
    case 12: return scalar(double{});
    default: return false;
    }
}

bool parseGguf(const std::filesystem::path& path, ModelDescriptor& descriptor) {
    BinaryReader reader(path);
    std::array<char, 4> magic{};
    std::uint32_t version{};
    std::uint64_t tensorCount{};
    std::uint64_t metadataCount{};
    if (!reader.valid() || !reader.read(magic) || std::memcmp(magic.data(), "GGUF", 4) != 0 ||
        !reader.read(version) || !reader.read(tensorCount) || !reader.read(metadataCount)) {
        return false;
    }
    if (version < 2 || version > 3 || metadataCount > 1000000ULL) return false;
    descriptor.metadata["gguf.version"] = std::to_string(version);
    descriptor.metadata["gguf.tensor_count"] = std::to_string(tensorCount);
    for (std::uint64_t index = 0; index < metadataCount; ++index) {
        std::string key;
        std::uint32_t type{};
        if (!reader.readString(key) || key.size() > 4096 || !reader.read(type)) return false;
        const bool retain = key == "general.architecture" || key == "general.name" ||
                            key == "general.file_type" || key.ends_with(".context_length") ||
                            key.ends_with(".embedding_length") || key.ends_with(".block_count") ||
                            key.find("tokenizer.chat_template") != std::string::npos;
        std::string value;
        if (!skipGgufValue(reader, type, retain ? &value : nullptr)) return false;
        if (retain) descriptor.metadata.insert_or_assign(std::move(key), std::move(value));
    }
    return true;
}

bool containsBytes(const std::filesystem::path& path, std::string_view needle, std::uint64_t limit) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    std::string data(static_cast<std::size_t>(std::min<std::uint64_t>(limit, 16ULL * 1024ULL * 1024ULL)), '\0');
    input.read(data.data(), static_cast<std::streamsize>(data.size()));
    data.resize(static_cast<std::size_t>(input.gcount()));
    return data.find(needle) != std::string::npos;
}

std::string makeId(const std::filesystem::path& path, std::uintmax_t bytes) {
    const auto canonical = std::filesystem::weakly_canonical(path).generic_string();
    const auto value = std::hash<std::string>{}(canonical + ':' + std::to_string(bytes));
    std::ostringstream output;
    output << std::hex << std::setw(16) << std::setfill('0') << value;
    return output.str();
}

std::uintmax_t directorySize(const std::filesystem::path& path) {
    std::uintmax_t total{};
    std::error_code error;
    for (std::filesystem::recursive_directory_iterator iterator(
             path, std::filesystem::directory_options::skip_permission_denied, error), end;
         iterator != end; iterator.increment(error)) {
        if (error) {
            error.clear();
            continue;
        }
        if (iterator->is_regular_file(error) && !iterator->is_symlink(error)) {
            const auto size = iterator->file_size(error);
            if (!error && size <= std::numeric_limits<std::uintmax_t>::max() - total) total += size;
        }
    }
    return total;
}

bool hasAnyFile(const std::filesystem::path& directory, const std::set<std::string>& names) {
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
        if (!error && entry.is_regular_file(error) && names.contains(entry.path().filename().string())) return true;
    }
    return false;
}

std::optional<std::filesystem::path> matchingProjector(const std::filesystem::path& modelPath) {
    std::error_code error;
    const auto directory = modelPath.parent_path();
    const auto modelName = modelPath.filename().string();
    std::filesystem::path firstProjector;
    for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
        if (error) break;
        if (!entry.is_regular_file(error) || entry.path().extension() != ".gguf") continue;
        const auto candidate = entry.path().filename().string();
        if (!candidate.starts_with("mmproj-")) continue;
        if (firstProjector.empty()) firstProjector = entry.path();
        auto suffix = candidate.substr(7);
        if (suffix.ends_with("-Q8_0.gguf")) suffix.resize(suffix.size() - 10);
        else if (suffix.ends_with("-f16.gguf")) suffix.resize(suffix.size() - 9);
        if (modelName.find(suffix) != std::string::npos) return entry.path();
    }
    if (!firstProjector.empty()) return firstProjector;
    return std::nullopt;
}

}

Result<ModelDescriptor> ModelInspector::inspect(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::exists(path, error) || error) {
        return Result<ModelDescriptor>::failure("model.missing", "The selected model path does not exist");
    }
    ModelDescriptor descriptor;
    descriptor.path = std::filesystem::weakly_canonical(path, error);
    if (error) descriptor.path = path;
    descriptor.displayName = path.filename().string();

    if (std::filesystem::is_directory(path, error)) {
        if (!hasAnyFile(path, {"model.onnx", "model.int8.onnx", "model.fp16.onnx"}) ||
            !hasAnyFile(path, {"tokens.txt"})) {
            return Result<ModelDescriptor>::failure(
                "model.bundle", "Offline TTS bundles require a supported ONNX model and tokens.txt");
        }
        descriptor.format = ModelFormat::AudioBundle;
        descriptor.capabilities = {Capability::TextToSpeech};
        descriptor.fileBytes = directorySize(path);
        descriptor.metadata["bundle.kind"] = "sherpa-onnx";
    } else if (std::filesystem::is_regular_file(path, error)) {
        descriptor.fileBytes = std::filesystem::file_size(path, error);
        if (error || descriptor.fileBytes == 0) {
            return Result<ModelDescriptor>::failure("model.file", "Model is empty or unreadable");
        }
        std::array<char, 8> header{};
        std::ifstream input(path, std::ios::binary);
        input.read(header.data(), static_cast<std::streamsize>(header.size()));
        const auto extension = path.extension().string();
        if (std::memcmp(header.data(), "GGUF", 4) == 0) {
            descriptor.format = ModelFormat::Gguf;
            if (!parseGguf(path, descriptor)) {
                return Result<ModelDescriptor>::failure("model.gguf", "GGUF metadata is malformed or unsupported");
            }
            const auto architecture = descriptor.metadata.find("general.architecture");
            const auto filename = path.filename().string();
            if (filename.starts_with("mmproj-")) {
                descriptor.capabilities.clear();
                descriptor.metadata["gguf.role"] = "multimodal projector";
            } else if (architecture != descriptor.metadata.end() &&
                (architecture->second.find("stable") != std::string::npos ||
                 architecture->second.find("flux") != std::string::npos)) {
                descriptor.capabilities = {Capability::ImageGeneration};
            } else {
                descriptor.capabilities = {Capability::TextGeneration};
                if (const auto projector = matchingProjector(path)) {
                    descriptor.capabilities.push_back(Capability::ImageUnderstanding);
                    descriptor.metadata["mmproj.path"] = projector->string();
                }
            }
        } else if (extension == ".safetensors") {
            std::uint64_t headerLength{};
            std::memcpy(&headerLength, header.data(), sizeof(headerLength));
            if (descriptor.fileBytes < 9 || headerLength == 0 || headerLength > descriptor.fileBytes - 8 ||
                headerLength > 64ULL * 1024ULL * 1024ULL) {
                return Result<ModelDescriptor>::failure("model.safetensors", "SafeTensors header length is invalid");
            }
            descriptor.format = ModelFormat::SafeTensors;
            descriptor.capabilities = {Capability::ImageGeneration};
            descriptor.metadata["container"] = "SafeTensors";
        } else if ((std::memcmp(header.data(), "ggml", 4) == 0 || std::memcmp(header.data(), "lmgg", 4) == 0) &&
                   extension == ".bin") {
            descriptor.format = ModelFormat::WhisperGgml;
            descriptor.capabilities = {Capability::SpeechToText};
        } else if ((extension == ".tflite" || extension == ".litert") &&
                   std::memcmp(header.data() + 4, "TFL3", 4) == 0) {
            descriptor.format = ModelFormat::LiteRt;
            descriptor.metadata["container"] = "TFLite FlatBuffer";
            if (descriptor.displayName.find("mobilenet_v1_1.0_224") != std::string::npos) {
                descriptor.capabilities = {Capability::ImageClassification};
                descriptor.metadata["adapter"] = "MobileNet V1 224 Float";
            }
        } else if (extension == ".onnx") {
            descriptor.format = ModelFormat::Onnx;
            if (descriptor.displayName == "squeezenet1.1-7.onnx") {
                descriptor.capabilities = {Capability::ImageClassification};
                descriptor.metadata["adapter"] = "SqueezeNet 1.1 ImageNet";
            }
        } else if (extension == ".pt" && static_cast<unsigned char>(header[0]) == 0x50 &&
                   static_cast<unsigned char>(header[1]) == 0x4b) {
            descriptor.format = ModelFormat::TorchScript;
            const bool executable = containsBytes(path, "constants.pkl", descriptor.fileBytes);
            descriptor.metadata["torch.archive"] = executable ? "TorchScript candidate" : "Python checkpoint candidate";
            if (!executable) {
                descriptor.metadata["compatibility"] =
                    "Native execution requires an exported TorchScript module; a Python state dictionary is not executable";
            }
        } else {
            return Result<ModelDescriptor>::failure("model.format", "File signature is not a supported model container");
        }
    } else {
        return Result<ModelDescriptor>::failure("model.type", "Model path is neither a regular file nor a directory");
    }

    if (const auto name = descriptor.metadata.find("general.name"); name != descriptor.metadata.end()) {
        descriptor.displayName = name->second;
    }
    descriptor.estimatedRamBytes = descriptor.fileBytes + descriptor.fileBytes / 5;
    descriptor.id = makeId(descriptor.path, descriptor.fileBytes);
    return Result<ModelDescriptor>::success(std::move(descriptor));
}

ModelCatalog::ModelCatalog(std::filesystem::path databasePath) : databasePath_(std::move(databasePath)) {}

Result<ModelDescriptor> ModelCatalog::importModel(const std::filesystem::path& path) {
    auto inspected = ModelInspector::inspect(path);
    if (!inspected) return inspected;
    const auto found = std::ranges::find(models_, inspected.value().id, &ModelDescriptor::id);
    if (found == models_.end()) models_.push_back(inspected.value());
    const auto saved = save();
    if (!saved) return Result<ModelDescriptor>::failure(saved.error().code, saved.error().message);
    return inspected;
}

bool ModelCatalog::remove(std::string_view id) {
    const auto original = models_.size();
    std::erase_if(models_, [id](const ModelDescriptor& item) { return item.id == id; });
    if (models_.size() != original) {
        static_cast<void>(save());
        return true;
    }
    return false;
}

Result<bool> ModelCatalog::load() {
    models_.clear();
    std::ifstream input(databasePath_);
    if (!input) {
        return std::filesystem::exists(databasePath_)
                   ? Result<bool>::failure("catalog.open", "Unable to open model catalog")
                   : Result<bool>::success(true);
    }
    std::string storedPath;
    while (input >> std::quoted(storedPath)) {
        auto inspected = ModelInspector::inspect(storedPath);
        if (inspected) models_.push_back(std::move(inspected.value()));
    }
    return Result<bool>::success(true);
}

Result<bool> ModelCatalog::save() const {
    std::error_code error;
    std::filesystem::create_directories(databasePath_.parent_path(), error);
    if (error) return Result<bool>::failure("catalog.directory", error.message());
    std::ofstream output(databasePath_, std::ios::trunc);
    if (!output) return Result<bool>::failure("catalog.write", "Unable to save model catalog");
    for (const auto& model : models_) output << std::quoted(model.path.string()) << '\n';
    return Result<bool>::success(true);
}

} // namespace localai

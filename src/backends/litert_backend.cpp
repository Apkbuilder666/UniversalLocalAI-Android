#include "localai/backend.hpp"

#include "image_tensor.hpp"

#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h>

#include "localai/log.hpp"

#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace localai {
namespace {

class LiteRtApi {
public:
    using ModelCreateFromFile = decltype(&TfLiteModelCreateFromFile);
    using ModelDelete = decltype(&TfLiteModelDelete);
    using OptionsCreate = decltype(&TfLiteInterpreterOptionsCreate);
    using OptionsDelete = decltype(&TfLiteInterpreterOptionsDelete);
    using OptionsSetThreads = decltype(&TfLiteInterpreterOptionsSetNumThreads);
    using OptionsAddDelegate = decltype(&TfLiteInterpreterOptionsAddDelegate);
    using InterpreterCreate = decltype(&TfLiteInterpreterCreate);
    using InterpreterDelete = decltype(&TfLiteInterpreterDelete);
    using AllocateTensors = decltype(&TfLiteInterpreterAllocateTensors);
    using InputCount = decltype(&TfLiteInterpreterGetInputTensorCount);
    using OutputCount = decltype(&TfLiteInterpreterGetOutputTensorCount);
    using InputTensor = decltype(&TfLiteInterpreterGetInputTensor);
    using OutputTensor = decltype(&TfLiteInterpreterGetOutputTensor);
    using Invoke = decltype(&TfLiteInterpreterInvoke);
    using TensorType = decltype(&TfLiteTensorType);
    using TensorNumDims = decltype(&TfLiteTensorNumDims);
    using TensorDim = decltype(&TfLiteTensorDim);
    using TensorByteSize = decltype(&TfLiteTensorByteSize);
    using TensorCopyFrom = decltype(&TfLiteTensorCopyFromBuffer);
    using TensorCopyTo = decltype(&TfLiteTensorCopyToBuffer);
    using NnapiOptionsDefault = decltype(&TfLiteNnapiDelegateOptionsDefault);
    using NnapiCreate = decltype(&TfLiteNnapiDelegateCreate);
    using NnapiDelete = decltype(&TfLiteNnapiDelegateDelete);
    using GpuCreate = decltype(&TfLiteGpuDelegateV2Create);
    using GpuDelete = decltype(&TfLiteGpuDelegateV2Delete);

    static LiteRtApi& instance() {
        static LiteRtApi api;
        return api;
    }

    bool available() const { return handle_ != nullptr; }
    bool nnapiAvailable() const { return nnapiOptionsDefault && nnapiCreate && nnapiDelete; }
    bool gpuAvailable() const { return gpuHandle_ && gpuCreate && gpuDelete; }
    const std::string& error() const { return error_; }

    ModelCreateFromFile modelCreateFromFile{};
    ModelDelete modelDelete{};
    OptionsCreate optionsCreate{};
    OptionsDelete optionsDelete{};
    OptionsSetThreads optionsSetThreads{};
    OptionsAddDelegate optionsAddDelegate{};
    InterpreterCreate interpreterCreate{};
    InterpreterDelete interpreterDelete{};
    AllocateTensors allocateTensors{};
    InputCount inputCount{};
    OutputCount outputCount{};
    InputTensor inputTensor{};
    OutputTensor outputTensor{};
    Invoke invoke{};
    TensorType tensorType{};
    TensorNumDims tensorNumDims{};
    TensorDim tensorDim{};
    TensorByteSize tensorByteSize{};
    TensorCopyFrom tensorCopyFrom{};
    TensorCopyTo tensorCopyTo{};
    NnapiOptionsDefault nnapiOptionsDefault{};
    NnapiCreate nnapiCreate{};
    NnapiDelete nnapiDelete{};
    GpuCreate gpuCreate{};
    GpuDelete gpuDelete{};

private:
    LiteRtApi() {
        std::vector<std::string> candidates;
        const char* configured = std::getenv("LOCALAI_LITERT_LIBRARY");
        if (configured && *configured) {
            candidates.emplace_back(configured);
        } else {
#ifdef LOCALAI_LITERT_COMPILED_LIBRARY
            candidates.emplace_back(LOCALAI_LITERT_COMPILED_LIBRARY);
#endif
#if defined(_WIN32)
            candidates.emplace_back("tensorflowlite_c.dll");
            candidates.emplace_back("tensorflow-lite.dll");
#elif defined(__ANDROID__)
            candidates.emplace_back("libtensorflowlite_jni.so");
#elif defined(__APPLE__)
            candidates.emplace_back("libtensorflowlite_c.dylib");
            candidates.emplace_back("libtensorflow-lite.dylib");
#else
            candidates.emplace_back("libtensorflowlite_c.so");
            candidates.emplace_back("libtensorflow-lite.so");
#endif
        }
        for (const auto& candidate : candidates) {
            if (candidate.empty()) continue;
#if defined(_WIN32)
            const auto library = LoadLibraryW(std::filesystem::path(candidate).wstring().c_str());
            handle_ = reinterpret_cast<void*>(library);
#else
            int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
            flags |= RTLD_DEEPBIND;
#endif
            handle_ = dlopen(candidate.c_str(), flags);
#endif
            if (handle_ && resolveAll()) {
                resolveOptional(nnapiOptionsDefault, "TfLiteNnapiDelegateOptionsDefault");
                resolveOptional(nnapiCreate, "TfLiteNnapiDelegateCreate");
                resolveOptional(nnapiDelete, "TfLiteNnapiDelegateDelete");
                openGpuDelegate();
                return;
            }
#if defined(_WIN32)
            if (!handle_) error_ = "LoadLibrary failed for " + candidate + " with code " + std::to_string(GetLastError());
#else
            if (!handle_) {
                const char* reason = dlerror();
                error_ = "dlopen failed for " + candidate + (reason ? ": " + std::string(reason) : std::string{});
            }
#endif
            close();
        }
        if (error_.empty()) error_ = "LiteRT native library or required C API symbols were not available";
    }

    ~LiteRtApi() = default;
    LiteRtApi(const LiteRtApi&) = delete;
    LiteRtApi& operator=(const LiteRtApi&) = delete;

    void* symbol(const char* name) const {
#if defined(_WIN32)
        return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle_), name));
#else
        return dlsym(handle_, name);
#endif
    }

    template<class Function>
    bool resolve(Function& destination, const char* name) {
        destination = reinterpret_cast<Function>(symbol(name));
        if (!destination) error_ = "LiteRT C API symbol is missing: " + std::string(name);
        return destination != nullptr;
    }

    template<class Function>
    void resolveOptional(Function& destination, const char* name) {
        destination = reinterpret_cast<Function>(symbol(name));
    }

    void* gpuSymbol(const char* name) const {
#if defined(_WIN32)
        return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(gpuHandle_), name));
#else
        return gpuHandle_ ? dlsym(gpuHandle_, name) : nullptr;
#endif
    }

    void openGpuDelegate() {
        std::vector<std::string> candidates;
#ifdef LOCALAI_LITERT_GPU_LIBRARY
        candidates.emplace_back(LOCALAI_LITERT_GPU_LIBRARY);
#endif
#if defined(__ANDROID__)
        candidates.emplace_back("libtensorflowlite_gpu_jni.so");
#elif defined(_WIN32)
        candidates.emplace_back("tensorflowlite_gpu_delegate.dll");
#else
        candidates.emplace_back("libtensorflowlite_gpu_delegate.so");
#endif
        for (const auto& candidate : candidates) {
            if (candidate.empty()) continue;
#if defined(_WIN32)
            gpuHandle_ = reinterpret_cast<void*>(LoadLibraryW(std::filesystem::path(candidate).wstring().c_str()));
#else
            gpuHandle_ = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
            if (!gpuHandle_) continue;
            gpuCreate = reinterpret_cast<GpuCreate>(gpuSymbol("TfLiteGpuDelegateV2Create"));
            gpuDelete = reinterpret_cast<GpuDelete>(gpuSymbol("TfLiteGpuDelegateV2Delete"));
            if (gpuCreate && gpuDelete) return;
#if defined(_WIN32)
            FreeLibrary(reinterpret_cast<HMODULE>(gpuHandle_));
#else
            dlclose(gpuHandle_);
#endif
            gpuHandle_ = nullptr;
        }
    }

    bool resolveAll() {
        return resolve(modelCreateFromFile, "TfLiteModelCreateFromFile") &&
               resolve(modelDelete, "TfLiteModelDelete") &&
               resolve(optionsCreate, "TfLiteInterpreterOptionsCreate") &&
               resolve(optionsDelete, "TfLiteInterpreterOptionsDelete") &&
               resolve(optionsSetThreads, "TfLiteInterpreterOptionsSetNumThreads") &&
               resolve(optionsAddDelegate, "TfLiteInterpreterOptionsAddDelegate") &&
               resolve(interpreterCreate, "TfLiteInterpreterCreate") &&
               resolve(interpreterDelete, "TfLiteInterpreterDelete") &&
               resolve(allocateTensors, "TfLiteInterpreterAllocateTensors") &&
               resolve(inputCount, "TfLiteInterpreterGetInputTensorCount") &&
               resolve(outputCount, "TfLiteInterpreterGetOutputTensorCount") &&
               resolve(inputTensor, "TfLiteInterpreterGetInputTensor") &&
               resolve(outputTensor, "TfLiteInterpreterGetOutputTensor") &&
               resolve(invoke, "TfLiteInterpreterInvoke") &&
               resolve(tensorType, "TfLiteTensorType") &&
               resolve(tensorNumDims, "TfLiteTensorNumDims") &&
               resolve(tensorDim, "TfLiteTensorDim") &&
               resolve(tensorByteSize, "TfLiteTensorByteSize") &&
               resolve(tensorCopyFrom, "TfLiteTensorCopyFromBuffer") &&
               resolve(tensorCopyTo, "TfLiteTensorCopyToBuffer");
    }

    void close() {
        if (!handle_) return;
#if defined(_WIN32)
        FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
        dlclose(handle_);
#endif
        handle_ = nullptr;
    }

    void* handle_{};
    void* gpuHandle_{};
    std::string error_;
};

class LiteRtBackend final : public IModelBackend {
public:
    ~LiteRtBackend() override { unload(); }
    bool canLoad(const ModelDescriptor& model) const override { return model.format == ModelFormat::LiteRt; }

    Result<BackendInfo> load(const ModelDescriptor& model, const LoadOptions& options) override {
        unload();
        auto& api = LiteRtApi::instance();
        if (!api.available()) return Result<BackendInfo>::failure("litert.runtime", api.error());
        const auto started = std::chrono::steady_clock::now();
        model_ = api.modelCreateFromFile(model.path.string().c_str());
        if (!model_) return Result<BackendInfo>::failure("litert.load", "LiteRT rejected the FlatBuffer model");
        modelDescriptor_ = model;
        options_ = options;
        requestedDevice_ = options.deviceId;
        std::string failure;
        if (options.deviceId == "gpu") {
            if (!createInterpreter(Provider::Cpu, failure) || !validateSignature(failure)) {
                unload();
                return Result<BackendInfo>::failure("litert.signature", failure);
            }
            destroyInterpreter();
            deferredGpu_ = true;
            provider_ = "GPU delegate (initializes on inference worker)";
        } else if (options.deviceId != "cpu" && api.nnapiAvailable()) {
            if (!createInterpreter(Provider::Nnapi, failure)) {
                Logger::instance().write(LogLevel::Warning, LogCategory::Backend,
                    "LiteRT NNAPI initialization failed; trying the GPU delegate: " + failure);
                if (!createInterpreter(Provider::Cpu, failure) || !validateSignature(failure)) {
                    unload();
                    return Result<BackendInfo>::failure("litert.signature", failure);
                }
                destroyInterpreter();
                deferredGpu_ = api.gpuAvailable();
                if (deferredGpu_) provider_ = "GPU delegate (initializes on inference worker)";
                else if (!createInterpreter(Provider::Cpu, failure)) {
                    unload();
                    return Result<BackendInfo>::failure("litert.allocate", failure);
                }
            }
        } else if (!createInterpreter(Provider::Cpu, failure)) {
            unload();
            return Result<BackendInfo>::failure("litert.allocate", failure);
        }
        if (interpreter_ && !validateSignature(failure)) {
            unload();
            return Result<BackendInfo>::failure("litert.signature", failure);
        }
        stats_.loadMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return Result<BackendInfo>::success(backendInfo());
    }

    Result<InferenceOutput> infer(const InferenceRequest& request,
                                  const CancellationFlag& cancellation) override {
        if (!model_) return Result<InferenceOutput>::failure("litert.state", "No LiteRT model is loaded");
        if (deferredGpu_) {
            std::string failure;
            deferredGpu_ = false;
            if (!createInterpreter(Provider::Gpu, failure)) {
                Logger::instance().write(LogLevel::Warning, LogCategory::Backend,
                    "LiteRT GPU delegate initialization failed; falling back to CPU: " + failure);
                if (!createInterpreter(Provider::Cpu, failure))
                    return Result<InferenceOutput>::failure("litert.fallback", failure);
            }
        }
        if (!interpreter_) return Result<InferenceOutput>::failure("litert.state", "LiteRT interpreter initialization failed");
        std::vector<float> prepared;
        const std::vector<float>* tensor = &request.tensor;
        if (request.kind == TaskKind::ImageClassification) {
            if (!imageClassifier_ || !image_tensor::validRgb(request)) {
                return Result<InferenceOutput>::failure("litert.image", "This model does not match the supported MobileNet V1 RGB image signature");
            }
            prepared = image_tensor::mobilenetNhwc(request, 224, 224);
            tensor = &prepared;
        }
        if (tensor->empty()) return Result<InferenceOutput>::failure("litert.input", "Float tensor input is required");
        cancelled_.store(false);
        if (cancellation && cancellation->load()) return Result<InferenceOutput>::failure("task.cancelled", "Task cancelled");
        auto& api = LiteRtApi::instance();
        auto* input = api.inputTensor(interpreter_, 0);
        if (api.tensorType(input) != kTfLiteFloat32 || api.tensorByteSize(input) != tensor->size() * sizeof(float)) {
            return Result<InferenceOutput>::failure("litert.input", "Input must match the model's float32 tensor byte size");
        }
        if (api.tensorCopyFrom(input, tensor->data(), tensor->size() * sizeof(float)) != kTfLiteOk) {
            return Result<InferenceOutput>::failure("litert.copy", "Could not copy data into the model input tensor");
        }
        const auto started = std::chrono::steady_clock::now();
        if (api.invoke(interpreter_) != kTfLiteOk) {
            if (activeProvider_ != Provider::Cpu) {
                const auto failedProvider = provider_;
                std::string failure;
                Logger::instance().write(LogLevel::Warning, LogCategory::Backend,
                    "LiteRT " + failedProvider + " invocation failed; retrying on CPU");
                if (!createInterpreter(Provider::Cpu, failure))
                    return Result<InferenceOutput>::failure("litert.fallback", failure);
                return infer(request, cancellation);
            }
            return Result<InferenceOutput>::failure("litert.run", "LiteRT CPU invocation failed");
        }
        const auto finished = std::chrono::steady_clock::now();
        if (cancelled_.load() || (cancellation && cancellation->load())) {
            return Result<InferenceOutput>::failure("task.cancelled", "Task cancelled after the current LiteRT invocation completed");
        }
        const auto* outputTensor = api.outputTensor(interpreter_, 0);
        if (api.tensorType(outputTensor) != kTfLiteFloat32) {
            return Result<InferenceOutput>::failure("litert.output", "This adapter requires a float32 first output");
        }
        InferenceOutput output;
        output.tensor.resize(api.tensorByteSize(outputTensor) / sizeof(float));
        if (api.tensorCopyTo(outputTensor, output.tensor.data(), output.tensor.size() * sizeof(float)) != kTfLiteOk) {
            return Result<InferenceOutput>::failure("litert.copy", "Could not copy the output tensor");
        }
        const int dimensions = api.tensorNumDims(outputTensor);
        for (int index = 0; index < dimensions; ++index) output.tensorShape.push_back(api.tensorDim(outputTensor, index));
        if (request.kind == TaskKind::ImageClassification) {
            const std::size_t offset = output.tensor.size() == 1001 ? 1 : 0;
            output.text = image_tensor::topClasses(output.tensor, offset);
        }
        output.performance.totalMilliseconds = std::chrono::duration<double, std::milli>(finished - started).count();
        output.performance.inputUnits = request.kind == TaskKind::ImageClassification ? 1 : tensor->size();
        output.performance.outputUnits = output.tensor.size();
        output.performance.outputUnitsPerSecond = request.kind == TaskKind::ImageClassification
            ? 1000.0 / std::max(0.001, output.performance.totalMilliseconds) : 0.0;
        output.execution = {{"engine", "LiteRT"}, {"provider", provider_},
                            {"device", activeProvider_ == Provider::Cpu ? "CPU" : "Android runtime selected device"},
                            {"verification", activeProvider_ == Provider::Cpu
                                ? "LiteRT CPU interpreter completed invocation"
                                : "LiteRT delegate completed invocation; exact per-operation hardware placement is unavailable"}};
        stats_ = output.performance;
        return Result<InferenceOutput>::success(std::move(output));
    }

    void cancel() override { cancelled_.store(true); }
    void unload() override {
        if (!interpreter_ && !model_) {
            imageClassifier_ = false;
            return;
        }
        auto& api = LiteRtApi::instance();
        destroyInterpreter();
        if (model_ && api.available()) api.modelDelete(model_);
        model_ = nullptr;
        imageClassifier_ = false;
        deferredGpu_ = false;
        provider_ = "CPU";
    }
    BackendInfo backendInfo() const override {
        return {"litert", "LiteRT", "Google LiteRT", provider_,
                activeProvider_ == Provider::Cpu ? "CPU" : "Android accelerated delegate", {ModelFormat::LiteRt},
                imageClassifier_ ? std::vector{Capability::ImageClassification} : std::vector<Capability>{}, true, true,
                interpreter_ != nullptr || deferredGpu_, interpreter_ || deferredGpu_
                    ? "Model accepted; selected provider is " + provider_
                    : "Runtime compiled with CPU, NNAPI, and GPU delegate discovery; no model initialized"};
    }
    PerformanceStats statistics() const override { return stats_; }

private:
    enum class Provider { Cpu, Nnapi, Gpu };

    void destroyInterpreter() {
        auto& api = LiteRtApi::instance();
        if (interpreter_ && api.available()) api.interpreterDelete(interpreter_);
        interpreter_ = nullptr;
        if (delegate_) {
            if (activeProvider_ == Provider::Nnapi && api.nnapiDelete) api.nnapiDelete(delegate_);
            if (activeProvider_ == Provider::Gpu && api.gpuDelete) api.gpuDelete(delegate_);
        }
        delegate_ = nullptr;
    }

    bool createInterpreter(Provider provider, std::string& failure) {
        destroyInterpreter();
        auto& api = LiteRtApi::instance();
        auto* interpreterOptions = api.optionsCreate();
        if (!interpreterOptions) {
            failure = "LiteRT could not allocate interpreter options";
            return false;
        }
        api.optionsSetThreads(interpreterOptions, std::max(1, options_.threads));
        if (provider == Provider::Nnapi) {
            if (!api.nnapiAvailable()) {
                api.optionsDelete(interpreterOptions);
                failure = "NNAPI delegate symbols are unavailable";
                return false;
            }
            auto delegateOptions = api.nnapiOptionsDefault();
            delegateOptions.disallow_nnapi_cpu = 1;
            delegateOptions.allow_fp16 = 1;
            delegateOptions.execution_preference = TfLiteNnapiDelegateOptions::kSustainedSpeed;
            delegate_ = api.nnapiCreate(&delegateOptions);
        } else if (provider == Provider::Gpu) {
            if (!api.gpuAvailable()) {
                api.optionsDelete(interpreterOptions);
                failure = "GPU delegate library is unavailable";
                return false;
            }
            delegate_ = api.gpuCreate(nullptr);
        }
        if (provider != Provider::Cpu) {
            if (!delegate_) {
                api.optionsDelete(interpreterOptions);
                failure = "The requested LiteRT delegate rejected this device";
                return false;
            }
            api.optionsAddDelegate(interpreterOptions, delegate_);
        }
        activeProvider_ = provider;
        interpreter_ = api.interpreterCreate(model_, interpreterOptions);
        api.optionsDelete(interpreterOptions);
        if (!interpreter_ || api.allocateTensors(interpreter_) != kTfLiteOk) {
            destroyInterpreter();
            failure = "LiteRT could not create the interpreter or allocate tensors with the selected provider";
            return false;
        }
        provider_ = provider == Provider::Nnapi ? "NNAPI delegate" :
                    provider == Provider::Gpu ? "GPU delegate" : "CPU";
        return true;
    }

    bool validateSignature(std::string& failure) {
        auto& api = LiteRtApi::instance();
        if (!interpreter_ || api.inputCount(interpreter_) != 1 || api.outputCount(interpreter_) < 1) {
            failure = "This adapter accepts models with exactly one numeric input and at least one output";
            return false;
        }
        const auto* input = api.inputTensor(interpreter_, 0);
        imageClassifier_ = modelDescriptor_.path.filename() == "mobilenet_v1_1.0_224.tflite" &&
                           api.tensorType(input) == kTfLiteFloat32 && api.tensorNumDims(input) == 4 &&
                           api.tensorDim(input, 0) == 1 && api.tensorDim(input, 1) == 224 &&
                           api.tensorDim(input, 2) == 224 && api.tensorDim(input, 3) == 3;
        return true;
    }

    TfLiteModel* model_{};
    TfLiteInterpreter* interpreter_{};
    TfLiteDelegate* delegate_{};
    ModelDescriptor modelDescriptor_;
    LoadOptions options_;
    std::string requestedDevice_;
    std::string provider_{"CPU"};
    Provider activeProvider_{Provider::Cpu};
    bool deferredGpu_{};
    PerformanceStats stats_;
    std::atomic_bool cancelled_{};
    bool imageClassifier_{};
};

}

void registerLiteRtBackend() {
    BackendRegistry::instance().registerBackend("litert", [] { return std::make_unique<LiteRtBackend>(); });
}

} // namespace localai

#include "localai/backend.hpp"

#include "image_tensor.hpp"

#include <onnxruntime_cxx_api.h>
#ifdef __ANDROID__
#include <nnapi_provider_factory.h>
#endif

#include <chrono>
#include <algorithm>
#include <numeric>

#include "localai/log.hpp"

namespace localai {
namespace {

class OnnxBackend final : public IModelBackend {
public:
    bool canLoad(const ModelDescriptor& model) const override { return model.format == ModelFormat::Onnx; }

    Result<BackendInfo> load(const ModelDescriptor& model, const LoadOptions& options) override {
        unload();
        model_ = model;
        options_ = options;
        try {
            const auto started = std::chrono::steady_clock::now();
            const bool acceleratorRequested = options.deviceId != "cpu";
            bool opened{};
#ifdef __ANDROID__
            if (acceleratorRequested) {
                try {
                    initializeSession(true);
                    opened = true;
                } catch (const Ort::Exception& exception) {
                    Logger::instance().write(LogLevel::Warning, LogCategory::Backend,
                        "ONNX Runtime NNAPI initialization failed; falling back to CPU: " +
                        std::string(exception.what()));
                }
            }
#else
            static_cast<void>(acceleratorRequested);
#endif
            if (!opened) initializeSession(false);
            stats_.loadMilliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            info_ = backendInfo();
            info_.initialized = true;
            info_.verification = provider_ == "NnapiExecutionProvider"
                ? "NNAPI execution provider accepted the model; exact accelerator placement is unavailable"
                : "ONNX Runtime session initialized with CPUExecutionProvider";
            return Result<BackendInfo>::success(info_);
        } catch (const Ort::Exception& exception) {
            unload();
            return Result<BackendInfo>::failure("onnx.load", exception.what());
        }
    }

    Result<InferenceOutput> infer(const InferenceRequest& request,
                                  const CancellationFlag& cancellation) override {
        if (!session_) return Result<InferenceOutput>::failure("onnx.state", "No ONNX model is loaded");
        std::vector<float> prepared;
        std::vector<std::int64_t> preparedShape;
        const std::vector<float>* tensor = &request.tensor;
        const std::vector<std::int64_t>* shape = &request.tensorShape;
        if (request.kind == TaskKind::ImageClassification) {
            if (!imageClassifier_ || !image_tensor::validRgb(request)) {
                std::string dimensions;
                for (const auto value : inputShape_) dimensions += (dimensions.empty() ? "" : "x") + std::to_string(value);
                return Result<InferenceOutput>::failure("onnx.image",
                    "This model does not match the supported SqueezeNet RGB image signature; runtime input type is " +
                    std::to_string(static_cast<int>(inputElementType_)) + " and shape is " + dimensions);
            }
            prepared = image_tensor::imagenetNchw(request, 224, 224);
            preparedShape = {1, 3, 224, 224};
            tensor = &prepared;
            shape = &preparedShape;
        }
        if (tensor->empty() || shape->empty()) {
            return Result<InferenceOutput>::failure("onnx.input", "Numeric tensor data and shape are required");
        }
        cancelled_.store(false);
        runOptions_.UnsetTerminate();
        if (cancellation && cancellation->load()) return Result<InferenceOutput>::failure("task.cancelled", "Task cancelled");
        const auto elements = std::accumulate(shape->begin(), shape->end(), std::int64_t{1},
                                              std::multiplies<>());
        if (elements <= 0 || static_cast<std::size_t>(elements) != tensor->size()) {
            return Result<InferenceOutput>::failure("onnx.shape", "Tensor shape does not match the supplied element count");
        }
        try {
            auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto input = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(tensor->data()),
                                                         tensor->size(), shape->data(), shape->size());
            const char* inputNames[] = {inputName_.c_str()};
            const char* outputNames[] = {outputName_.c_str()};
            const auto started = std::chrono::steady_clock::now();
            auto outputs = session_->Run(runOptions_, inputNames, &input, 1, outputNames, 1);
            const auto finished = std::chrono::steady_clock::now();
            if (cancelled_.load() || (cancellation && cancellation->load())) {
                return Result<InferenceOutput>::failure("task.cancelled", "Task cancelled");
            }
            if (!outputs.front().IsTensor()) {
                return Result<InferenceOutput>::failure("onnx.output", "First model output is not a tensor");
            }
            auto outputInfo = outputs.front().GetTensorTypeAndShapeInfo();
            if (outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                return Result<InferenceOutput>::failure("onnx.output_type", "This adapter requires a float32 first output");
            }
            InferenceOutput result;
            result.tensorShape = outputInfo.GetShape();
            const auto count = outputInfo.GetElementCount();
            const auto* data = outputs.front().GetTensorData<float>();
            result.tensor.assign(data, data + count);
            if (request.kind == TaskKind::ImageClassification) result.text = image_tensor::topClasses(result.tensor);
            result.performance.totalMilliseconds = std::chrono::duration<double, std::milli>(finished - started).count();
            result.performance.inputUnits = request.kind == TaskKind::ImageClassification ? 1 : tensor->size();
            result.performance.outputUnits = count;
            result.performance.outputUnitsPerSecond = request.kind == TaskKind::ImageClassification
                ? 1000.0 / std::max(0.001, result.performance.totalMilliseconds) : 0.0;
            result.execution = {{"engine", "ONNX Runtime"}, {"provider", provider_},
                                {"device", provider_ == "NnapiExecutionProvider" ? "Android NNAPI selected devices" : "CPU"},
                                {"verification", provider_ == "NnapiExecutionProvider"
                                    ? "NNAPI provider executed successfully; exact per-operation GPU/NPU placement is not exposed"
                                    : "CPUExecutionProvider executed successfully"}};
            stats_ = result.performance;
            return Result<InferenceOutput>::success(std::move(result));
        } catch (const Ort::Exception& exception) {
            if (provider_ == "NnapiExecutionProvider") {
                Logger::instance().write(LogLevel::Warning, LogCategory::Backend,
                    "ONNX Runtime NNAPI inference failed; retrying on CPU: " + std::string(exception.what()));
                try {
                    initializeSession(false);
                    return infer(request, cancellation);
                } catch (const Ort::Exception& fallbackException) {
                    return Result<InferenceOutput>::failure("onnx.fallback", fallbackException.what());
                }
            }
            return Result<InferenceOutput>::failure("onnx.run", exception.what());
        }
    }

    void cancel() override { cancelled_.store(true); runOptions_.SetTerminate(); }
    void unload() override {
        session_.reset();
        inputName_.clear();
        outputName_.clear();
        inputShape_.clear();
        inputElementType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        imageClassifier_ = false;
        provider_ = "CPUExecutionProvider";
        info_.initialized = false;
    }
    BackendInfo backendInfo() const override {
        BackendInfo info{"onnxruntime", "ONNX Runtime", "ONNX Runtime", provider_,
                         provider_ == "NnapiExecutionProvider" ? "Android NNAPI" : "CPU",
                         {ModelFormat::Onnx}, imageClassifier_ ? std::vector{Capability::ImageClassification} : std::vector<Capability>{}, true, true, session_ != nullptr,
                         session_ ? "Session initialized" :
#ifdef __ANDROID__
                         "Runtime compiled with NNAPI and CPU providers; no model initialized"
#else
                         "Runtime compiled; no model initialized"
#endif
        };
        return info;
    }
    PerformanceStats statistics() const override { return stats_; }

private:
    void initializeSession(bool nnapi) {
        session_.reset();
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(std::clamp(options_.threads, 1, 64));
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef __ANDROID__
        if (nnapi) {
            const std::uint32_t flags = NNAPI_FLAG_CPU_DISABLED | NNAPI_FLAG_USE_FP16;
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions, flags));
            provider_ = "NnapiExecutionProvider";
        } else
#else
        static_cast<void>(nnapi);
#endif
        {
            provider_ = "CPUExecutionProvider";
        }
#ifdef _WIN32
        session_ = std::make_unique<Ort::Session>(environment(), model_.path.wstring().c_str(), sessionOptions);
#else
        session_ = std::make_unique<Ort::Session>(environment(), model_.path.string().c_str(), sessionOptions);
#endif
        Ort::AllocatorWithDefaultOptions allocator;
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() < 1) {
            session_.reset();
            throw Ort::Exception(std::string("This adapter requires exactly one numeric input and at least one output"),
                                 ORT_INVALID_GRAPH);
        }
        inputName_ = session_->GetInputNameAllocated(0, allocator).get();
        outputName_ = session_->GetOutputNameAllocated(0, allocator).get();
        const auto inputTypeInfo = session_->GetInputTypeInfo(0);
        const auto inputInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        inputShape_ = inputInfo.GetShape();
        inputElementType_ = inputInfo.GetElementType();
        imageClassifier_ = inputElementType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                           (inputShape_.empty() || inputShape_ == std::vector<std::int64_t>({1, 3, 224, 224})) &&
                           model_.path.filename() == "squeezenet1.1-7.onnx";
    }

    static Ort::Env& environment() {
        static Ort::Env value(ORT_LOGGING_LEVEL_WARNING, "universal-local-ai");
        return value;
    }
    std::unique_ptr<Ort::Session> session_;
    ModelDescriptor model_;
    LoadOptions options_;
    std::string provider_{"CPUExecutionProvider"};
    std::string inputName_;
    std::string outputName_;
    std::vector<std::int64_t> inputShape_;
    ONNXTensorElementDataType inputElementType_{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    bool imageClassifier_{};
    BackendInfo info_;
    PerformanceStats stats_;
    Ort::RunOptions runOptions_;
    std::atomic_bool cancelled_{};
};

}

void registerOnnxBackend() {
    BackendRegistry::instance().registerBackend("onnxruntime", [] { return std::make_unique<OnnxBackend>(); });
}

} // namespace localai

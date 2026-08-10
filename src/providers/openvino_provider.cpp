#include "localai/backend.hpp"

#include <openvino/openvino.hpp>

#include <chrono>
#include <numeric>

namespace localai {
namespace {

class OpenVinoBackend final : public IModelBackend {
public:
    explicit OpenVinoBackend(std::string device) : device_(std::move(device)) {}
    bool canLoad(const ModelDescriptor& model) const override { return model.format == ModelFormat::Onnx; }

    Result<BackendInfo> load(const ModelDescriptor& model, const LoadOptions& options) override {
        static_cast<void>(options);
        unload();
        try {
            const auto started = std::chrono::steady_clock::now();
            auto network = core().read_model(model.path.string());
            if (network->inputs().size() != 1 || network->outputs().empty()) {
                return Result<BackendInfo>::failure(
                    "openvino.signature", "This adapter accepts models with exactly one numeric input and at least one output");
            }
            compiled_ = std::make_unique<ov::CompiledModel>(core().compile_model(network, device_));
            request_ = std::make_unique<ov::InferRequest>(compiled_->create_infer_request());
            stats_.loadMilliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            return Result<BackendInfo>::success(backendInfo());
        } catch (const std::exception& exception) {
            unload();
            return Result<BackendInfo>::failure("openvino.load", exception.what());
        }
    }

    Result<InferenceOutput> infer(const InferenceRequest& input,
                                  const CancellationFlag& cancellation) override {
        if (!request_ || !compiled_) return Result<InferenceOutput>::failure("openvino.state", "No model is loaded");
        if (input.tensor.empty() || input.tensorShape.empty()) {
            return Result<InferenceOutput>::failure("openvino.input", "Float tensor data and shape are required");
        }
        if (cancellation && cancellation->load()) return Result<InferenceOutput>::failure("task.cancelled", "Task cancelled");
        try {
            ov::Shape shape;
            for (const auto dimension : input.tensorShape) {
                if (dimension <= 0) return Result<InferenceOutput>::failure("openvino.shape", "Tensor dimensions must be positive");
                shape.push_back(static_cast<std::size_t>(dimension));
            }
            const auto elements = ov::shape_size(shape);
            if (elements != input.tensor.size()) {
                return Result<InferenceOutput>::failure("openvino.shape", "Tensor shape does not match the supplied element count");
            }
            ov::Tensor tensor(ov::element::f32, shape, const_cast<float*>(input.tensor.data()));
            request_->set_input_tensor(tensor);
            const auto started = std::chrono::steady_clock::now();
            request_->infer();
            const auto finished = std::chrono::steady_clock::now();
            if (cancellation && cancellation->load()) return Result<InferenceOutput>::failure("task.cancelled", "Task cancelled");
            const auto resultTensor = request_->get_output_tensor();
            if (resultTensor.get_element_type() != ov::element::f32) {
                return Result<InferenceOutput>::failure("openvino.output", "This adapter requires a float32 first output");
            }
            InferenceOutput output;
            for (const auto dimension : resultTensor.get_shape()) output.tensorShape.push_back(static_cast<std::int64_t>(dimension));
            const auto* data = resultTensor.data<const float>();
            output.tensor.assign(data, data + resultTensor.get_size());
            output.performance.totalMilliseconds = std::chrono::duration<double, std::milli>(finished - started).count();
            output.execution = {{"engine", "OpenVINO"}, {"provider", device_},
                                {"verification", "OpenVINO compiled and executed the model for the selected device"}};
            stats_ = output.performance;
            return Result<InferenceOutput>::success(std::move(output));
        } catch (const std::exception& exception) {
            return Result<InferenceOutput>::failure("openvino.run", exception.what());
        }
    }

    void cancel() override {
        if (request_) request_->cancel();
    }
    void unload() override { request_.reset(); compiled_.reset(); }
    BackendInfo backendInfo() const override {
        const bool npu = device_.find("NPU") != std::string::npos;
        const bool gpu = device_.find("GPU") != std::string::npos;
        return {"openvino:" + device_, "OpenVINO " + device_, "OpenVINO", device_, device_, {ModelFormat::Onnx}, {},
                true, true, compiled_ != nullptr,
                compiled_ ? "Compiled model initialized for " + device_
                          : (npu ? "Intel NPU enumerated by OpenVINO" : gpu ? "GPU enumerated by OpenVINO" : "Device enumerated by OpenVINO")};
    }
    PerformanceStats statistics() const override { return stats_; }

private:
    static ov::Core& core() {
        static ov::Core instance;
        return instance;
    }
    std::string device_;
    std::unique_ptr<ov::CompiledModel> compiled_;
    std::unique_ptr<ov::InferRequest> request_;
    PerformanceStats stats_;
};

}

void registerOpenVinoProvider() {
    static ov::Core core;
    for (const auto& device : core.get_available_devices()) {
        BackendRegistry::instance().registerBackend("openvino:" + device,
            [device] { return std::make_unique<OpenVinoBackend>(device); });
    }
}

} // namespace localai

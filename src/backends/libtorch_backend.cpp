#include "localai/backend.hpp"

#include <torch/script.h>

#include <chrono>
#include <numeric>

namespace localai {
namespace {

class LibTorchBackend final : public IModelBackend {
public:
    bool canLoad(const ModelDescriptor& model) const override {
        const auto kind = model.metadata.find("torch.archive");
        return model.format == ModelFormat::TorchScript && kind != model.metadata.end() &&
               kind->second == "TorchScript candidate";
    }

    Result<BackendInfo> load(const ModelDescriptor& model, const LoadOptions& options) override {
        static_cast<void>(options);
        unload();
        try {
            const auto started = std::chrono::steady_clock::now();
            module_ = std::make_unique<torch::jit::Module>(torch::jit::load(model.path.string(), torch::kCPU));
            module_->eval();
            stats_.loadMilliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            return Result<BackendInfo>::success(backendInfo());
        } catch (const c10::Error& exception) {
            unload();
            return Result<BackendInfo>::failure(
                "torch.load", std::string("LibTorch rejected this archive as executable TorchScript: ") + exception.what());
        }
    }

    Result<InferenceOutput> infer(const InferenceRequest& request,
                                  const CancellationFlag& cancellation) override {
        if (!module_) return Result<InferenceOutput>::failure("torch.state", "No TorchScript module is loaded");
        if (request.tensor.empty() || request.tensorShape.empty()) {
            return Result<InferenceOutput>::failure("torch.input", "Numeric tensor data and shape are required");
        }
        cancelled_.store(false);
        if (cancellation && cancellation->load()) return Result<InferenceOutput>::failure("task.cancelled", "Task cancelled");
        const auto elements = std::accumulate(request.tensorShape.begin(), request.tensorShape.end(), std::int64_t{1},
                                              std::multiplies<>());
        if (elements <= 0 || static_cast<std::size_t>(elements) != request.tensor.size()) {
            return Result<InferenceOutput>::failure("torch.shape", "Tensor shape does not match the supplied element count");
        }
        try {
            auto input = torch::from_blob(const_cast<float*>(request.tensor.data()), request.tensorShape,
                                          torch::TensorOptions().dtype(torch::kFloat32)).clone();
            const auto started = std::chrono::steady_clock::now();
            auto value = module_->forward({input});
            const auto finished = std::chrono::steady_clock::now();
            if (cancelled_.load() || (cancellation && cancellation->load())) {
                return Result<InferenceOutput>::failure("task.cancelled", "Task cancelled after the current TorchScript operation completed");
            }
            if (!value.isTensor()) {
                return Result<InferenceOutput>::failure("torch.output", "Module output is not a tensor");
            }
            auto output = value.toTensor().to(torch::kCPU).to(torch::kFloat32).contiguous();
            InferenceOutput result;
            result.tensorShape.assign(output.sizes().begin(), output.sizes().end());
            result.tensor.assign(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
            result.performance.totalMilliseconds = std::chrono::duration<double, std::milli>(finished - started).count();
            result.execution = {{"engine", "LibTorch"}, {"provider", "CPU"},
                                {"verification", "TorchScript module executed on a CPU tensor"}};
            stats_ = result.performance;
            return Result<InferenceOutput>::success(std::move(result));
        } catch (const c10::Error& exception) {
            return Result<InferenceOutput>::failure("torch.run", exception.what());
        }
    }

    void cancel() override { cancelled_.store(true); }
    void unload() override { module_.reset(); }
    BackendInfo backendInfo() const override {
        return {"libtorch", "LibTorch", "PyTorch C++", "CPU", "CPU", {ModelFormat::TorchScript}, {},
                true, true, module_ != nullptr,
                module_ ? "TorchScript module initialized" : "Runtime compiled; no model initialized"};
    }
    PerformanceStats statistics() const override { return stats_; }

private:
    std::unique_ptr<torch::jit::Module> module_;
    PerformanceStats stats_;
    std::atomic_bool cancelled_{};
};

}

void registerLibTorchBackend() {
    BackendRegistry::instance().registerBackend("libtorch", [] { return std::make_unique<LibTorchBackend>(); });
}

} // namespace localai

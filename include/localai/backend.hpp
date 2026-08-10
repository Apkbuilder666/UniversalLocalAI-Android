#pragma once

#include "localai/types.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace localai {

class IModelBackend {
public:
    virtual ~IModelBackend() = default;
    virtual bool canLoad(const ModelDescriptor& model) const = 0;
    virtual Result<BackendInfo> load(const ModelDescriptor& model, const LoadOptions& options) = 0;
    virtual Result<InferenceOutput> infer(const InferenceRequest& request,
                                          const CancellationFlag& cancellation) = 0;
    virtual void cancel() = 0;
    virtual void unload() = 0;
    virtual BackendInfo backendInfo() const = 0;
    virtual PerformanceStats statistics() const = 0;
};

using BackendFactory = std::function<std::unique_ptr<IModelBackend>()>;

class BackendRegistry {
public:
    static BackendRegistry& instance();
    void registerBackend(std::string id, BackendFactory factory);
    std::vector<std::string> ids() const;
    std::unique_ptr<IModelBackend> create(const std::string& id) const;
    std::vector<BackendInfo> probe() const;
    std::vector<std::string> compatible(const ModelDescriptor& model) const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, BackendFactory> factories_;
};

void registerCompiledBackends();

} // namespace localai

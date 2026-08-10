#pragma once

#include "localai/types.hpp"

#include <filesystem>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

namespace localai {

struct BenchmarkRecord {
    std::string modelId;
    std::string backendId;
    std::string provider;
    std::string device;
    PerformanceStats performance;
};

class BenchmarkStore {
public:
    explicit BenchmarkStore(std::filesystem::path path);
    Result<bool> load();
    Result<bool> save() const;
    void record(BenchmarkRecord value);
    std::vector<BenchmarkRecord> forModel(std::string_view modelId) const;
    std::optional<std::string> fastestBackend(std::string_view modelId,
                                               const std::vector<std::string>& candidates) const;

private:
    std::filesystem::path path_;
    mutable std::shared_mutex mutex_;
    std::vector<BenchmarkRecord> records_;
};

} // namespace localai

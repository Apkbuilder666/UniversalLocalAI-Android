#include "localai/benchmark.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <ranges>

namespace localai {

BenchmarkStore::BenchmarkStore(std::filesystem::path path) : path_(std::move(path)) {}

Result<bool> BenchmarkStore::load() {
    std::ifstream input(path_);
    if (!input) {
        if (!std::filesystem::exists(path_)) return Result<bool>::success(true);
        return Result<bool>::failure("benchmark.open", "Unable to open benchmark database");
    }
    std::vector<BenchmarkRecord> loaded;
    BenchmarkRecord value;
    while (input >> std::quoted(value.modelId) >> std::quoted(value.backendId) >>
           std::quoted(value.provider) >> std::quoted(value.device) >>
           value.performance.loadMilliseconds >> value.performance.firstOutputMilliseconds >>
           value.performance.totalMilliseconds >> value.performance.promptUnitsPerSecond >>
           value.performance.outputUnitsPerSecond >> value.performance.inputUnits >> value.performance.outputUnits) {
        loaded.push_back(value);
    }
    if (!input.eof()) return Result<bool>::failure("benchmark.parse", "Benchmark database contains malformed data");
    std::unique_lock lock(mutex_);
    records_ = std::move(loaded);
    return Result<bool>::success(true);
}

Result<bool> BenchmarkStore::save() const {
    std::shared_lock lock(mutex_);
    std::error_code error;
    std::filesystem::create_directories(path_.parent_path(), error);
    if (error) return Result<bool>::failure("benchmark.directory", error.message());
    const auto temporary = path_.string() + ".new";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) return Result<bool>::failure("benchmark.write", "Unable to create benchmark database");
    output << std::setprecision(17);
    for (const auto& value : records_) {
        output << std::quoted(value.modelId) << ' ' << std::quoted(value.backendId) << ' '
               << std::quoted(value.provider) << ' ' << std::quoted(value.device) << ' '
               << value.performance.loadMilliseconds << ' ' << value.performance.firstOutputMilliseconds << ' '
               << value.performance.totalMilliseconds << ' ' << value.performance.promptUnitsPerSecond << ' '
               << value.performance.outputUnitsPerSecond << ' ' << value.performance.inputUnits << ' '
               << value.performance.outputUnits << '\n';
    }
    output.close();
    std::filesystem::rename(temporary, path_, error);
    if (error) {
        std::filesystem::remove(path_, error);
        error.clear();
        std::filesystem::rename(temporary, path_, error);
    }
    return error ? Result<bool>::failure("benchmark.commit", error.message()) : Result<bool>::success(true);
}

void BenchmarkStore::record(BenchmarkRecord value) {
    std::unique_lock lock(mutex_);
    const auto match = [&](const BenchmarkRecord& item) {
        return item.modelId == value.modelId && item.backendId == value.backendId &&
               item.provider == value.provider && item.device == value.device;
    };
    const auto found = std::ranges::find_if(records_, match);
    if (found == records_.end()) records_.push_back(std::move(value));
    else *found = std::move(value);
}

std::vector<BenchmarkRecord> BenchmarkStore::forModel(std::string_view modelId) const {
    std::shared_lock lock(mutex_);
    std::vector<BenchmarkRecord> result;
    std::ranges::copy_if(records_, std::back_inserter(result),
                         [&](const BenchmarkRecord& value) { return value.modelId == modelId; });
    return result;
}

std::optional<std::string> BenchmarkStore::fastestBackend(
    std::string_view modelId, const std::vector<std::string>& candidates) const {
    std::shared_lock lock(mutex_);
    const BenchmarkRecord* best{};
    for (const auto& value : records_) {
        if (value.modelId != modelId || value.performance.outputUnitsPerSecond <= 0.0 ||
            std::ranges::find(candidates, value.backendId) == candidates.end()) continue;
        if (!best || value.performance.outputUnitsPerSecond > best->performance.outputUnitsPerSecond) best = &value;
    }
    return best ? std::optional<std::string>(best->backendId) : std::nullopt;
}

} // namespace localai

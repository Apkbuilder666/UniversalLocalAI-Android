#pragma once

#include "localai/types.hpp"

#include <filesystem>
#include <map>
#include <shared_mutex>
#include <string>

namespace localai {

class Settings {
public:
    explicit Settings(std::filesystem::path path);
    Result<bool> load();
    Result<bool> save() const;
    std::string get(std::string_view key, std::string fallback = {}) const;
    int getInt(std::string_view key, int fallback) const;
    bool getBool(std::string_view key, bool fallback) const;
    void set(std::string key, std::string value);

private:
    std::filesystem::path path_;
    mutable std::shared_mutex mutex_;
    std::map<std::string, std::string> values_;
};

std::filesystem::path applicationDataDirectory();

} // namespace localai

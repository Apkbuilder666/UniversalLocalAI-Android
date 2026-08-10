#include "localai/settings.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>

namespace localai {

Settings::Settings(std::filesystem::path path) : path_(std::move(path)) {}

Result<bool> Settings::load() {
    std::ifstream input(path_);
    if (!input) {
        if (!std::filesystem::exists(path_)) {
            return Result<bool>::success(true);
        }
        return Result<bool>::failure("settings.open", "Unable to open settings file");
    }
    std::map<std::string, std::string> loaded;
    std::string key;
    std::string value;
    while (input >> std::quoted(key) >> std::quoted(value)) {
        loaded.insert_or_assign(std::move(key), std::move(value));
    }
    if (!input.eof()) {
        return Result<bool>::failure("settings.parse", "Settings file contains malformed data");
    }
    std::unique_lock lock(mutex_);
    values_ = std::move(loaded);
    return Result<bool>::success(true);
}

Result<bool> Settings::save() const {
    std::shared_lock lock(mutex_);
    std::error_code error;
    std::filesystem::create_directories(path_.parent_path(), error);
    if (error) {
        return Result<bool>::failure("settings.directory", error.message());
    }
    const auto temporary = path_.string() + ".new";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) {
        return Result<bool>::failure("settings.write", "Unable to create settings file");
    }
    for (const auto& [key, value] : values_) {
        output << std::quoted(key) << ' ' << std::quoted(value) << '\n';
    }
    output.close();
    std::filesystem::rename(temporary, path_, error);
    if (error) {
        std::filesystem::remove(path_, error);
        error.clear();
        std::filesystem::rename(temporary, path_, error);
    }
    return error ? Result<bool>::failure("settings.commit", error.message())
                 : Result<bool>::success(true);
}

std::string Settings::get(std::string_view key, std::string fallback) const {
    std::shared_lock lock(mutex_);
    const auto found = values_.find(std::string(key));
    return found == values_.end() ? std::move(fallback) : found->second;
}

int Settings::getInt(std::string_view key, int fallback) const {
    try {
        return std::stoi(get(key));
    } catch (...) {
        return fallback;
    }
}

bool Settings::getBool(std::string_view key, bool fallback) const {
    const auto value = get(key);
    if (value == "true" || value == "1") return true;
    if (value == "false" || value == "0") return false;
    return fallback;
}

void Settings::set(std::string key, std::string value) {
    std::unique_lock lock(mutex_);
    values_.insert_or_assign(std::move(key), std::move(value));
}

std::filesystem::path applicationDataDirectory() {
#ifdef __ANDROID__
    if (const char* temporary = std::getenv("TMPDIR")) {
        const std::filesystem::path cache(temporary);
        if (cache.filename() == "cache" && !cache.parent_path().empty()) return cache.parent_path() / "files";
    }
    std::ifstream processName("/proc/self/cmdline", std::ios::binary);
    std::string package;
    std::getline(processName, package, '\0');
    if (!package.empty() && package.find('/') == std::string::npos) {
        return std::filesystem::path("/data/data") / package / "files";
    }
    return std::filesystem::temp_directory_path() / "universal-local-ai";
#elif defined(_WIN32)
    if (const char* value = std::getenv("LOCALAPPDATA")) {
        return std::filesystem::path(value) / "UniversalLocalAI";
    }
#elif defined(__APPLE__)
    if (const char* value = std::getenv("HOME")) {
        return std::filesystem::path(value) / "Library" / "Application Support" / "UniversalLocalAI";
    }
#else
    if (const char* value = std::getenv("XDG_DATA_HOME")) {
        return std::filesystem::path(value) / "universal-local-ai";
    }
    if (const char* value = std::getenv("HOME")) {
        return std::filesystem::path(value) / ".local" / "share" / "universal-local-ai";
    }
#endif
    return std::filesystem::temp_directory_path() / "universal-local-ai";
}

} // namespace localai

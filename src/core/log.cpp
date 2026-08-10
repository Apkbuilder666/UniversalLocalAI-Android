#include "localai/log.hpp"

#include <iomanip>
#include <sstream>

namespace localai {
namespace {
std::string escapeJson(std::string_view input) {
    std::string result;
    result.reserve(input.size());
    for (const char value : input) {
        switch (value) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result += value;
        }
    }
    return result;
}
}

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::setFile(const std::filesystem::path& path) {
    std::scoped_lock lock(mutex_);
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    file_.close();
    file_.open(path, std::ios::app);
}

void Logger::write(LogLevel level, LogCategory category, std::string message) {
    const LogEntry entry{std::chrono::system_clock::now(), level, category, std::move(message)};
    std::function<void(const LogEntry&)> listener;
    {
        std::scoped_lock lock(mutex_);
        entries_.push_back(entry);
        if (entries_.size() > 10000) {
            entries_.erase(entries_.begin(), entries_.begin() + 1000);
        }
        if (file_) {
            const auto value = std::chrono::system_clock::to_time_t(entry.timestamp);
            std::tm time{};
#ifdef _WIN32
            gmtime_s(&time, &value);
#else
            gmtime_r(&value, &time);
#endif
            file_ << "{\"time\":\"" << std::put_time(&time, "%FT%TZ")
                  << "\",\"level\":\"" << toString(level)
                  << "\",\"category\":\"" << toString(category)
                  << "\",\"message\":\"" << escapeJson(entry.message) << "\"}\n";
            file_.flush();
        }
        listener = listener_;
    }
    if (listener) {
        listener(entry);
    }
}

std::vector<LogEntry> Logger::entries() const {
    std::scoped_lock lock(mutex_);
    return entries_;
}

void Logger::setListener(std::function<void(const LogEntry&)> listener) {
    std::scoped_lock lock(mutex_);
    listener_ = std::move(listener);
}

std::string toString(LogLevel value) {
    switch (value) {
    case LogLevel::Debug: return "DEBUG";
    case LogLevel::Info: return "INFO";
    case LogLevel::Warning: return "WARNING";
    case LogLevel::Error: return "ERROR";
    }
    return "UNKNOWN";
}

std::string toString(LogCategory value) {
    switch (value) {
    case LogCategory::App: return "APP";
    case LogCategory::Model: return "MODEL";
    case LogCategory::Inference: return "INFERENCE";
    case LogCategory::Hardware: return "HARDWARE";
    case LogCategory::Backend: return "BACKEND";
    case LogCategory::Memory: return "MEMORY";
    case LogCategory::Image: return "IMAGE";
    case LogCategory::Audio: return "AUDIO";
    case LogCategory::Error: return "ERROR";
    }
    return "UNKNOWN";
}

} // namespace localai

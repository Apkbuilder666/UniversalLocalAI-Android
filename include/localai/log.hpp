#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace localai {

enum class LogCategory { App, Model, Inference, Hardware, Backend, Memory, Image, Audio, Error };
enum class LogLevel { Debug, Info, Warning, Error };

struct LogEntry {
    std::chrono::system_clock::time_point timestamp;
    LogLevel level;
    LogCategory category;
    std::string message;
};

class Logger {
public:
    static Logger& instance();
    void setFile(const std::filesystem::path& path);
    void write(LogLevel level, LogCategory category, std::string message);
    std::vector<LogEntry> entries() const;
    void setListener(std::function<void(const LogEntry&)> listener);

private:
    mutable std::mutex mutex_;
    std::vector<LogEntry> entries_;
    std::ofstream file_;
    std::function<void(const LogEntry&)> listener_;
};

std::string toString(LogLevel value);
std::string toString(LogCategory value);

} // namespace localai

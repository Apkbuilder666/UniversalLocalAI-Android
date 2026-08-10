#pragma once

#include "localai/types.hpp"

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>

namespace localai {

struct ScheduledTask {
    std::future<Result<InferenceOutput>> future;
    CancellationFlag cancellation;
    void cancel() const { cancellation->store(true); }
};

class InferenceScheduler {
public:
    explicit InferenceScheduler(std::size_t workers = 1);
    ~InferenceScheduler();
    InferenceScheduler(const InferenceScheduler&) = delete;
    InferenceScheduler& operator=(const InferenceScheduler&) = delete;

    ScheduledTask submit(std::function<Result<InferenceOutput>(const CancellationFlag&)> operation);
    void shutdown();

private:
    void workerLoop();
    std::vector<std::jthread> workers_;
    std::deque<std::function<void()>> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_{};
};

} // namespace localai

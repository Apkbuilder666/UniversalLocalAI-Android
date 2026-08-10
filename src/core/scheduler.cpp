#include "localai/scheduler.hpp"

#include <algorithm>
#include <stdexcept>

namespace localai {

InferenceScheduler::InferenceScheduler(std::size_t workers) {
    workers = std::max<std::size_t>(1, workers);
    workers_.reserve(workers);
    for (std::size_t index = 0; index < workers; ++index) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

InferenceScheduler::~InferenceScheduler() {
    shutdown();
}

ScheduledTask InferenceScheduler::submit(
    std::function<Result<InferenceOutput>(const CancellationFlag&)> operation) {
    auto cancellation = std::make_shared<std::atomic_bool>(false);
    auto task = std::make_shared<std::packaged_task<Result<InferenceOutput>()>>(
        [operation = std::move(operation), cancellation] { return operation(cancellation); });
    auto future = task->get_future();
    {
        std::scoped_lock lock(mutex_);
        if (stopping_) {
            throw std::runtime_error("Inference scheduler is shutting down");
        }
        queue_.emplace_back([task] { (*task)(); });
    }
    condition_.notify_one();
    return {std::move(future), std::move(cancellation)};
}

void InferenceScheduler::shutdown() {
    {
        std::scoped_lock lock(mutex_);
        if (stopping_) {
            return;
        }
        stopping_ = true;
    }
    condition_.notify_all();
    workers_.clear();
}

void InferenceScheduler::workerLoop() {
    while (true) {
        std::function<void()> operation;
        {
            std::unique_lock lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_ && queue_.empty()) {
                return;
            }
            operation = std::move(queue_.front());
            queue_.pop_front();
        }
        operation();
    }
}

} // namespace localai

#ifndef framewriter
#define framewriter

#include "frame.hpp"
#include "bmpwriter.hpp"

#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <atomic>
#include <cstddef>
#include <utility>

namespace Grid {

using u_lock = std::unique_lock<std::shared_mutex>;
using s_lock = std::shared_lock<std::shared_mutex>;

///@brief Format the writer encodes a queued frame to
enum class WriteFormat : uint8_t { BMP = 0 };

///@brief One queued disk write; owns its pixels so the caller can reuse its own
struct FrameWriteJob {
    frame image;
    std::string path;
    WriteFormat format = WriteFormat::BMP;

    FrameWriteJob() = default;
    FrameWriteJob(frame&& img, const std::string& p, WriteFormat fmt = WriteFormat::BMP)
            : image(std::move(img)), path(p), format(fmt) {
    }
};

///@brief Background frame-to-disk writer, keeps encode/IO off the submitting thread
class FrameWriter {
public:
    ///@brief Starts the worker pool
    ///@param threads Number of writer threads
    ///@param maxQueued Backpressure limit; 0 disables the cap
    explicit FrameWriter(size_t threads = 2, size_t maxQueued = 8)
            : maxQueued_(maxQueued) {
        if (threads == 0) threads = 1;
        stopWorker_.store(false, std::memory_order_relaxed);
        workers_.reserve(threads);
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back(&FrameWriter::workerLoop, this);
        }
    }

    ~FrameWriter() {
        shutdown();
    }

    FrameWriter(const FrameWriter&) = delete;
    FrameWriter& operator=(const FrameWriter&) = delete;

    ///@brief Queues a frame for writing, moving the pixels out of the caller
    ///@param image Frame to hand off; left empty afterwards
    ///@param path Destination file path
    ///@param format Encoder to use
    void enqueue(frame&& image, const std::string& path, WriteFormat format = WriteFormat::BMP) {
        //TIME_FUNCTION;
        u_lock lock(queueMutex_);
        if (maxQueued_ != 0) {
            while (jobs_.size() >= maxQueued_ && !stopWorker_.load(std::memory_order_relaxed)) {
                spaceCV_.wait(lock);
            }
        }
        if (stopWorker_.load(std::memory_order_relaxed)) return;
        jobs_.emplace(std::move(image), path, format);
        queued_.fetch_add(1, std::memory_order_relaxed);
        lock.unlock();
        jobCV_.notify_one();
    }

    ///@brief Blocks until every queued frame has hit disk
    void drain() {
        //TIME_FUNCTION;
        u_lock lock(queueMutex_);
        while (!jobs_.empty() || active_.load(std::memory_order_relaxed) != 0) {
            idleCV_.wait(lock);
        }
    }

    ///@brief Drains outstanding work then joins the workers
    void shutdown() {
        if (stopWorker_.load(std::memory_order_relaxed)) return;
        drain();
        u_lock lock(queueMutex_);
        stopWorker_.store(true, std::memory_order_relaxed);
        lock.unlock();
        jobCV_.notify_all();
        spaceCV_.notify_all();
        for (std::thread& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();
    }

    ///@brief Frames handed to the writer since construction
    uint64_t queuedCount() const {
        return queued_.load(std::memory_order_relaxed);
    }

    ///@brief Frames fully flushed to disk
    uint64_t writtenCount() const {
        return written_.load(std::memory_order_relaxed);
    }

    ///@brief Jobs still waiting or in flight
    size_t pending() const {
        s_lock lock(queueMutex_);
        return jobs_.size() + active_.load(std::memory_order_relaxed);
    }

private:
    ///@brief Pulls jobs off the queue until shutdown
    void workerLoop() {
        for (;;) {
            u_lock lock(queueMutex_);
            while (jobs_.empty() && !stopWorker_.load(std::memory_order_relaxed)) {
                jobCV_.wait(lock);
            }
            if (jobs_.empty() && stopWorker_.load(std::memory_order_relaxed)) return;

            FrameWriteJob job = std::move(jobs_.front());
            jobs_.pop();
            active_.fetch_add(1, std::memory_order_relaxed);
            lock.unlock();
            spaceCV_.notify_one();

            writeJob(job);

            lock.lock();
            active_.fetch_sub(1, std::memory_order_relaxed);
            written_.fetch_add(1, std::memory_order_relaxed);
            lock.unlock();
            idleCV_.notify_all();
        }
    }

    ///@brief Encodes and writes a single job
    void writeJob(FrameWriteJob& job) {
        //TIME_FUNCTION;
        switch (job.format) {
            case WriteFormat::BMP:
                BMPWriter::saveBMP(job.path, job.image);
                break;
        }
    }

    mutable std::shared_mutex queueMutex_;
    std::queue<FrameWriteJob> jobs_;
    std::condition_variable_any jobCV_;
    std::condition_variable_any spaceCV_;
    std::condition_variable_any idleCV_;
    std::vector<std::thread> workers_;
    std::atomic<bool> stopWorker_{false};
    std::atomic<size_t> active_{0};
    std::atomic<uint64_t> queued_{0};
    std::atomic<uint64_t> written_{0};
    size_t maxQueued_ = 8;
};

}

#endif
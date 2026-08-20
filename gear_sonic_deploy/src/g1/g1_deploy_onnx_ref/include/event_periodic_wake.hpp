/**
 * @file event_periodic_wake.hpp
 * @brief One-consumer event wake with a fixed maintenance cadence.
 *
 * The command writer owns the sole DDS publisher.  Producers do not call DDS;
 * they only notify this loop after atomically replacing or clearing the command
 * envelope.  Notifications carry a monotonically increasing generation rather
 * than a boolean so one arriving while the callback is running remains visible
 * on the next wait.
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>

class EventPeriodicWake {
 public:
  using Clock = std::chrono::steady_clock;
  using Duration = Clock::duration;

  explicit EventPeriodicWake(Duration period) : period_(period) {
    if (period_ <= Duration::zero()) {
      throw std::invalid_argument("EventPeriodicWake period must be positive");
    }
  }

  EventPeriodicWake(const EventPeriodicWake&) = delete;
  EventPeriodicWake& operator=(const EventPeriodicWake&) = delete;

  /// Request one prompt callback without changing the periodic maintenance clock.
  void Notify() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++notification_generation_;
    }
    cv_.notify_one();
  }

  /// Terminally stop the loop and wake an idle consumer so its join cannot wait a period.
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    cv_.notify_all();
  }

  /**
   * Invoke @p callback immediately, on each unseen notification, and at the
   * fixed maintenance cadence. The callback runs without this object's mutex,
   * so producers can notify while DDS work is in progress. A callback that runs
   * long advances an overdue periodic deadline instead of spinning to catch up.
   */
  template <typename Callback>
  void Run(Callback&& callback) {
    std::uint64_t consumed_notification_generation = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopped_) {
        return;
      }
      consumed_notification_generation = notification_generation_;
    }

    auto next_periodic_deadline = Clock::now() + period_;
    for (;;) {
      callback();

      std::unique_lock<std::mutex> lock(mutex_);
      if (stopped_) {
        return;
      }

      const auto now = Clock::now();
      if (notification_generation_ != consumed_notification_generation) {
        consumed_notification_generation = notification_generation_;
        // If the event arrived at or after the periodic deadline, this
        // callback also satisfies that maintenance slot.
        if (now >= next_periodic_deadline) {
          do {
            next_periodic_deadline += period_;
          } while (next_periodic_deadline <= now);
        }
        lock.unlock();
        continue;
      }
      if (now >= next_periodic_deadline) {
        do {
          next_periodic_deadline += period_;
        } while (next_periodic_deadline <= now);
        lock.unlock();
        continue;
      }

      cv_.wait_until(lock, next_periodic_deadline, [&] {
        return stopped_ || notification_generation_ != consumed_notification_generation;
      });
      if (stopped_) {
        return;
      }
      if (notification_generation_ != consumed_notification_generation) {
        consumed_notification_generation = notification_generation_;
        const auto notification_now = Clock::now();
        if (notification_now >= next_periodic_deadline) {
          do {
            next_periodic_deadline += period_;
          } while (next_periodic_deadline <= notification_now);
        }
        lock.unlock();
        continue;
      }

      // The deadline woke us. Advance before the next callback so a callback
      // longer than one period cannot create a busy catch-up loop.
      const auto deadline_now = Clock::now();
      do {
        next_periodic_deadline += period_;
      } while (next_periodic_deadline <= deadline_now);
      lock.unlock();
    }
  }

 private:
  Duration period_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::uint64_t notification_generation_ = 0;
  bool stopped_ = false;
};

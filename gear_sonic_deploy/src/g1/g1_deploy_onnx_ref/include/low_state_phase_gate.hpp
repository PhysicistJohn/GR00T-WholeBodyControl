/**
 * @file low_state_phase_gate.hpp
 * @brief Race-free post-receipt phase gate for authoritative LowState data.
 *
 * A policy observation is built exclusively from a CRC-validated LowState
 * snapshot.  Other DDS streams may be useful telemetry, but they are sampled
 * independently and must not delay this control phase: DDS does not provide a
 * cross-topic atomic delivery boundary.  The LowState callback stores the
 * payload before calling NotifyLowState().  The gate then retains its own
 * immutable copy so a returned snapshot cannot be mixed with a later write.
 *
 * Generation counters and the condition-variable predicate are protected by
 * the same mutex.  This closes the check-then-wait notification race without
 * making an unrelated topic part of the policy clock.
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

template <typename LowStateT>
class LowStatePhaseGate {
 public:
  using Clock = std::chrono::steady_clock;
  using Duration = Clock::duration;

  enum class Result {
    kReady,
    kTimeout,
    kStopped,
  };

  struct Snapshot {
    std::shared_ptr<const LowStateT> low_state;
    std::uint64_t generation = 0;
    Clock::time_point receipt_time{};

    [[nodiscard]] bool HasLowState() const { return low_state != nullptr; }
  };

  struct WaitOutcome {
    Result result;
    Snapshot snapshot;
  };

  /// Record that a CRC-validated LowState has already been stored by the caller.
  void NotifyLowState(const LowStateT& low_state,
                      Clock::time_point receipt_time = Clock::now()) {
    auto immutable_low_state = std::make_shared<const LowStateT>(low_state);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopped_) {
        return;
      }
      snapshot_.low_state = std::move(immutable_low_state);
      snapshot_.receipt_time = receipt_time;
      ++snapshot_.generation;
    }
    cv_.notify_one();
  }

  /**
   * Wait for an immutable LowState snapshot no older than @p max_receipt_age.
   *
   * A fresh snapshot already present at entry returns immediately.  Otherwise
   * the method captures the generation under the mutex and waits for it to
   * advance.  Receipt age, rather than an arbitrary timer-entry boundary,
   * defines freshness.  Keeping the captured generation and the
   * condition-variable predicate under this lock closes the check-then-wait
   * notification race.
   */
  [[nodiscard]] WaitOutcome WaitForFreshLowState(Duration max_receipt_age,
                                                   Duration max_wait) {
    const auto deadline = Clock::now() +
                          (max_wait < Duration::zero() ? Duration::zero() : max_wait);
    std::unique_lock<std::mutex> lock(mutex_);

    if (stopped_) {
      return {Result::kStopped, snapshot_};
    }

    if (IsFreshLowStateLocked(Clock::now(), max_receipt_age)) {
      return {Result::kReady, snapshot_};
    }

    for (;;) {
      if (stopped_) {
        return {Result::kStopped, snapshot_};
      }

      const auto now = Clock::now();
      if (IsFreshLowStateLocked(now, max_receipt_age)) {
        return {Result::kReady, snapshot_};
      }
      if (now >= deadline) {
        return {Result::kTimeout, snapshot_};
      }

      const std::uint64_t observed_generation = snapshot_.generation;
      cv_.wait_until(lock, deadline, [&] {
        return stopped_ || snapshot_.generation != observed_generation;
      });
    }
  }

  /**
   * Check whether a previously returned snapshot is still fresh at the point
   * where a consumer is about to use it.  This is deliberately pure: callers
   * can revalidate after a scheduler delay without taking the gate's mutex or
   * changing its generation state.
   */
  [[nodiscard]] static bool IsFreshLowState(const Snapshot& snapshot,
                                             Clock::time_point now,
                                             Duration max_receipt_age) {
    if (!snapshot.HasLowState() || max_receipt_age < Duration::zero() ||
        snapshot.receipt_time > now) {
      return false;
    }
    return now - snapshot.receipt_time <= max_receipt_age;
  }

  /// Wake a pending control wait during shutdown.  Stop is terminal by design.
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    cv_.notify_all();
  }

 private:
  [[nodiscard]] bool IsFreshLowStateLocked(Clock::time_point now,
                                           Duration max_receipt_age) const {
    return IsFreshLowState(snapshot_, now, max_receipt_age);
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  Snapshot snapshot_;
  bool stopped_ = false;
};

/**
 * @file low_state_phase_gate.hpp
 * @brief Race-free post-wake gate for paired LowState and torso-IMU receipts.
 *
 * The writers must store their payload in the corresponding DataBuffer before
 * calling NotifyLowState() or NotifyTorsoImu().  The gate retains immutable
 * copies of both messages, so the returned pair cannot be mixed with a later
 * write to either DataBuffer.  Generation counters, timestamps, and payloads
 * are deliberately protected by the same mutex so a notification cannot be
 * lost between the freshness check and condition-variable wait.
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

template <typename LowStateT, typename TorsoImuT>
class LowStatePhaseGate {
 public:
  using Clock = std::chrono::steady_clock;
  using Duration = Clock::duration;

  enum class Result {
    kReady,
    kTimeout,
    kStopped,
  };

  struct PairSnapshot {
    std::shared_ptr<const LowStateT> low_state;
    std::shared_ptr<const TorsoImuT> torso_imu;
    std::uint64_t low_state_generation = 0;
    std::uint64_t torso_imu_generation = 0;
    Clock::time_point low_state_receipt_time{};
    Clock::time_point torso_imu_receipt_time{};

    [[nodiscard]] bool HasPair() const {
      return low_state != nullptr && torso_imu != nullptr;
    }
  };

  struct WaitOutcome {
    Result result;
    PairSnapshot pair;
  };

  /// Record that a CRC-validated LowState has already been stored by the caller.
  void NotifyLowState(const LowStateT& low_state, Clock::time_point receipt_time = Clock::now()) {
    auto immutable_low_state = std::make_shared<const LowStateT>(low_state);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopped_) {
        return;
      }
      pair_.low_state = std::move(immutable_low_state);
      pair_.low_state_receipt_time = receipt_time;
      ++pair_.low_state_generation;
    }
    cv_.notify_one();
  }

  /// Record that torso IMU data has already been stored by the caller.
  void NotifyTorsoImu(const TorsoImuT& torso_imu, Clock::time_point receipt_time = Clock::now()) {
    auto immutable_torso_imu = std::make_shared<const TorsoImuT>(torso_imu);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopped_) {
        return;
      }
      pair_.torso_imu = std::move(immutable_torso_imu);
      pair_.torso_imu_receipt_time = receipt_time;
      ++pair_.torso_imu_generation;
    }
    cv_.notify_one();
  }

  /**
   * Wait for a pair that is no older than @p max_receipt_age and whose
   * receipts are no farther apart than @p max_receipt_skew.
   *
   * A fresh pair already present at entry returns immediately.  Otherwise the
   * method captures both generations under the mutex and waits for either to
   * advance.  After each advance it reevaluates the complete immutable pair;
   * receipt age and skew, rather than an arbitrary timer-entry boundary,
   * define whether the pair is coherent.  Keeping the captured generations
   * and the condition-variable predicate under this lock closes the
   * check-then-wait notification race.
   */
  [[nodiscard]] WaitOutcome WaitForFreshPair(
      Duration max_receipt_age, Duration max_receipt_skew, Duration max_wait) {
    const auto deadline = Clock::now() + (max_wait < Duration::zero() ? Duration::zero() : max_wait);
    std::unique_lock<std::mutex> lock(mutex_);

    if (stopped_) {
      return {Result::kStopped, pair_};
    }

    if (IsFreshPairLocked(Clock::now(), max_receipt_age, max_receipt_skew)) {
      return {Result::kReady, pair_};
    }

    for (;;) {
      if (stopped_) {
        return {Result::kStopped, pair_};
      }

      const auto now = Clock::now();
      if (IsFreshPairLocked(now, max_receipt_age, max_receipt_skew)) {
        return {Result::kReady, pair_};
      }
      if (now >= deadline) {
        return {Result::kTimeout, pair_};
      }

      // An unusable update (for example, one stream without a recent peer)
      // is observed once, then the next update to either stream wakes us to
      // reconsider the complete pair.
      const std::uint64_t observed_low_state_generation = pair_.low_state_generation;
      const std::uint64_t observed_torso_imu_generation = pair_.torso_imu_generation;

      cv_.wait_until(lock, deadline, [&] {
        return stopped_ ||
               pair_.low_state_generation != observed_low_state_generation ||
               pair_.torso_imu_generation != observed_torso_imu_generation;
      });
    }
  }

  /**
   * Check whether a previously returned snapshot is still fresh at the point
   * where a consumer is about to use it.  This is deliberately pure: callers
   * can revalidate a snapshot after a scheduler delay without taking the
   * gate's mutex or changing its generation state.
   */
  [[nodiscard]] static bool IsFreshPair(
      const PairSnapshot& pair, Clock::time_point now, Duration max_receipt_age,
      Duration max_receipt_skew) {
    if (!pair.HasPair() || max_receipt_age < Duration::zero() ||
        max_receipt_skew < Duration::zero()) {
      return false;
    }
    if (pair.low_state_receipt_time > now || pair.torso_imu_receipt_time > now) {
      return false;
    }
    const auto low_state_age = now - pair.low_state_receipt_time;
    const auto torso_imu_age = now - pair.torso_imu_receipt_time;
    const auto receipt_skew = pair.low_state_receipt_time >= pair.torso_imu_receipt_time
                                  ? pair.low_state_receipt_time - pair.torso_imu_receipt_time
                                  : pair.torso_imu_receipt_time - pair.low_state_receipt_time;
    return low_state_age <= max_receipt_age && torso_imu_age <= max_receipt_age &&
           receipt_skew <= max_receipt_skew;
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
  bool IsFreshPairLocked(
      Clock::time_point now, Duration max_receipt_age, Duration max_receipt_skew) const {
    return IsFreshPair(pair_, now, max_receipt_age, max_receipt_skew);
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  PairSnapshot pair_;
  bool stopped_ = false;
};

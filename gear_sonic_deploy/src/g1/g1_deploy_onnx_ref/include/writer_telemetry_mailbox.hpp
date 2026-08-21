/**
 * @file writer_telemetry_mailbox.hpp
 * @brief Lock-free latest-snapshot handoff from the DDS writer to telemetry.
 *
 * The sole low-command writer must not block on formatting or stdout while a
 * new LowState generation may be approaching its 3 ms first-write boundary.
 * This mailbox lets that writer publish compact numeric snapshots with only
 * lock-free atomics; a non-writer thread may format and emit them later.
 */
#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace writer_telemetry {

template <std::size_t WordCount>
class LatestSnapshotMailbox {
 public:
  static_assert(WordCount > 0, "a telemetry snapshot needs at least one word");
  static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                "writer telemetry requires lock-free 64-bit atomics");

  using Payload = std::array<std::uint64_t, WordCount>;

  LatestSnapshotMailbox() noexcept {
    for (auto& word : payload_) {
      word.store(0, std::memory_order_relaxed);
    }
  }

  /// Sole-producer publication. The odd/even sequence makes the snapshot
  /// coherent without a mutex or allocation; readers never wait for a writer.
  void Publish(const Payload& payload) noexcept {
    sequence_.fetch_add(1, std::memory_order_acq_rel);  // write in progress
    for (std::size_t i = 0; i < WordCount; ++i) {
      payload_[i].store(payload[i], std::memory_order_relaxed);
    }
    sequence_.fetch_add(1, std::memory_order_release);  // complete snapshot
  }

  /// Return the newest coherent snapshot once per observed sequence. A
  /// concurrent writer simply makes this non-blocking read retry on a later
  /// telemetry tick.
  [[nodiscard]] bool TryRead(Payload& payload,
                             std::uint64_t& observed_sequence) const noexcept {
    const std::uint64_t begin = sequence_.load(std::memory_order_acquire);
    if (begin == 0 || begin == observed_sequence || (begin & 1U) != 0U) {
      return false;
    }
    Payload candidate {};
    for (std::size_t i = 0; i < WordCount; ++i) {
      candidate[i] = payload_[i].load(std::memory_order_relaxed);
    }
    const std::uint64_t end = sequence_.load(std::memory_order_acquire);
    if (begin != end || (end & 1U) != 0U) {
      return false;
    }
    payload = candidate;
    observed_sequence = end;
    return true;
  }

 private:
  std::array<std::atomic<std::uint64_t>, WordCount> payload_ {};
  std::atomic<std::uint64_t> sequence_ {0};
};

inline std::uint64_t PackDouble(double value) noexcept {
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  return std::bit_cast<std::uint64_t>(value);
}

inline double UnpackDouble(std::uint64_t value) noexcept {
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  return std::bit_cast<double>(value);
}

}  // namespace writer_telemetry

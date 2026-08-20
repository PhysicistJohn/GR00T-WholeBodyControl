/**
 * @file command_freshness_fence.hpp
 * @brief Writer-side freshness fence for LowState-phased motor commands.
 *
 * A DataBuffer write timestamp says when the control producer happened to
 * publish an envelope, not when the LowState observation that produced that
 * command arrived.  A stalled control thread can therefore wake up and make
 * an old policy result look new by writing it late.  This fence evaluates the
 * immutable LowState receipt provenance carried with the command instead.
 *
 * In addition to the 50 ms command lease, the first rt/lowcmd write for each
 * distinct LowState generation must occur no more than 3 ms after receipt.
 * That is the meaningful source-side phase boundary: subsequent 500 Hz
 * retransmits intentionally keep the accepted command lease alive, but can
 * never be mistaken for the first application of a policy update.
 *
 * The fence is intentionally writer-owned.  A 500 Hz writer may resend one
 * accepted 50 Hz command until its source receipt expires, but it may never
 * replay that command beyond the fixed 50 ms lease.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>

template <typename CommandT>
struct PhasedCommand {
  CommandT command;
  std::chrono::steady_clock::time_point source_low_state_receipt{};
  std::chrono::steady_clock::time_point gate_wake_time{};
  std::chrono::steady_clock::time_point envelope_published_time{};
  std::uint64_t source_generation = 0;
};

template <typename CommandT>
class CommandFreshnessFence {
 public:
  using Clock = std::chrono::steady_clock;
  using Envelope = PhasedCommand<CommandT>;

  // Fixed deliberately: it matches the existing LowState late threshold,
  // exceeds the nominal 20 ms control cadence, and covers normal inference
  // latency without allowing indefinite command replay.
  static constexpr auto kMaxCommandAge = std::chrono::milliseconds{50};
  // A phase deadline applies only to the first writer dispatch of a new
  // generation. It is evaluated at the local rt/lowcmd call boundary on the
  // same steady clock as receipt, gate wake, and envelope publication.
  static constexpr auto kMaxFirstWritePhase = std::chrono::milliseconds{3};

  enum class Result {
    kAccepted,
    kMissing,
    kFutureReceipt,
    kMissingPhaseProvenance,
    kInvalidPhaseOrdering,
    kFuturePhaseTimestamp,
    kExpired,
    kFirstWritePhaseDeadlineMissed,
    // The SDK rejected the first local DDS dispatch. This generation is
    // terminally fenced so a later 500 Hz retransmission cannot revive it.
    kFirstWriteDispatchFailed,
    kGenerationRegressed,
  };

  struct Decision {
    Result result = Result::kMissing;
    bool is_first_write = false;

    [[nodiscard]] bool IsAccepted() const { return result == Result::kAccepted; }
    [[nodiscard]] bool IsFirstWrite() const {
      return IsAccepted() && is_first_write;
    }
  };

  [[nodiscard]] static bool IsFirstWritePhaseOnTime(const Envelope& envelope,
                                                     Clock::time_point now) {
    return envelope.source_low_state_receipt <= now &&
           now - envelope.source_low_state_receipt <= kMaxFirstWritePhase;
  }

  /// Convert an accepted-but-not-yet-dispatched first write into a terminal
  /// result. The caller uses this for a final deadline recheck or an SDK-local
  /// dispatch failure; the same generation can never be accepted again.
  void RejectAcceptedFirstWrite(std::uint64_t generation, Result terminal_result) {
    if (has_processed_generation_ && generation == last_processed_generation_ &&
        last_generation_result_ == Result::kAccepted) {
      last_generation_result_ = terminal_result;
    }
  }

  /**
   * Evaluate one immutable command envelope at the writer's current time.
   *
   * The producer's DataBuffer write time is deliberately absent from this API.
   * A generation is allowed to repeat so the writer can publish the same 50 Hz
   * command at 500 Hz. Its first writer dispatch is nevertheless one-shot:
   * once a newer generation misses the 3 ms phase deadline it is terminally
   * rejected and the writer must emit damping until a still newer generation
   * arrives. A lower generation can never replace a processed newer one.
   */
  [[nodiscard]] Decision Evaluate(const std::shared_ptr<const Envelope>& envelope,
                                  Clock::time_point now = Clock::now()) {
    if (!envelope || envelope->source_generation == 0 ||
        envelope->source_low_state_receipt == Clock::time_point{}) {
      return {Result::kMissing};
    }
    if (envelope->gate_wake_time == Clock::time_point{} ||
        envelope->envelope_published_time == Clock::time_point{}) {
      return {Result::kMissingPhaseProvenance};
    }
    if (envelope->source_low_state_receipt > envelope->gate_wake_time ||
        envelope->gate_wake_time > envelope->envelope_published_time) {
      return {Result::kInvalidPhaseOrdering};
    }
    if (envelope->source_low_state_receipt > now) {
      return {Result::kFutureReceipt};
    }
    if (envelope->gate_wake_time > now || envelope->envelope_published_time > now) {
      return {Result::kFuturePhaseTimestamp};
    }

    if (has_processed_generation_ &&
        envelope->source_generation < last_processed_generation_) {
      return {Result::kGenerationRegressed};
    }

    if (has_processed_generation_ &&
        envelope->source_generation == last_processed_generation_) {
      if (last_generation_result_ != Result::kAccepted) {
        return {last_generation_result_};
      }
      if (now - envelope->source_low_state_receipt > kMaxCommandAge) {
        return {Result::kExpired};
      }
      return {Result::kAccepted, false};
    }

    // A valid newer generation fences all older ones even when it has already
    // missed its own deadline. This prevents a late producer from silently
    // reviving a command whose successor has been observed by the writer.
    has_processed_generation_ = true;
    last_processed_generation_ = envelope->source_generation;
    if (now - envelope->source_low_state_receipt > kMaxCommandAge) {
      last_generation_result_ = Result::kExpired;
      return {last_generation_result_};
    }
    if (!IsFirstWritePhaseOnTime(*envelope, now)) {
      last_generation_result_ = Result::kFirstWritePhaseDeadlineMissed;
      return {last_generation_result_};
    }

    last_generation_result_ = Result::kAccepted;
    return {Result::kAccepted, true};
  }

 private:
  bool has_processed_generation_ = false;
  std::uint64_t last_processed_generation_ = 0;
  Result last_generation_result_ = Result::kMissing;
};

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

  enum class Result {
    kAccepted,
    kMissing,
    kFutureReceipt,
    kExpired,
    kGenerationRegressed,
  };

  struct Decision {
    Result result = Result::kMissing;

    [[nodiscard]] bool IsAccepted() const { return result == Result::kAccepted; }
  };

  /**
   * Evaluate one immutable command envelope at the writer's current time.
   *
   * The producer's DataBuffer write time is deliberately absent from this API.
   * A generation is allowed to repeat so the writer can publish the same 50 Hz
   * command at 500 Hz, but a lower generation can never replace a newer one
   * that this writer has already accepted.
   */
  [[nodiscard]] Decision Evaluate(const std::shared_ptr<const Envelope>& envelope,
                                  Clock::time_point now = Clock::now()) {
    if (!envelope || envelope->source_generation == 0 ||
        envelope->source_low_state_receipt == Clock::time_point{}) {
      return {Result::kMissing};
    }
    if (envelope->source_low_state_receipt > now) {
      return {Result::kFutureReceipt};
    }
    if (now - envelope->source_low_state_receipt > kMaxCommandAge) {
      return {Result::kExpired};
    }
    if (has_accepted_generation_ &&
        envelope->source_generation < last_accepted_generation_) {
      return {Result::kGenerationRegressed};
    }

    has_accepted_generation_ = true;
    last_accepted_generation_ = envelope->source_generation;
    return {Result::kAccepted};
  }

 private:
  bool has_accepted_generation_ = false;
  std::uint64_t last_accepted_generation_ = 0;
};

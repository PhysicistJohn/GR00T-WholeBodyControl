/**
 * @file low_command_safety.hpp
 * @brief Small, dependency-free primitives for the final low-command writer.
 *
 * These primitives deliberately model the writer cadence, not the policy
 * cadence.  The policy publishes a new target at 50 Hz, while the sole DDS
 * writer can publish every 2 ms.  Any per-write slew limit must therefore use
 * the latter interval.
 */
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <utility>

namespace low_command_safety {

inline float SlewStepForElapsedTime(
    float max_slew_radians_per_second,
    std::chrono::steady_clock::duration elapsed_since_last_successful_write,
    std::chrono::steady_clock::duration max_writer_period) noexcept {
  if (elapsed_since_last_successful_write <=
          std::chrono::steady_clock::duration::zero() ||
      max_writer_period <= std::chrono::steady_clock::duration::zero()) {
    return 0.0F;
  }
  // Event wakes may be closer together than the periodic cadence, while a
  // scheduler stall may be much longer.  Honor the former without letting the
  // latter accumulate a large one-packet target step.
  const auto bounded_elapsed = std::min(elapsed_since_last_successful_write,
                                        max_writer_period);
  const float elapsed_seconds =
      std::chrono::duration<float>(bounded_elapsed).count();
  return max_slew_radians_per_second * elapsed_seconds;
}

inline bool IsMeasuredDampingReferenceFresh(
    std::chrono::steady_clock::time_point measured_time,
    std::chrono::steady_clock::time_point now,
    std::chrono::steady_clock::duration max_age) noexcept {
  return max_age >= std::chrono::steady_clock::duration::zero() &&
         measured_time <= now && now - measured_time <= max_age;
}

inline float ClampTargetSlew(float candidate, float previously_published,
                             bool has_previously_published,
                             float max_step_radians) noexcept {
  if (!has_previously_published) {
    return candidate;
  }
  return std::clamp(candidate, previously_published - max_step_radians,
                    previously_published + max_step_radians);
}

inline float ApplyTargetSlew(float candidate, float previously_published,
                             bool has_previously_published,
                             float max_step_radians,
                             bool enforce_slew_limit) noexcept {
  return enforce_slew_limit
      ? ClampTargetSlew(candidate, previously_published,
                         has_previously_published, max_step_radians)
      : candidate;
}

inline float DampingTargetFromMeasuredQ(float measured_q,
                                        float previously_published,
                                        bool has_previously_published) noexcept {
  if (std::isfinite(measured_q)) {
    return measured_q;
  }
  return has_previously_published ? previously_published : 0.0F;
}

inline float DampingTargetFromFreshMeasuredQ(
    float measured_q, std::chrono::steady_clock::time_point measured_time,
    std::chrono::steady_clock::time_point now,
    std::chrono::steady_clock::duration max_age,
    float previously_published, bool has_previously_published) noexcept {
  if (!IsMeasuredDampingReferenceFresh(measured_time, now, max_age)) {
    return has_previously_published ? previously_published : 0.0F;
  }
  return DampingTargetFromMeasuredQ(measured_q, previously_published,
                                    has_previously_published);
}

template <std::size_t Count>
[[nodiscard]] bool HasOnlyFiniteActuatorFields(
    const std::array<float, Count>& q_target,
    const std::array<float, Count>& dq_target,
    const std::array<float, Count>& tau_ff,
    const std::array<float, Count>& kp,
    const std::array<float, Count>& kd) noexcept {
  for (std::size_t i = 0; i < Count; ++i) {
    if (!std::isfinite(q_target[i]) || !std::isfinite(dq_target[i]) ||
        !std::isfinite(tau_ff[i]) || !std::isfinite(kp[i]) ||
        !std::isfinite(kd[i])) {
      return false;
    }
  }
  return true;
}

template <std::size_t Count, typename ReadMeasuredQ>
std::array<float, Count> MeasuredDampingTargets(ReadMeasuredQ&& read_measured_q) {
  std::array<float, Count> targets {};
  for (std::size_t i = 0; i < Count; ++i) {
    targets[i] = static_cast<float>(std::forward<ReadMeasuredQ>(read_measured_q)(i));
  }
  return targets;
}

}  // namespace low_command_safety

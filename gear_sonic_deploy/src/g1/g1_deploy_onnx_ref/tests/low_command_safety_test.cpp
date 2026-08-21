#include "command_freshness_fence.hpp"
#include "low_command_safety.hpp"

#include <algorithm>
#include <vector>

#include "policy_parameters.hpp"
#include "robot_parameters.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>

namespace {

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "low_command_safety_test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    Fail(message);
  }
}

bool NearlyEqual(float lhs, float rhs, float tolerance = 1.0e-6F) {
  return std::fabs(lhs - rhs) <= tolerance;
}

struct TestActuatorCommand {
  std::array<float, G1_NUM_MOTOR> tau_ff {};
  std::array<float, G1_NUM_MOTOR> q_target {};
  std::array<float, G1_NUM_MOTOR> dq_target {};
  std::array<float, G1_NUM_MOTOR> kp {};
  std::array<float, G1_NUM_MOTOR> kd {};
};

void TestCanonicalDampingFieldsCoverEveryG1Joint() {
  using namespace std::chrono_literals;
  using Clock = std::chrono::steady_clock;
  std::array<float, G1_NUM_MOTOR> measured {};
  std::array<float, G1_NUM_MOTOR> previously_published {};
  for (std::size_t i = 0; i < measured.size(); ++i) {
    measured[i] = -0.312F + 0.021F * static_cast<float>(i);
    previously_published[i] = 0.60F - 0.013F * static_cast<float>(i);
  }

  const auto now = Clock::time_point {100ms};
  const auto damping = low_command_safety::BuildCanonicalDampingFields<G1_NUM_MOTOR>(
      /*has_measured_state=*/true,
      [&measured](std::size_t index) { return measured[index]; }, now - 1ms,
      now, 10ms, previously_published, /*has_previously_published=*/true);
  for (std::size_t i = 0; i < measured.size(); ++i) {
    Expect(NearlyEqual(damping.q_target[i], measured[i]),
           "every fresh measured joint position must become the damping q target");
    Expect(NearlyEqual(damping.dq_target[i], 0.0F),
           "canonical damping must zero every velocity target");
    Expect(NearlyEqual(damping.tau_ff[i], 0.0F),
           "canonical damping must zero every feedforward torque");
    Expect(NearlyEqual(damping.kp[i], 0.0F),
           "canonical damping must zero every proportional gain");
    Expect(NearlyEqual(damping.kd[i], 8.0F),
           "canonical damping must set every derivative gain to eight");
  }
  Expect(low_command_safety::HasOnlyFiniteActuatorFields(
             damping.q_target, damping.dq_target, damping.tau_ff, damping.kp,
             damping.kd),
         "the 29-joint canonical damping command must be fully finite");
}

void TestCanonicalDampingRetainsEveryLastSafeReferenceWhenUnavailable() {
  using namespace std::chrono_literals;
  using Clock = std::chrono::steady_clock;
  std::array<float, G1_NUM_MOTOR> previously_published {};
  for (std::size_t i = 0; i < previously_published.size(); ++i) {
    previously_published[i] = -0.53F + 0.019F * static_cast<float>(i);
  }

  const auto now = Clock::time_point {100ms};
  int unavailable_reader_calls = 0;
  const auto unavailable =
      low_command_safety::BuildCanonicalDampingFields<G1_NUM_MOTOR>(
          /*has_measured_state=*/false,
          [&unavailable_reader_calls](std::size_t) {
            ++unavailable_reader_calls;
            return std::numeric_limits<float>::quiet_NaN();
          },
          now, now, 10ms, previously_published,
          /*has_previously_published=*/true);
  const auto stale = low_command_safety::BuildCanonicalDampingFields<G1_NUM_MOTOR>(
      /*has_measured_state=*/true,
      [](std::size_t) { return 42.0F; }, now - 10ms - 1us, now, 10ms,
      previously_published, /*has_previously_published=*/true);
  const auto invalid =
      low_command_safety::BuildCanonicalDampingFields<G1_NUM_MOTOR>(
          /*has_measured_state=*/true,
          [](std::size_t) { return std::numeric_limits<float>::quiet_NaN(); },
          now - 1ms, now, 10ms, previously_published,
      /*has_previously_published=*/true);
  Expect(unavailable_reader_calls == 0,
         "a missing LowState must not dereference a measurement reader");
  for (std::size_t i = 0; i < previously_published.size(); ++i) {
    Expect(NearlyEqual(unavailable.q_target[i], previously_published[i]),
           "a missing LowState must retain every last safe q target");
    Expect(NearlyEqual(stale.q_target[i], previously_published[i]),
           "a stale LowState must retain every last safe q target");
    Expect(NearlyEqual(invalid.q_target[i], previously_published[i]),
           "an invalid LowState q must retain every last safe q target");
    Expect(NearlyEqual(unavailable.dq_target[i], 0.0F) &&
               NearlyEqual(unavailable.tau_ff[i], 0.0F) &&
               NearlyEqual(unavailable.kp[i], 0.0F) &&
               NearlyEqual(unavailable.kd[i], 8.0F),
           "unavailable-state damping must retain the canonical kd-only fields");
  }
}

void TestSlewLimitUsesWriterCadence() {
  using namespace std::chrono_literals;
  const float max_step = low_command_safety::SlewStepForElapsedTime(
      Q_TARGET_SLEW_LIMIT, 2ms, 2ms);
  Expect(NearlyEqual(max_step, 0.07F),
         "35 rad/s at 500 Hz must allow 0.07 rad per publish, not 0.7 rad");

  float published = 0.0F;
  for (int i = 0; i < 10; ++i) {
    const float next = low_command_safety::ClampTargetSlew(
        10.0F, published, true, max_step);
    Expect(next - published <= max_step + 1.0e-6F,
           "each writer publish must stay within its slew allowance");
    published = next;
  }
  Expect(NearlyEqual(published, Q_TARGET_SLEW_LIMIT * 0.02F),
         "ten 500 Hz writes must cover only 20 ms of the configured slew rate");
}

void TestEventWakesCannotBorrowAFullWriterStep() {
  using namespace std::chrono_literals;
  const float event_step = low_command_safety::SlewStepForElapsedTime(
      Q_TARGET_SLEW_LIMIT, 500us, 2ms);
  Expect(NearlyEqual(event_step, 0.0175F),
         "an event wake after 0.5 ms must receive only 0.5 ms of slew budget");

  float published = 0.0F;
  for (int i = 0; i < 4; ++i) {
    const float next = low_command_safety::ClampTargetSlew(
        10.0F, published, true, event_step);
    Expect(next - published <= event_step + 1.0e-6F,
           "each closely bunched wake must remain rate-bounded");
    published = next;
  }
  Expect(NearlyEqual(published, 0.07F),
         "four 0.5 ms event wakes must cover no more than 2 ms of slew");
}

void TestWriterStallCannotAccumulateARecoveryJump() {
  using namespace std::chrono_literals;
  const float stalled_step = low_command_safety::SlewStepForElapsedTime(
      Q_TARGET_SLEW_LIMIT, 20ms, 2ms);
  Expect(NearlyEqual(stalled_step, 0.07F),
         "a long writer stall must retain the one-packet 500 Hz slew cap");
}

void TestMeasuredDampingDoesNotRebaseToZero() {
  constexpr float previously_published = -0.28F;
  constexpr float measured_joint_q = -0.31F;
  const float max_step = low_command_safety::SlewStepForElapsedTime(
      Q_TARGET_SLEW_LIMIT, std::chrono::milliseconds {2},
      std::chrono::milliseconds {2});
  const float damping_target = low_command_safety::ClampTargetSlew(
      measured_joint_q, previously_published, true, max_step);
  Expect(NearlyEqual(damping_target, measured_joint_q),
         "a nearby measured damping target must be retained instead of rebasing to zero");
}

void TestDampingCommitsMeasuredReferenceAndPolicyRecoversFromIt() {
  using namespace std::chrono_literals;
  constexpr float stale_full_gain_target = 0.60F;
  constexpr float measured_joint_q = -0.31F;
  const float max_step = low_command_safety::SlewStepForElapsedTime(
      Q_TARGET_SLEW_LIMIT, 2ms, 2ms);

  const float damping_target = low_command_safety::ApplyTargetSlew(
      measured_joint_q, stale_full_gain_target, true, max_step,
      /*enforce_slew_limit=*/false);
  Expect(NearlyEqual(damping_target, measured_joint_q),
         "kd-only damping must commit the measured q reference without a stale-policy slew clamp");

  const float recovered_policy_target = low_command_safety::ApplyTargetSlew(
      0.50F, damping_target, true, max_step,
      /*enforce_slew_limit=*/true);
  Expect(NearlyEqual(recovered_policy_target, measured_joint_q + max_step),
         "the next full-gain policy command must ramp from the measured damping reference");
  Expect(std::fabs(recovered_policy_target - damping_target) <= max_step + 1.0e-6F,
         "recovery after damping must retain the per-packet slew cap");
}

void TestLateGenerationFallsBackToFreshMeasuredDampingThenRecovers() {
  using namespace std::chrono_literals;
  using Fence = CommandFreshnessFence<TestActuatorCommand>;

  std::array<float, G1_NUM_MOTOR> last_full_gain_targets {};
  std::array<float, G1_NUM_MOTOR> fresh_measured_q {};
  std::array<float, G1_NUM_MOTOR> recovered_policy_q {};
  for (std::size_t i = 0; i < last_full_gain_targets.size(); ++i) {
    last_full_gain_targets[i] = (i % 2 == 0 ? 0.60F : -0.60F) +
        0.005F * static_cast<float>(i);
    fresh_measured_q[i] = (i % 2 == 0 ? -0.31F : 0.42F) -
        0.003F * static_cast<float>(i);
    recovered_policy_q[i] = (i % 2 == 0 ? 0.70F : -0.50F) +
        0.002F * static_cast<float>(i);
  }
  constexpr auto kWriterPeriod = 2ms;
  // Exercise the writer-stall cap explicitly: recovery arrives 7 ms after
  // the successful damping write, but it may use no more than one 2 ms step.
  constexpr auto kRecoveryDt = 7ms;
  constexpr auto kDampingReferenceMaxAge = 10ms;
  const auto damping_time = Fence::Clock::time_point {std::chrono::seconds {1}};

  Fence fence;
  TestActuatorCommand late_command;
  late_command.q_target = last_full_gain_targets;
  late_command.kp.fill(12.0F);
  late_command.kd.fill(1.0F);
  const auto late_receipt = damping_time - Fence::kMaxFirstWritePhase - 1us;
  const auto late_envelope = std::make_shared<const Fence::Envelope>(
      Fence::Envelope{
          .command = late_command,
          .source_low_state_receipt = late_receipt,
          .gate_wake_time = late_receipt + 100us,
          .envelope_published_time = late_receipt + 200us,
          .source_generation = 41,
      });
  const auto late_decision = fence.Evaluate(late_envelope, damping_time);
  Expect(late_decision.result == Fence::Result::kFirstWritePhaseDeadlineMissed,
         "a complete generation evaluated after 3 ms must fall back to damping");
  Expect(fence.Evaluate(late_envelope, damping_time + 1ms).result ==
             Fence::Result::kFirstWritePhaseDeadlineMissed,
         "a late generation must remain terminally fenced on writer replay");

  // This is the same dependency-free field constructor used directly by
  // MakeDampingCommand. It must cover all G1 joints, not a small model.
  const auto damping = low_command_safety::BuildCanonicalDampingFields<G1_NUM_MOTOR>(
      /*has_measured_state=*/true,
      [&fresh_measured_q](std::size_t index) { return fresh_measured_q[index]; },
      damping_time - 1ms, damping_time, kDampingReferenceMaxAge,
      last_full_gain_targets, /*has_previously_published=*/true);
  for (std::size_t i = 0; i < damping.q_target.size(); ++i) {
    Expect(NearlyEqual(damping.q_target[i], fresh_measured_q[i]),
           "fresh measured q must be the canonical damping reference");
    Expect(NearlyEqual(damping.tau_ff[i], 0.0F),
           "canonical damping must have zero feedforward torque");
    Expect(NearlyEqual(damping.dq_target[i], 0.0F),
           "canonical damping must have zero velocity target");
    Expect(NearlyEqual(damping.kp[i], 0.0F),
           "canonical damping must have zero proportional gain");
    Expect(NearlyEqual(damping.kd[i], 8.0F),
           "canonical damping must have derivative gain eight");
  }
  Expect(low_command_safety::HasOnlyFiniteActuatorFields(
             damping.q_target, damping.dq_target, damping.tau_ff, damping.kp,
             damping.kd),
         "canonical damping fields must all be finite");
  const std::array<float, G1_NUM_MOTOR> last_successfully_published_q =
      damping.q_target;

  // A new, in-phase policy envelope may recover after that successful local
  // damping write. Its full-gain q targets must ramp from the measured
  // damping reference, not from the stale policy target.
  const auto recovery_time = damping_time + kRecoveryDt;
  TestActuatorCommand recovery_command;
  recovery_command.q_target = recovered_policy_q;
  recovery_command.kp.fill(12.0F);
  recovery_command.kd.fill(1.0F);
  const auto recovered_envelope = std::make_shared<const Fence::Envelope>(
      Fence::Envelope{
          .command = recovery_command,
          .source_low_state_receipt = recovery_time - 100us,
          .gate_wake_time = recovery_time - 50us,
          .envelope_published_time = recovery_time - 25us,
          .source_generation = 42,
      });
  const auto recovery_decision = fence.Evaluate(recovered_envelope, recovery_time);
  Expect(recovery_decision.IsAccepted() && recovery_decision.IsFirstWrite(),
         "only a fresh generation after damping may recover command output");

  const float max_recovery_delta = Q_TARGET_SLEW_LIMIT *
      std::chrono::duration<float>(std::min(kRecoveryDt, kWriterPeriod)).count();
  const float recovery_step = low_command_safety::SlewStepForElapsedTime(
      Q_TARGET_SLEW_LIMIT, kRecoveryDt, kWriterPeriod);
  Expect(NearlyEqual(recovery_step, max_recovery_delta),
         "recovery allowance must be 35 rad/s times min(dt, 2 ms)");
  for (std::size_t i = 0; i < last_successfully_published_q.size(); ++i) {
    const float recovered_target = low_command_safety::ApplyTargetSlew(
        recovered_envelope->command.q_target[i], last_successfully_published_q[i],
        /*has_previously_published=*/true, recovery_step,
        /*enforce_slew_limit=*/true);
    Expect(std::fabs(recovered_target - last_successfully_published_q[i]) <=
               max_recovery_delta + 1.0e-6F,
           "recovery after canonical damping must remain within 35*min(dt, 2 ms)");
  }
}

void TestMissingOrInvalidMeasurementKeepsTheLastSafeReference() {
  const float last_published = -0.28F;
  const float invalid_measured = std::numeric_limits<float>::quiet_NaN();
  Expect(NearlyEqual(low_command_safety::DampingTargetFromMeasuredQ(
                         invalid_measured, last_published, true),
                     last_published),
         "an invalid measured q must not rebase damping to zero");
  Expect(NearlyEqual(low_command_safety::DampingTargetFromMeasuredQ(
                         invalid_measured, last_published, false),
                     0.0F),
         "only the pre-first-publish case may use the zero q placeholder");
}

void TestDampingReferenceRequiresARecentLowStateSample() {
  using namespace std::chrono_literals;
  using Clock = std::chrono::steady_clock;
  constexpr float last_published = -0.28F;
  constexpr float measured_q = -0.31F;
  const auto now = Clock::time_point {100ms};
  const auto max_age = 10ms;

  Expect(low_command_safety::IsMeasuredDampingReferenceFresh(
             now - max_age, now, max_age),
         "the damping reference freshness boundary must be inclusive");
  Expect(!low_command_safety::IsMeasuredDampingReferenceFresh(
             now - max_age - 1us, now, max_age),
         "a LowState older than the damping-reference bound must be stale");
  Expect(!low_command_safety::IsMeasuredDampingReferenceFresh(
             now + 1us, now, max_age),
         "a future LowState timestamp must never be accepted as measured state");

  Expect(NearlyEqual(low_command_safety::DampingTargetFromFreshMeasuredQ(
                         measured_q, now - 5ms, now, max_age, last_published,
                         true),
                     measured_q),
         "a fresh finite measurement must rebase the damping reference");
  Expect(NearlyEqual(low_command_safety::DampingTargetFromFreshMeasuredQ(
                         std::numeric_limits<float>::quiet_NaN(), now - 5ms,
                         now, max_age, last_published, true),
                     last_published),
         "a fresh non-finite measurement must retain the last successful target");
  Expect(NearlyEqual(low_command_safety::DampingTargetFromFreshMeasuredQ(
                         measured_q, now - 11ms, now, max_age, last_published,
                         true),
                     last_published),
         "a stale measurement must retain the last successful target");
  Expect(NearlyEqual(low_command_safety::DampingTargetFromFreshMeasuredQ(
                         measured_q, now - 11ms, now, max_age, last_published,
                         false),
                     0.0F),
         "only a stale pre-first-publish fallback may use the zero placeholder");
}

void TestEveryPhasedActuatorFieldMustBeFinite() {
  std::array<float, 3> q {0.1F, 0.2F, 0.3F};
  std::array<float, 3> dq {0.0F, 0.0F, 0.0F};
  std::array<float, 3> tau {0.0F, 0.0F, 0.0F};
  std::array<float, 3> kp {12.0F, 12.0F, 12.0F};
  std::array<float, 3> kd {1.0F, 1.0F, 1.0F};
  const auto valid = [&] {
    return low_command_safety::HasOnlyFiniteActuatorFields(q, dq, tau, kp, kd);
  };
  const float invalid = std::numeric_limits<float>::quiet_NaN();

  Expect(valid(), "a fully finite phased actuator command must pass validation");
  q[0] = invalid;
  Expect(!valid(), "a non-finite q target must fail closed");
  q[0] = 0.1F;
  dq[0] = invalid;
  Expect(!valid(), "a non-finite dq target must fail closed");
  dq[0] = 0.0F;
  tau[0] = invalid;
  Expect(!valid(), "a non-finite feedforward torque must fail closed");
  tau[0] = 0.0F;
  kp[0] = invalid;
  Expect(!valid(), "a non-finite proportional gain must fail closed");
  kp[0] = 12.0F;
  kd[0] = std::numeric_limits<float>::infinity();
  Expect(!valid(), "a non-finite derivative gain must fail closed");
}

}  // namespace

int main() {
  TestCanonicalDampingFieldsCoverEveryG1Joint();
  TestCanonicalDampingRetainsEveryLastSafeReferenceWhenUnavailable();
  TestSlewLimitUsesWriterCadence();
  TestEventWakesCannotBorrowAFullWriterStep();
  TestWriterStallCannotAccumulateARecoveryJump();
  TestMeasuredDampingDoesNotRebaseToZero();
  TestDampingCommitsMeasuredReferenceAndPolicyRecoversFromIt();
  TestLateGenerationFallsBackToFreshMeasuredDampingThenRecovers();
  TestMissingOrInvalidMeasurementKeepsTheLastSafeReference();
  TestDampingReferenceRequiresARecentLowStateSample();
  TestEveryPhasedActuatorFieldMustBeFinite();
  std::cout << "low command safety tests passed\n";
  return EXIT_SUCCESS;
}

#include "command_freshness_fence.hpp"
#include "low_command_safety.hpp"

#include <algorithm>
#include <vector>

#include "policy_parameters.hpp"

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
  std::array<float, 3> tau_ff {};
  std::array<float, 3> q_target {};
  std::array<float, 3> dq_target {};
  std::array<float, 3> kp {};
  std::array<float, 3> kd {};
};

void TestDampingTargetsFollowMeasuredPositions() {
  constexpr std::array<float, 4> measured {-0.312F, 0.669F, -0.363F, 0.2F};
  const auto targets = low_command_safety::MeasuredDampingTargets<measured.size()>(
      [&measured](std::size_t index) { return measured[index]; });
  Expect(targets == measured,
         "damping q targets must use the current measured joint positions");
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

  constexpr std::array<float, 3> last_full_gain_targets {0.60F, -0.60F, 0.75F};
  constexpr std::array<float, 3> fresh_measured_q {-0.31F, 0.42F, -0.18F};
  constexpr std::array<float, 3> recovered_policy_q {0.70F, -0.50F, 0.65F};
  constexpr auto kWriterPeriod = 2ms;
  // Exercise the writer-stall cap explicitly: recovery arrives 7 ms after
  // the successful damping write, but it may use no more than one 2 ms step.
  constexpr auto kRecoveryDt = 7ms;
  constexpr auto kDampingReferenceMaxAge = 10ms;
  const auto damping_time = Fence::Clock::time_point {std::chrono::seconds {1}};

  Fence fence;
  TestActuatorCommand late_command;
  late_command.q_target = last_full_gain_targets;
  late_command.kp = {12.0F, 12.0F, 12.0F};
  late_command.kd = {1.0F, 1.0F, 1.0F};
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

  // Model MakeDampingCommand followed by the canonical kd-only sanitizer
  // path. The fresh measured reference must replace the old policy target,
  // even when that replacement is larger than a full-gain slew step.
  TestActuatorCommand damping;
  const float one_packet_step = low_command_safety::SlewStepForElapsedTime(
      Q_TARGET_SLEW_LIMIT, kWriterPeriod, kWriterPeriod);
  for (std::size_t i = 0; i < damping.q_target.size(); ++i) {
    const float measured_target = low_command_safety::DampingTargetFromFreshMeasuredQ(
        fresh_measured_q[i], damping_time - 1ms, damping_time,
        kDampingReferenceMaxAge, last_full_gain_targets[i],
        /*has_previously_published=*/true);
    damping.q_target[i] = low_command_safety::ApplyTargetSlew(
        measured_target, last_full_gain_targets[i],
        /*has_previously_published=*/true, one_packet_step,
        /*enforce_slew_limit=*/false);
    damping.tau_ff[i] = 0.0F;
    damping.dq_target[i] = 0.0F;
    damping.kp[i] = 0.0F;
    damping.kd[i] = 8.0F;

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
  const std::array<float, 3> last_successfully_published_q = damping.q_target;

  // A new, in-phase policy envelope may recover after that successful local
  // damping write. Its full-gain q targets must ramp from the measured
  // damping reference, not from the stale policy target.
  const auto recovery_time = damping_time + kRecoveryDt;
  TestActuatorCommand recovery_command;
  recovery_command.q_target = recovered_policy_q;
  recovery_command.kp = {12.0F, 12.0F, 12.0F};
  recovery_command.kd = {1.0F, 1.0F, 1.0F};
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
  TestDampingTargetsFollowMeasuredPositions();
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

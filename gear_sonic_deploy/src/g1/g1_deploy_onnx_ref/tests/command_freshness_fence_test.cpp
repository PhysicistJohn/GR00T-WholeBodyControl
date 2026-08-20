#include "command_freshness_fence.hpp"

#include <mutex>

#include "utils.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>

namespace {

using namespace std::chrono_literals;

struct TestMotorCommand {
  std::array<float, 3> tau_ff{};
  std::array<float, 3> q_target{};
  std::array<float, 3> dq_target{};
  std::array<float, 3> kp{};
  std::array<float, 3> kd{};
};

using Fence = CommandFreshnessFence<TestMotorCommand>;
using Envelope = Fence::Envelope;

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "command_freshness_fence_test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    Fail(message);
  }
}

TestMotorCommand StandHoldCommand() {
  TestMotorCommand command;
  command.q_target = {0.1F, -0.2F, 0.3F};
  command.kp = {12.0F, 12.0F, 12.0F};
  command.kd = {1.0F, 1.0F, 1.0F};
  return command;
}

TestMotorCommand DampingCommand() {
  TestMotorCommand command;
  command.tau_ff = {0.0F, 0.0F, 0.0F};
  command.q_target = {0.0F, 0.0F, 0.0F};
  command.dq_target = {0.0F, 0.0F, 0.0F};
  command.kp = {0.0F, 0.0F, 0.0F};
  command.kd = {8.0F, 8.0F, 8.0F};
  return command;
}

std::shared_ptr<const Envelope> CommandAt(std::uint64_t generation,
                                          Fence::Clock::time_point receipt_time) {
  return std::make_shared<const Envelope>(Envelope{
      .command = StandHoldCommand(),
      .source_low_state_receipt = receipt_time,
      .source_generation = generation,
  });
}

const TestMotorCommand& SelectOrDamping(const Fence::Decision& decision,
                                        const std::shared_ptr<const Envelope>& candidate,
                                        const TestMotorCommand& damping) {
  return decision.IsAccepted() ? candidate->command : damping;
}

void TestOneControlCommandCanServeRepeatedWriterTicks() {
  Fence fence;
  const auto receipt = Fence::Clock::time_point{100s};
  const auto command = CommandAt(7, receipt);

  Expect(fence.Evaluate(command, receipt).IsAccepted(),
         "a fresh command must be accepted at its LowState receipt");
  Expect(fence.Evaluate(command, receipt + 2ms).IsAccepted(),
         "the same command generation must be reusable at 500 Hz");
  Expect(fence.Evaluate(command, receipt + Fence::kMaxCommandAge).IsAccepted(),
         "the command lease must be inclusive at exactly 50 ms");
}

void TestCommandExpiresFromLowStateReceipt() {
  Fence fence;
  const auto now = Fence::Clock::time_point{200s};
  const auto expired = CommandAt(8, now - Fence::kMaxCommandAge - 1us);

  Expect(fence.Evaluate(expired, now).result == Fence::Result::kExpired,
         "a command must expire immediately after its 50 ms lease");
}

void TestLateBufferWriteCannotFreshenAnOldCommand() {
  Fence fence;
  DataBuffer<Envelope> buffer;
  const auto now = Fence::Clock::now();
  const Envelope stale{
      .command = StandHoldCommand(),
      .source_low_state_receipt = now - Fence::kMaxCommandAge - 1ms,
      .source_generation = 9,
  };

  // This SetData happens now, after a simulated producer stall.  Its buffer
  // timestamp must not participate in the writer fence decision.
  buffer.SetData(stale);
  const auto late_write = buffer.GetDataWithTime();
  Expect(late_write.HasData(), "the late buffer write must be observable");
  Expect(late_write.timestamp > stale.source_low_state_receipt,
         "the test must distinguish buffer-write time from source receipt time");
  Expect(fence.Evaluate(late_write.data, Fence::Clock::now()).result == Fence::Result::kExpired,
         "a new buffer write must not revive an old LowState command");
}

void TestFutureReceiptsAreRejected() {
  Fence fence;
  const auto now = Fence::Clock::time_point{300s};

  Expect(fence.Evaluate(CommandAt(10, now + 1us), now).result == Fence::Result::kFutureReceipt,
         "a command whose LowState receipt is in the future must be rejected");
}

void TestGenerationCannotRegress() {
  Fence fence;
  const auto now = Fence::Clock::time_point{400s};
  Expect(fence.Evaluate(CommandAt(21, now - 1ms), now).IsAccepted(),
         "the first fresh generation must be accepted");
  Expect(fence.Evaluate(CommandAt(20, now - 1ms), now).result ==
             Fence::Result::kGenerationRegressed,
         "a lower generation must not replace an accepted newer generation");
}

void TestFreshNewGenerationRecoversFromDamping() {
  Fence fence;
  const auto now = Fence::Clock::time_point{500s};
  const auto expired = CommandAt(30, now - Fence::kMaxCommandAge - 1us);
  Expect(fence.Evaluate(expired, now).result == Fence::Result::kExpired,
         "the writer must enter damping when its command lease expires");
  Expect(fence.Evaluate(CommandAt(31, now - 1ms), now).IsAccepted(),
         "a fresh newer LowState generation must restore command output");
}

void TestRejectedCommandsSelectKnownDampingShape() {
  Fence fence;
  const auto now = Fence::Clock::time_point{600s};
  const std::shared_ptr<const Envelope> missing;
  const TestMotorCommand damping = DampingCommand();
  const auto decision = fence.Evaluate(missing, now);
  const TestMotorCommand& selected = SelectOrDamping(decision, missing, damping);

  Expect(!decision.IsAccepted(), "a missing command must be rejected");
  for (std::size_t i = 0; i < selected.tau_ff.size(); ++i) {
    Expect(selected.tau_ff[i] == 0.0F, "damping torque feedforward must be zero");
    Expect(selected.kp[i] == 0.0F, "damping proportional gain must be zero");
    Expect(selected.kd[i] == 8.0F, "damping derivative gain must be eight");
  }
}

void TestInitAndWaitLeasesRenewThenExpire() {
  Fence fence;
  const auto init_receipt = Fence::Clock::time_point{700s};
  const auto init_hold = CommandAt(40, init_receipt);
  Expect(fence.Evaluate(init_hold, init_receipt + 49ms).IsAccepted(),
         "the INIT stand-hold lease must be valid before expiry");

  // WAIT_FOR_CONTROL renews its neutral stand hold using a newer LowState
  // snapshot before the INIT lease expires.
  const auto wait_receipt = init_receipt + 49ms;
  const auto wait_hold = CommandAt(41, wait_receipt);
  Expect(fence.Evaluate(wait_hold, init_receipt + 98ms).IsAccepted(),
         "a fresh WAIT_FOR_CONTROL stand-hold lease must renew output");
  Expect(fence.Evaluate(wait_hold, wait_receipt + Fence::kMaxCommandAge + 1us).result ==
             Fence::Result::kExpired,
         "a stalled INIT or WAIT producer must expire to damping");
}

}  // namespace

int main() {
  TestOneControlCommandCanServeRepeatedWriterTicks();
  TestCommandExpiresFromLowStateReceipt();
  TestLateBufferWriteCannotFreshenAnOldCommand();
  TestFutureReceiptsAreRejected();
  TestGenerationCannotRegress();
  TestFreshNewGenerationRecoversFromDamping();
  TestRejectedCommandsSelectKnownDampingShape();
  TestInitAndWaitLeasesRenewThenExpire();
  std::cout << "command freshness fence tests passed\n";
  return EXIT_SUCCESS;
}

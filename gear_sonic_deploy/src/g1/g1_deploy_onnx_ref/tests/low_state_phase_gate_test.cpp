#include "low_state_phase_gate.hpp"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <string_view>
#include <thread>

namespace {

using namespace std::chrono_literals;

struct LowStateSample {
  int sequence = 0;
};

using PhaseGate = LowStatePhaseGate<LowStateSample>;

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "low_state_phase_gate_test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    Fail(message);
  }
}

PhaseGate::Snapshot SnapshotAt(int sequence, PhaseGate::Clock::time_point receipt_time) {
  return {std::make_shared<const LowStateSample>(LowStateSample{sequence}), 1, receipt_time};
}

void TestFreshLowStateNeedsNoSecondaryTelemetry() {
  PhaseGate gate;
  LowStateSample low_state{7};
  const auto receipt_time = PhaseGate::Clock::now() - 1ms;
  gate.NotifyLowState(low_state, receipt_time);
  low_state.sequence = 99;

  const auto outcome = gate.WaitForFreshLowState(1s, 0ms);

  Expect(outcome.result == PhaseGate::Result::kReady,
         "a fresh LowState must be ready without a secondary-IMU receipt");
  Expect(outcome.snapshot.low_state->sequence == 7,
         "the returned LowState must be an immutable copy");
}

void TestFreshnessBoundariesArePure() {
  const auto receipt_time = PhaseGate::Clock::time_point{10s};
  const auto snapshot = SnapshotAt(7, receipt_time);

  Expect(PhaseGate::IsFreshLowState(snapshot, receipt_time, 1ms),
         "a snapshot must be fresh at its receipt time");
  Expect(PhaseGate::IsFreshLowState(snapshot, receipt_time + 1ms, 1ms),
         "the receipt-age bound must be inclusive");
  Expect(!PhaseGate::IsFreshLowState(snapshot, receipt_time + 1001us, 1ms),
         "a snapshot beyond the receipt-age bound must be stale");
  Expect(!PhaseGate::IsFreshLowState(snapshot, receipt_time - 1us, 1ms),
         "a receipt after the consumer time must be rejected");
  Expect(!PhaseGate::IsFreshLowState(snapshot, receipt_time, -1ms),
         "a negative freshness bound must be rejected");
}

void TestStaleLowStateIsRejected() {
  PhaseGate gate;
  gate.NotifyLowState({1}, PhaseGate::Clock::now() - 1min);

  const auto outcome = gate.WaitForFreshLowState(1ms, 0ms);

  Expect(outcome.result == PhaseGate::Result::kTimeout,
         "a stale LowState must not be accepted");
}

void TestNotificationRaceDoesNotLoseWake() {
  // Notify immediately after the waiter announces that it is about to enter
  // the gate.  The notification may land before or after the waiter takes the
  // mutex; either way generation tracking makes the fresh LowState visible.
  for (int iteration = 0; iteration < 64; ++iteration) {
    PhaseGate gate;
    gate.NotifyLowState({1}, PhaseGate::Clock::now() - 1min);

    std::promise<void> waiter_started;
    std::promise<PhaseGate::WaitOutcome> waiter_result;
    auto started = waiter_started.get_future();
    auto result = waiter_result.get_future();
    std::thread waiter([&] {
      waiter_started.set_value();
      waiter_result.set_value(gate.WaitForFreshLowState(30s, 30s));
    });

    started.wait();
    gate.NotifyLowState({iteration + 2});

    if (result.wait_for(5s) != std::future_status::ready) {
      gate.Stop();
      waiter.join();
      Fail("a concurrent LowState notification did not wake the waiter");
    }
    const auto outcome = result.get();
    waiter.join();
    Expect(outcome.result == PhaseGate::Result::kReady,
           "a concurrent LowState notification must not be lost");
    Expect(outcome.snapshot.low_state->sequence == iteration + 2,
           "the returned snapshot must be the post-wait LowState");
  }
}

void TestStopIsTerminal() {
  PhaseGate gate;
  gate.Stop();

  const auto outcome = gate.WaitForFreshLowState(1s, 1s);

  Expect(outcome.result == PhaseGate::Result::kStopped,
         "Stop must make pending and subsequent waits report stopped");
}

void TestStopUnblocksWait() {
  PhaseGate gate;
  gate.NotifyLowState({1}, PhaseGate::Clock::now() - 1min);

  std::promise<void> waiter_started;
  std::promise<PhaseGate::WaitOutcome> waiter_result;
  auto started = waiter_started.get_future();
  auto result = waiter_result.get_future();
  std::thread waiter([&] {
    waiter_started.set_value();
    waiter_result.set_value(gate.WaitForFreshLowState(1ms, 30s));
  });

  started.wait();
  gate.Stop();
  if (result.wait_for(5s) != std::future_status::ready) {
    gate.Stop();
    waiter.join();
    Fail("Stop did not unblock a pending LowState wait");
  }
  const auto outcome = result.get();
  waiter.join();
  Expect(outcome.result == PhaseGate::Result::kStopped,
         "Stop must unblock a LowState wait with the stopped result");
}

}  // namespace

int main() {
  TestFreshLowStateNeedsNoSecondaryTelemetry();
  TestFreshnessBoundariesArePure();
  TestStaleLowStateIsRejected();
  TestNotificationRaceDoesNotLoseWake();
  TestStopIsTerminal();
  TestStopUnblocksWait();
  std::cout << "low state phase gate tests passed\n";
  return EXIT_SUCCESS;
}

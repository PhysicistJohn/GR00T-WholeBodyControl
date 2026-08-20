#include "low_state_phase_gate.hpp"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string_view>
#include <thread>

namespace {

using namespace std::chrono_literals;

struct LowStateSample {
  int sequence = 0;
};

struct TorsoImuSample {
  int sequence = 0;
};

using PhaseGate = LowStatePhaseGate<LowStateSample, TorsoImuSample>;

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "low_state_phase_gate_test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    Fail(message);
  }
}

void SeedStalePair(PhaseGate& gate) {
  const auto stale_time = PhaseGate::Clock::now() - 1min;
  gate.NotifyLowState({1}, stale_time);
  gate.NotifyTorsoImu({1}, stale_time);
}

void TestAlreadyFreshReturnsImmediately() {
  PhaseGate gate;
  LowStateSample low_state{7};
  TorsoImuSample torso_imu{11};
  gate.NotifyLowState(low_state);
  gate.NotifyTorsoImu(torso_imu);
  low_state.sequence = 99;
  torso_imu.sequence = 99;

  const auto start = PhaseGate::Clock::now();
  const auto outcome = gate.WaitForFreshPair(100ms, 2ms, 250ms);
  const auto elapsed = PhaseGate::Clock::now() - start;

  Expect(outcome.result == PhaseGate::Result::kReady,
         "an already-fresh pair must be ready");
  Expect(elapsed < 100ms, "an already-fresh pair must not wait for a new sample");
  Expect(outcome.pair.low_state->sequence == 7 && outcome.pair.torso_imu->sequence == 11,
         "the returned pair must own immutable copies of both messages");
}

void TestReadyPairCanExpireBeforeConsumption() {
  PhaseGate gate;
  const auto receipt_time = PhaseGate::Clock::now();
  gate.NotifyLowState({7}, receipt_time);
  gate.NotifyTorsoImu({11}, receipt_time);

  const auto outcome = gate.WaitForFreshPair(100ms, 2ms, 250ms);
  Expect(outcome.result == PhaseGate::Result::kReady,
         "the initial pair must be ready before testing post-wake expiry");
  Expect(PhaseGate::IsFreshPair(outcome.pair, receipt_time + 500us, 1ms, 2ms),
         "a pair inside the receipt-age bound must remain consumable");
  Expect(!PhaseGate::IsFreshPair(outcome.pair, receipt_time - 1us, 1ms, 2ms),
         "a pair with a receipt time after consumption must be rejected");
  std::this_thread::sleep_for(5ms);
  Expect(!PhaseGate::IsFreshPair(outcome.pair, PhaseGate::Clock::now(), 1ms, 2ms),
         "a ready pair that aged before consumption must be rejected");
}

void TestStalePairWaitsForBothNotifications() {
  PhaseGate gate;
  SeedStalePair(gate);

  std::promise<void> waiter_started;
  std::promise<PhaseGate::WaitOutcome> waiter_result;
  auto started = waiter_started.get_future();
  auto result = waiter_result.get_future();
  std::thread waiter([&] {
    waiter_started.set_value();
    waiter_result.set_value(gate.WaitForFreshPair(100ms, 2ms, 500ms));
  });

  started.wait();
  std::this_thread::sleep_for(20ms);
  gate.NotifyLowState({2});
  std::this_thread::sleep_for(20ms);
  Expect(result.wait_for(0ms) == std::future_status::timeout,
         "a new LowState alone must not release the paired gate");
  // The first post-entry LowState is paired with the stale IMU receipt.
  // Publish a new coherent LowState/IMU pair to satisfy both requirements.
  gate.NotifyLowState({3});
  gate.NotifyTorsoImu({3});

  const auto outcome = result.get();
  waiter.join();
  Expect(outcome.result == PhaseGate::Result::kReady,
         "a stale pair must wake after both streams advance");
  Expect(outcome.pair.low_state_generation >= 3 && outcome.pair.torso_imu_generation >= 2,
         "the ready pair must include both post-snapshot notifications");
  Expect(outcome.pair.low_state->sequence == 3 && outcome.pair.torso_imu->sequence == 3,
         "the returned pair must contain the coherent post-entry samples");
}

void TestTimeout() {
  PhaseGate gate;
  SeedStalePair(gate);

  const auto start = PhaseGate::Clock::now();
  const auto outcome = gate.WaitForFreshPair(100ms, 2ms, 60ms);
  const auto elapsed = PhaseGate::Clock::now() - start;

  Expect(outcome.result == PhaseGate::Result::kTimeout,
         "a stale pair with no notifications must time out");
  Expect(elapsed >= 40ms && elapsed < 500ms, "timeout must honor the bounded wait");
}

void TestSkewedPairIsNotAccepted() {
  PhaseGate gate;
  const auto now = PhaseGate::Clock::now();
  gate.NotifyLowState({1}, now);
  gate.NotifyTorsoImu({1}, now - 5ms);

  const auto outcome = gate.WaitForFreshPair(100ms, 2ms, 60ms);
  Expect(outcome.result == PhaseGate::Result::kTimeout,
         "a pair exceeding the allowed receipt skew must not be accepted");
}

void TestNotificationRaceDoesNotLoseWake() {
  // Repeatedly notify just as the waiter starts.  A check-then-wait
  // implementation without a generation predicate can lose this wake and
  // intermittently time out; the gate's single mutex makes every iteration
  // either immediately fresh or visibly advanced.
  for (int iteration = 0; iteration < 64; ++iteration) {
    PhaseGate gate;
    SeedStalePair(gate);

    std::promise<void> waiter_started;
    std::promise<PhaseGate::WaitOutcome> waiter_result;
    auto started = waiter_started.get_future();
    auto result = waiter_result.get_future();
    std::thread waiter([&] {
      waiter_started.set_value();
      // Keep the freshness and skew bounds comfortably above the bounded
      // wait. This test isolates the generation/condition-variable handoff
      // rather than treating an unrelated host scheduling delay as a lost
      // wake.
      waiter_result.set_value(gate.WaitForFreshPair(30s, 30s, 300ms));
    });

    started.wait();
    gate.NotifyLowState({iteration + 2});
    gate.NotifyTorsoImu({iteration + 2});

    const auto outcome = result.get();
    waiter.join();
    Expect(outcome.result == PhaseGate::Result::kReady,
           "a concurrent pair notification must not be lost");
  }
}

void TestStopWakesWaiter() {
  PhaseGate gate;
  SeedStalePair(gate);

  std::promise<void> waiter_started;
  std::promise<PhaseGate::WaitOutcome> waiter_result;
  auto started = waiter_started.get_future();
  auto result = waiter_result.get_future();
  std::thread waiter([&] {
    waiter_started.set_value();
    waiter_result.set_value(gate.WaitForFreshPair(100ms, 2ms, 2s));
  });

  started.wait();
  std::this_thread::sleep_for(20ms);
  gate.Stop();

  Expect(result.wait_for(300ms) == std::future_status::ready,
         "Stop must wake a pending phase wait promptly");
  const auto outcome = result.get();
  waiter.join();
  Expect(outcome.result == PhaseGate::Result::kStopped,
         "Stop must report the stopped result");
}

}  // namespace

int main() {
  TestAlreadyFreshReturnsImmediately();
  TestReadyPairCanExpireBeforeConsumption();
  TestStalePairWaitsForBothNotifications();
  TestTimeout();
  TestSkewedPairIsNotAccepted();
  TestNotificationRaceDoesNotLoseWake();
  TestStopWakesWaiter();
  std::cout << "low state phase gate tests passed\n";
  return EXIT_SUCCESS;
}

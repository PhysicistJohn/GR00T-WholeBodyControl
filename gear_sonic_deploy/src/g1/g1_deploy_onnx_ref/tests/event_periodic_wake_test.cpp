#include "event_periodic_wake.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string_view>
#include <thread>

namespace {

using namespace std::chrono_literals;

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "event_periodic_wake_test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    Fail(message);
  }
}

void StopAndJoin(EventPeriodicWake& wake, std::thread& worker) {
  wake.Stop();
  if (worker.joinable()) {
    worker.join();
  }
}

void TestEventWakeIsPrompt() {
  EventPeriodicWake wake(10s);
  std::promise<void> first_callback;
  std::promise<EventPeriodicWake::Clock::time_point> event_callback;
  auto first_ready = first_callback.get_future();
  auto event_ready = event_callback.get_future();
  std::atomic<int> callbacks{0};

  std::thread worker([&] {
    wake.Run([&] {
      const int callback_number = ++callbacks;
      if (callback_number == 1) {
        first_callback.set_value();
      } else if (callback_number == 2) {
        event_callback.set_value(EventPeriodicWake::Clock::now());
      }
    });
  });

  if (first_ready.wait_for(1s) != std::future_status::ready) {
    StopAndJoin(wake, worker);
    Fail("the initial callback did not run");
  }
  const auto notification_time = EventPeriodicWake::Clock::now();
  wake.Notify();
  if (event_ready.wait_for(100ms) != std::future_status::ready) {
    StopAndJoin(wake, worker);
    Fail("an event did not wake a 10 second maintenance loop promptly");
  }
  const auto callback_time = event_ready.get();
  StopAndJoin(wake, worker);

  Expect(callback_time >= notification_time,
         "the event callback timestamp must follow notification");
  Expect(callback_time - notification_time <= 100ms,
         "event wake must not wait for the periodic maintenance deadline");
}

void TestNotificationDuringCallbackIsNotLost() {
  EventPeriodicWake wake(10s);
  std::promise<void> second_callback;
  auto second_ready = second_callback.get_future();
  std::atomic<int> callbacks{0};

  std::thread worker([&] {
    wake.Run([&] {
      const int callback_number = ++callbacks;
      if (callback_number == 1) {
        // This exercises the race where a producer publishes while the writer
        // is running rather than sleeping in wait_until().
        wake.Notify();
      } else if (callback_number == 2) {
        second_callback.set_value();
      }
    });
  });

  if (second_ready.wait_for(100ms) != std::future_status::ready) {
    StopAndJoin(wake, worker);
    Fail("a notification during the callback was lost");
  }
  StopAndJoin(wake, worker);
  Expect(callbacks.load() >= 2,
         "the notification generation must schedule a follow-up callback");
}

void TestEarlyEventDoesNotPostponePeriodicMaintenance() {
  constexpr auto kPeriod = 20ms;
  EventPeriodicWake wake(kPeriod);
  std::promise<EventPeriodicWake::Clock::time_point> first_callback;
  std::promise<EventPeriodicWake::Clock::time_point> third_callback;
  auto first_ready = first_callback.get_future();
  auto third_ready = third_callback.get_future();
  std::atomic<int> callbacks{0};

  std::thread worker([&] {
    wake.Run([&] {
      const int callback_number = ++callbacks;
      const auto now = EventPeriodicWake::Clock::now();
      if (callback_number == 1) {
        first_callback.set_value(now);
      } else if (callback_number == 3) {
        third_callback.set_value(now);
      }
    });
  });

  if (first_ready.wait_for(1s) != std::future_status::ready) {
    StopAndJoin(wake, worker);
    Fail("the initial callback did not run before cadence test");
  }
  const auto first_time = first_ready.get();
  std::this_thread::sleep_for(2ms);
  wake.Notify();
  if (third_ready.wait_for(300ms) != std::future_status::ready) {
    StopAndJoin(wake, worker);
    Fail("an early event prevented the next periodic callback");
  }
  const auto third_time = third_ready.get();
  StopAndJoin(wake, worker);

  // The event is callback two. Callback three must remain the original 20 ms
  // maintenance slot, rather than being deferred to 40 ms after callback one.
  Expect(third_time - first_time >= 15ms,
         "the periodic tick must not collapse into an immediate duplicate event callback");
  Expect(third_time - first_time <= 35ms,
         "an early event must not postpone the next periodic maintenance tick");
}

void TestPeriodicMaintenanceRetransmits() {
  EventPeriodicWake wake(5ms);
  std::promise<void> third_callback;
  auto third_ready = third_callback.get_future();
  std::atomic<int> callbacks{0};

  std::thread worker([&] {
    wake.Run([&] {
      if (++callbacks == 3) {
        third_callback.set_value();
      }
    });
  });

  if (third_ready.wait_for(250ms) != std::future_status::ready) {
    StopAndJoin(wake, worker);
    Fail("maintenance callbacks did not retransmit without notifications");
  }
  StopAndJoin(wake, worker);
  Expect(callbacks.load() >= 3,
         "three periodic callbacks must run without a producer notification");
}

void TestStopUnblocksIdleWriter() {
  EventPeriodicWake wake(10s);
  std::promise<void> first_callback;
  auto first_ready = first_callback.get_future();

  std::thread worker([&] {
    wake.Run([&] { first_callback.set_value(); });
  });

  if (first_ready.wait_for(1s) != std::future_status::ready) {
    StopAndJoin(wake, worker);
    Fail("the writer did not enter its idle wait");
  }
  const auto stop_time = EventPeriodicWake::Clock::now();
  StopAndJoin(wake, worker);
  Expect(EventPeriodicWake::Clock::now() - stop_time <= 100ms,
         "Stop must unblock an idle periodic wait without waiting 10 seconds");
}

}  // namespace

int main() {
  TestEventWakeIsPrompt();
  TestNotificationDuringCallbackIsNotLost();
  TestEarlyEventDoesNotPostponePeriodicMaintenance();
  TestPeriodicMaintenanceRetransmits();
  TestStopUnblocksIdleWriter();
  std::cout << "event periodic wake tests passed\n";
  return EXIT_SUCCESS;
}

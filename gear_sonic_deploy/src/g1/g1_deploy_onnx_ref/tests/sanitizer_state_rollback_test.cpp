#include "sanitizer_state_rollback.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using namespace std::chrono_literals;

using Targets = std::array<float, 3>;
using Clock = std::chrono::steady_clock;
using Rollback = SanitizerStateRollback<Targets, Clock::time_point>;

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "sanitizer_state_rollback_test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void Expect(bool condition, std::string_view message) {
  if (!condition) { Fail(message); }
}

void TestFailedLocalWriteRollsBackEverySanitizerField() {
  Targets last_targets {0.1F, -0.2F, 0.3F};
  bool has_last_targets = true;
  std::uint64_t event_count = 17;
  const Clock::time_point last_log_time {42s};
  Clock::time_point mutable_last_log_time = last_log_time;

  {
    Rollback rollback(last_targets, has_last_targets, event_count, mutable_last_log_time);
    // Model SanitizeLowCommand mutating all state before Write() reports false.
    last_targets = {0.4F, 0.5F, 0.6F};
    has_last_targets = false;
    event_count = 23;
    mutable_last_log_time = Clock::time_point {99s};
  }

  Expect(last_targets == Targets {0.1F, -0.2F, 0.3F}, "a failed local write must restore the slew reference");
  Expect(has_last_targets, "a failed local write must restore whether a slew reference existed");
  Expect(event_count == 17, "a failed local write must restore sanitizer accounting");
  Expect(mutable_last_log_time == last_log_time, "a failed local write must restore sanitizer log throttling state");
}

void TestSuccessfulLocalWriteCommitsSanitizerState() {
  Targets last_targets {0.1F, -0.2F, 0.3F};
  bool has_last_targets = false;
  std::uint64_t event_count = 0;
  Clock::time_point last_log_time {};

  {
    Rollback rollback(last_targets, has_last_targets, event_count, last_log_time);
    last_targets = {0.4F, 0.5F, 0.6F};
    has_last_targets = true;
    event_count = 6;
    last_log_time = Clock::time_point {99s};
    rollback.Commit();
  }

  Expect(last_targets == Targets {0.4F, 0.5F, 0.6F}, "a successful local write must retain the new slew reference");
  Expect(has_last_targets, "a successful local write must retain the initialized slew reference");
  Expect(event_count == 6, "a successful local write must retain sanitizer accounting");
  Expect(last_log_time == Clock::time_point {99s},
         "a successful local write must retain sanitizer log throttling state");
}

void TestExplicitRollbackIsIdempotent() {
  Targets last_targets {0.1F, -0.2F, 0.3F};
  bool has_last_targets = true;
  std::uint64_t event_count = 17;
  Clock::time_point last_log_time {42s};

  {
    Rollback rollback(last_targets, has_last_targets, event_count, last_log_time);
    last_targets = {0.4F, 0.5F, 0.6F};
    has_last_targets = false;
    event_count = 23;
    last_log_time = Clock::time_point {99s};
    rollback.Restore();
  }

  Expect(last_targets == Targets {0.1F, -0.2F, 0.3F}, "an explicit rollback must not be applied twice by destruction");
  Expect(has_last_targets && event_count == 17 && last_log_time == Clock::time_point {42s},
         "an explicit rollback must restore every sanitizer field");
}

} // namespace

int main() {
  TestFailedLocalWriteRollsBackEverySanitizerField();
  TestSuccessfulLocalWriteCommitsSanitizerState();
  TestExplicitRollbackIsIdempotent();
  std::cout << "sanitizer state rollback tests passed\n";
  return EXIT_SUCCESS;
}

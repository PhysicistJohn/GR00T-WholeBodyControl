#include "sanitizer_state_rollback.hpp"

#include <array>
#include <chrono>
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
  const Clock::time_point last_target_time {21s};
  Clock::time_point mutable_last_target_time = last_target_time;

  {
    Rollback rollback(last_targets, has_last_targets, mutable_last_target_time);
    // Model the pre-write sanitizer state changing before Write() reports false.
    last_targets = {0.4F, 0.5F, 0.6F};
    has_last_targets = false;
    mutable_last_target_time = Clock::time_point {88s};
  }

  Expect(last_targets == Targets {0.1F, -0.2F, 0.3F}, "a failed local write must restore the slew reference");
  Expect(has_last_targets, "a failed local write must restore whether a slew reference existed");
  Expect(mutable_last_target_time == last_target_time,
         "a failed local write must restore the slew-reference timestamp");
}

void TestSuccessfulLocalWriteCommitsSanitizerState() {
  Targets last_targets {0.1F, -0.2F, 0.3F};
  bool has_last_targets = false;
  Clock::time_point last_target_time {};

  {
    Rollback rollback(last_targets, has_last_targets, last_target_time);
    last_targets = {0.4F, 0.5F, 0.6F};
    has_last_targets = true;
    last_target_time = Clock::time_point {88s};
    rollback.Commit();
  }

  Expect(last_targets == Targets {0.4F, 0.5F, 0.6F}, "a successful local write must retain the new slew reference");
  Expect(has_last_targets, "a successful local write must retain the initialized slew reference");
  Expect(last_target_time == Clock::time_point {88s},
         "a successful local write must retain the slew-reference timestamp");
}

void TestExplicitRollbackIsIdempotent() {
  Targets last_targets {0.1F, -0.2F, 0.3F};
  bool has_last_targets = true;
  const Clock::time_point last_target_time {21s};
  Clock::time_point mutable_last_target_time = last_target_time;

  {
    Rollback rollback(last_targets, has_last_targets, mutable_last_target_time);
    last_targets = {0.4F, 0.5F, 0.6F};
    has_last_targets = false;
    mutable_last_target_time = Clock::time_point {88s};
    rollback.Restore();
  }

  Expect(last_targets == Targets {0.1F, -0.2F, 0.3F}, "an explicit rollback must not be applied twice by destruction");
  Expect(has_last_targets, "an explicit rollback must restore whether a slew reference existed");
  Expect(mutable_last_target_time == last_target_time,
         "an explicit rollback must restore the slew-reference timestamp");
}

} // namespace

int main() {
  TestFailedLocalWriteRollsBackEverySanitizerField();
  TestSuccessfulLocalWriteCommitsSanitizerState();
  TestExplicitRollbackIsIdempotent();
  std::cout << "sanitizer state rollback tests passed\n";
  return EXIT_SUCCESS;
}

#include "writer_telemetry_mailbox.hpp"

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>

namespace {

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "writer_telemetry_mailbox_test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    Fail(message);
  }
}

void TestMailboxPublishesOnlyCompleteNewSnapshots() {
  writer_telemetry::LatestSnapshotMailbox<3> mailbox;
  writer_telemetry::LatestSnapshotMailbox<3>::Payload snapshot {};
  std::uint64_t observed_sequence = 0;

  Expect(!mailbox.TryRead(snapshot, observed_sequence),
         "an empty mailbox must not invent telemetry");

  const writer_telemetry::LatestSnapshotMailbox<3>::Payload first {
      50, writer_telemetry::PackDouble(2999.5), 3000};
  mailbox.Publish(first);
  Expect(mailbox.TryRead(snapshot, observed_sequence),
         "a published writer snapshot must become observable");
  Expect(snapshot == first, "the reader must receive one coherent payload");
  Expect(!mailbox.TryRead(snapshot, observed_sequence),
         "the same completed snapshot must not be emitted twice");

  const writer_telemetry::LatestSnapshotMailbox<3>::Payload second {
      100, writer_telemetry::PackDouble(812.25), 3000};
  mailbox.Publish(second);
  Expect(mailbox.TryRead(snapshot, observed_sequence),
         "a later writer snapshot must replace the earlier one");
  Expect(snapshot == second, "the latest snapshot must be intact");
  Expect(writer_telemetry::UnpackDouble(snapshot[1]) == 812.25,
         "double timing fields must round-trip through lock-free words");
}

void TestConcurrentReaderNeverObservesATornSnapshot() {
  constexpr std::uint64_t kPublications = 20'000;
  writer_telemetry::LatestSnapshotMailbox<3> mailbox;
  std::atomic<bool> producer_done {false};
  std::thread producer([&] {
    for (std::uint64_t value = 1; value <= kPublications; ++value) {
      mailbox.Publish({
          value,
          ~value,
          writer_telemetry::PackDouble(static_cast<double>(value) + 0.5),
      });
    }
    producer_done.store(true, std::memory_order_release);
  });

  writer_telemetry::LatestSnapshotMailbox<3>::Payload snapshot {};
  std::uint64_t observed_sequence = 0;
  std::uint64_t last_value = 0;
  const auto Validate = [&] {
    Expect(snapshot[1] == ~snapshot[0],
           "the reader must never observe fields from different snapshots");
    Expect(writer_telemetry::UnpackDouble(snapshot[2]) ==
               static_cast<double>(snapshot[0]) + 0.5,
           "encoded timing must match the same snapshot generation");
    last_value = snapshot[0];
  };
  while (!producer_done.load(std::memory_order_acquire)) {
    if (mailbox.TryRead(snapshot, observed_sequence)) {
      Validate();
    } else {
      std::this_thread::yield();
    }
  }
  producer.join();
  if (mailbox.TryRead(snapshot, observed_sequence)) {
    Validate();
  }
  Expect(last_value == kPublications,
         "the final completed producer snapshot must be observable");
}

}  // namespace

int main() {
  TestMailboxPublishesOnlyCompleteNewSnapshots();
  TestConcurrentReaderNeverObservesATornSnapshot();
  std::cout << "writer telemetry mailbox tests passed\n";
  return EXIT_SUCCESS;
}

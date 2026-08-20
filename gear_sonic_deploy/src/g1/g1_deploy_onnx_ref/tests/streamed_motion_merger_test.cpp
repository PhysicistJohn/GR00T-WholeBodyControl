#include "motion_data_reader.hpp"
#include "input_interface/streamed_motion_merger.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

namespace {

StreamedMotionMerger::IncomingData frame(std::int64_t index, double yaw_marker) {
  StreamedMotionMerger::IncomingData data;
  data.joint_pos = {{yaw_marker}};
  data.joint_vel = {{0.0}};
  data.body_quat = {{{yaw_marker, 0.0, 0.0, 1.0}}};
  data.frame_indices = {index};
  data.protocol_version = 1;
  data.num_frames = 1;
  data.num_joints = 1;
  data.num_pos_bodies = 0;
  data.num_quat_bodies = 1;
  return data;
}

}  // namespace

int main() {
  StreamedMotionMerger merger;

  auto first = merger.MergeIncomingData(frame(0, 0.0), 0);
  assert(first.motion);
  assert(first.did_catchup_reset);

  auto contiguous = merger.MergeIncomingData(frame(1, 1.0), 0);
  assert(contiguous.motion);
  assert(!contiguous.did_catchup_reset);

  // Frames 2 and 3 were dropped. Real-time playback must jump to frame 4
  // without changing the physical heading anchor.
  auto dropped = merger.MergeIncomingData(frame(4, 4.0), 1);
  assert(dropped.motion);
  assert(!dropped.did_catchup_reset);
  assert(dropped.window_start == 4);
  assert(dropped.motion->timesteps == 1);
  assert(dropped.motion->JointPositions(0)[0] == 4.0);

  // A delayed stale packet must not replace the current window or reset yaw.
  auto stale = merger.MergeIncomingData(frame(3, 3.0), 0);
  assert(stale.motion);
  assert(!stale.did_catchup_reset);
  assert(stale.window_start == 4);
  assert(stale.motion->JointPositions(0)[0] == 4.0);

  // A deliberate large forward discontinuity still identifies a new stream.
  auto new_stream = merger.MergeIncomingData(frame(1000, 1000.0), 0);
  assert(new_stream.motion);
  assert(new_stream.did_catchup_reset);
  assert(new_stream.window_start == 1000);

  std::cout << "streamed motion merger gap policy passed\n";
  return 0;
}

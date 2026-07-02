"""Independent safety watchdog for the G1 deploy stack.

Runs as a SEPARATE process (on the onboard Jetson for hardware, or any host
on the robot's DDS domain) so the kill path does not depend on the controller
process staying alive. The controller's own stop is cooperative — if its
control thread wedges, nothing damps the robot. This daemon closes that gap.

Monitors:
  * rt/lowcmd staleness — the controller publishes at 500 Hz; once commands
    have been seen, silence longer than --lowcmd-timeout-s means the
    controller died or wedged mid-motion.
  * rt/lowstate joint velocity — any |dq| above --dq-limit (default 40 rad/s,
    above the controller's own 35 rad/s abort) means runaway.
  * torso tilt deviation from the orientation captured at watchdog start,
    sustained for --tilt-hold-s. Deviation-based on purpose: it works whether
    the IMU frame reads "upright" as identity or as ~180 deg.
  * operator e-stop: touch --estop-file (default /tmp/g1_estop).

On trigger (latched — manual restart required):
  1. SIGTERM then SIGKILL any local process matching --controller-pattern
     (skipped with --no-kill, e.g. when the controller runs on another host).
  2. Hand the low-level channel back to Unitree firmware via
     MotionSwitcherClient.SelectMode(--select-mode), then LocoClient.Damp().
     This is the manufacturer's own damping path. In MuJoCo sim these RPC
     services do not exist; the calls time out and are logged — the sim test
     only exercises detection + kill.

VERIFY ON HARDWARE before trusting: the SelectMode name for the G1 EDU
firmware ("ai" vs "normal") and that Damp() is accepted right after a mode
re-acquire. Bench-test with the robot on the gantry.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_


def log(msg: str) -> None:
    print(f"[watchdog] {time.strftime('%H:%M:%S')} {msg}", flush=True)


class Monitor:
    def __init__(self) -> None:
        self.last_cmd_t: float | None = None
        self.last_state: LowState_ | None = None
        self.last_state_t: float | None = None

    def on_cmd(self, _msg: LowCmd_) -> None:
        self.last_cmd_t = time.monotonic()

    def on_state(self, msg: LowState_) -> None:
        self.last_state = msg
        self.last_state_t = time.monotonic()


def quat_angle_deg(q_ref: tuple, q_now: tuple) -> float:
    """Angle between two [x, y, z, w] quaternions, in degrees."""
    dot = abs(sum(a * b for a, b in zip(q_ref, q_now)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def kill_controller(pattern: str) -> None:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        pids = [int(p) for p in out.stdout.split() if int(p) != os.getpid()]
    except Exception as exc:
        log(f"pgrep failed: {exc}")
        return
    if not pids:
        log(f"no local process matches '{pattern}' (controller remote or already dead)")
        return
    for pid in pids:
        log(f"SIGTERM {pid}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        alive = [p for p in pids if os.path.exists(f"/proc/{p}")]
        if not alive:
            return
        time.sleep(0.1)
    for pid in [p for p in pids if os.path.exists(f"/proc/{p}")]:
        log(f"SIGKILL {pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def firmware_damp(select_mode: str) -> None:
    """Give the low-level channel back to Unitree firmware, then damp."""
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
        msc = MotionSwitcherClient()
        msc.SetTimeout(2.0)
        msc.Init()
        code, mode = msc.CheckMode()
        log(f"motion_switcher CheckMode -> code={code} mode={mode}")
        name = (mode or {}).get("name", "") if isinstance(mode, dict) else ""
        if not name:
            code = msc.SelectMode(select_mode)
            log(f"motion_switcher SelectMode('{select_mode}') -> code={code}")
    except Exception as exc:
        log(f"motion_switcher unavailable ({exc}) — sim, or robot unreachable")
    try:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        loco = LocoClient()
        loco.SetTimeout(2.0)
        loco.Init()
        code = loco.Damp()
        log(f"loco Damp() -> code={code}")
    except Exception as exc:
        log(f"loco client unavailable ({exc}) — sim, or robot unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--interface", default="lo",
                    help="DDS network interface (robot eth on hardware, lo in sim)")
    ap.add_argument("--channel", type=int, default=0, help="DDS domain id")
    ap.add_argument("--lowcmd-timeout-s", type=float, default=0.3)
    ap.add_argument("--dq-limit", type=float, default=40.0)
    ap.add_argument("--tilt-dev-deg", type=float, default=60.0,
                    help="torso tilt deviation from boot orientation; 0 disables")
    ap.add_argument("--tilt-hold-s", type=float, default=0.3)
    ap.add_argument("--estop-file", default="/tmp/g1_estop")
    ap.add_argument("--controller-pattern", default="g1_deploy_onnx_ref")
    ap.add_argument("--no-kill", action="store_true",
                    help="do not kill local controller processes on trigger")
    ap.add_argument("--select-mode", default="ai",
                    help="motion_switcher mode name to re-acquire (verify on hardware)")
    ap.add_argument("--dry-run", action="store_true",
                    help="log triggers but take no action")
    args = ap.parse_args()

    # Stale sentinel from a previous session must not instant-trigger.
    if os.path.exists(args.estop_file):
        os.remove(args.estop_file)
        log(f"removed stale e-stop sentinel {args.estop_file}")

    ChannelFactoryInitialize(args.channel, args.interface)
    mon = Monitor()
    state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    state_sub.Init(mon.on_state, 16)
    cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    cmd_sub.Init(mon.on_cmd, 16)
    log(f"armed on {args.interface} domain {args.channel} "
        f"(lowcmd timeout {args.lowcmd_timeout_s}s, dq limit {args.dq_limit}, "
        f"tilt dev {args.tilt_dev_deg} deg, estop {args.estop_file})")

    boot_quat: tuple | None = None
    tilt_bad_since: float | None = None

    def trigger(reason: str) -> None:
        log(f"TRIGGER: {reason}")
        if args.dry_run:
            log("dry-run: no action taken")
            return
        if not args.no_kill:
            kill_controller(args.controller_pattern)
        firmware_damp(args.select_mode)
        log("latched — restart the watchdog to re-arm")

    while True:
        time.sleep(0.02)
        now = time.monotonic()

        if os.path.exists(args.estop_file):
            trigger(f"operator e-stop ({args.estop_file})")
            break

        if mon.last_cmd_t is not None and now - mon.last_cmd_t > args.lowcmd_timeout_s:
            trigger(f"rt/lowcmd stale {now - mon.last_cmd_t:.2f}s — controller dead mid-motion")
            break

        s = mon.last_state
        if s is None or mon.last_state_t is None or now - mon.last_state_t > 0.5:
            continue

        dq_max = max(abs(float(m.dq)) for m in s.motor_state[:29])
        if dq_max > args.dq_limit:
            trigger(f"joint velocity runaway |dq|={dq_max:.1f} > {args.dq_limit}")
            break

        if args.tilt_dev_deg > 0:
            q = tuple(float(v) for v in s.imu_state.quaternion)
            if boot_quat is None:
                boot_quat = q
                log(f"boot orientation captured (quat {tuple(round(v, 3) for v in q)})")
                continue
            dev = quat_angle_deg(boot_quat, q)
            if dev > args.tilt_dev_deg:
                if tilt_bad_since is None:
                    tilt_bad_since = now
                elif now - tilt_bad_since > args.tilt_hold_s:
                    trigger(f"tilt deviation {dev:.0f} deg > {args.tilt_dev_deg} deg "
                            f"for {args.tilt_hold_s}s")
                    break
            else:
                tilt_bad_since = None

    return 0


if __name__ == "__main__":
    sys.exit(main())

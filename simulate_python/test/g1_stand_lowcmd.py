#!/home/meldose/.pyenv/versions/unitree310/bin/python
import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

import sys
from pathlib import Path

# Ensure we import the local simulate_python/config.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


G1_ACTIVE_MOTORS = 29


class G1JointIndex:
    LeftHipPitch = 0
    LeftKnee = 3
    LeftAnklePitch = 4
    RightHipPitch = 6
    RightKnee = 9
    RightAnklePitch = 10
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14


def main() -> int:
    parser = argparse.ArgumentParser(description="Stand the G1 in MuJoCo via low-level commands.")
    parser.add_argument("--stand-z", type=float, default=0.85, help="Base height (for reference only).")
    parser.add_argument("--hip", type=float, default=-0.5, help="Hip pitch (rad)")
    parser.add_argument("--knee", type=float, default=1.0, help="Knee pitch (rad)")
    parser.add_argument("--ankle", type=float, default=-0.5, help="Ankle pitch (rad)")
    parser.add_argument("--waist-pitch", type=float, default=0.2, help="Waist pitch (rad)")
    parser.add_argument("--waist-roll", type=float, default=0.0, help="Waist roll (rad)")
    parser.add_argument("--ramp", type=float, default=2.0, help="Time to ramp into stand pose (s)")
    parser.add_argument("--hold", type=float, default=5.0, help="Hold time after standing (s)")
    parser.add_argument("--kp", type=float, default=120.0, help="Position gain for stand joints")
    parser.add_argument("--kd", type=float, default=3.0, help="Velocity gain for stand joints")
    parser.add_argument("--kp-hold", type=float, default=40.0, help="Hold gain for other joints")
    parser.add_argument("--kd-hold", type=float, default=1.5, help="Hold D gain for other joints")
    args = parser.parse_args()

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)

    baseline_q = {"values": None}
    mode_machine = {"value": 0}

    def _lowstate_handler(msg: LowState_):
        if baseline_q["values"] is None:
            baseline_q["values"] = [msg.motor_state[i].q for i in range(G1_ACTIVE_MOTORS)]
        mode_machine["value"] = msg.mode_machine

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(_lowstate_handler, 10)

    print("[g1_stand_lowcmd] Waiting for lowstate...")
    t_wait = time.time()
    while baseline_q["values"] is None and time.time() - t_wait < 5.0:
        time.sleep(0.05)
    if baseline_q["values"] is None:
        print("[g1_stand_lowcmd] No lowstate received. Is the sim running?")
        return 1

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()

    cmd = unitree_hg_msg_dds__LowCmd_()
    for i in range(G1_ACTIVE_MOTORS):
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].q = baseline_q["values"][i]
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kp = args.kp_hold
        cmd.motor_cmd[i].kd = args.kd_hold
        cmd.motor_cmd[i].tau = 0.0

    start = time.time()
    print("[g1_stand_lowcmd] Ramping to stand pose...")
    while True:
        now = time.time()
        t = now - start

        cmd.mode_pr = 0
        cmd.mode_machine = mode_machine["value"]

        ratio = min(1.0, t / max(0.1, args.ramp))

        # Hold all joints at their initial pose by default.
        for i in range(G1_ACTIVE_MOTORS):
            cmd.motor_cmd[i].q = baseline_q["values"][i]
            cmd.motor_cmd[i].kp = args.kp_hold
            cmd.motor_cmd[i].kd = args.kd_hold

        # Blend legs and waist to target stand angles.
        for joint, target in (
            (G1JointIndex.LeftHipPitch, args.hip),
            (G1JointIndex.LeftKnee, args.knee),
            (G1JointIndex.LeftAnklePitch, args.ankle),
            (G1JointIndex.RightHipPitch, args.hip),
            (G1JointIndex.RightKnee, args.knee),
            (G1JointIndex.RightAnklePitch, args.ankle),
            (G1JointIndex.WaistPitch, args.waist_pitch),
            (G1JointIndex.WaistRoll, args.waist_roll),
        ):
            start_q = baseline_q["values"][joint]
            cmd.motor_cmd[joint].q = (1.0 - ratio) * start_q + ratio * target
            cmd.motor_cmd[joint].kp = args.kp
            cmd.motor_cmd[joint].kd = args.kd

        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        if t > args.ramp + args.hold:
            break
        time.sleep(0.005)

    print("[g1_stand_lowcmd] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

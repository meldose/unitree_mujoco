#!/home/meldose/.pyenv/versions/unitree310/bin/python
import argparse
import math
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


G1_NUM_MOTOR = 35
G1_ACTIVE_MOTORS = 29


class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11


def main() -> int:
    parser = argparse.ArgumentParser(description="Low-level leg wiggle for G1 sim.")
    parser.add_argument("--freq", type=float, default=0.5, help="Oscillation frequency (Hz)")
    parser.add_argument("--amp-swing", type=float, default=0.25, help="Hip swing amplitude (rad)")
    parser.add_argument("--amp-knee", type=float, default=0.45, help="Knee amplitude (rad)")
    parser.add_argument("--amp-ankle", type=float, default=0.2, help="Ankle amplitude (rad)")
    parser.add_argument("--stand-time", type=float, default=3.0, help="Time to hold a stand pose (s)")
    parser.add_argument("--stand-hip", type=float, default=-0.4, help="Stand pose hip pitch (rad)")
    parser.add_argument("--stand-knee", type=float, default=0.8, help="Stand pose knee (rad)")
    parser.add_argument("--stand-ankle", type=float, default=-0.4, help="Stand pose ankle pitch (rad)")
    args = parser.parse_args()

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)

    # Capture the current joint positions so we can oscillate around them.
    baseline_q = {"values": None}
    mode_machine = {"value": 0}

    def _lowstate_handler(msg: LowState_):
        if baseline_q["values"] is None:
            baseline_q["values"] = [msg.motor_state[i].q for i in range(G1_NUM_MOTOR)]
        mode_machine["value"] = msg.mode_machine

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(_lowstate_handler, 10)

    print("[g1_wiggle_lowcmd] Waiting for lowstate...")
    t_wait = time.time()
    while baseline_q["values"] is None and time.time() - t_wait < 5.0:
        time.sleep(0.05)
    if baseline_q["values"] is None:
        print("[g1_wiggle_lowcmd] No lowstate received. Is the sim running?")
        return 1

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()

    cmd = unitree_hg_msg_dds__LowCmd_()

    # Basic PD settings.
    kp_hold = 60.0
    kd_hold = 2.0
    kp_leg = 80.0
    kd_leg = 2.0
    kp_stand = 120.0
    kd_stand = 3.0

    for i in range(G1_NUM_MOTOR):
        cmd.motor_cmd[i].mode = 1  # enable
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    print("[g1_wiggle_lowcmd] Sending low-level joint commands. Ctrl+C to stop.")
    t0 = time.time()

    try:
        while True:
            t = time.time() - t0
            cmd.mode_pr = 0
            cmd.mode_machine = mode_machine["value"]

            freq = max(0.01, args.freq)
            swing = args.amp_swing * math.sin(2.0 * math.pi * freq * t)
            knee = args.amp_knee * math.sin(2.0 * math.pi * freq * t + math.pi / 2.0)
            ankle = -args.amp_ankle * math.sin(2.0 * math.pi * freq * t + math.pi / 2.0)

            # Stand-up phase: drive a simple upright pose before wiggling.
            if t < args.stand_time:
                for i in range(G1_ACTIVE_MOTORS):
                    cmd.motor_cmd[i].q = 0.0
                    cmd.motor_cmd[i].kp = kp_hold
                    cmd.motor_cmd[i].kd = kd_hold

                cmd.motor_cmd[G1JointIndex.LeftHipPitch].q = args.stand_hip
                cmd.motor_cmd[G1JointIndex.LeftKnee].q = args.stand_knee
                cmd.motor_cmd[G1JointIndex.LeftAnklePitch].q = args.stand_ankle

                cmd.motor_cmd[G1JointIndex.RightHipPitch].q = args.stand_hip
                cmd.motor_cmd[G1JointIndex.RightKnee].q = args.stand_knee
                cmd.motor_cmd[G1JointIndex.RightAnklePitch].q = args.stand_ankle

                for j in (
                    G1JointIndex.LeftHipPitch,
                    G1JointIndex.LeftKnee,
                    G1JointIndex.LeftAnklePitch,
                    G1JointIndex.RightHipPitch,
                    G1JointIndex.RightKnee,
                    G1JointIndex.RightAnklePitch,
                ):
                    cmd.motor_cmd[j].kp = kp_stand
                    cmd.motor_cmd[j].kd = kd_stand
            else:
                # Hold current pose for all active joints.
                for i in range(G1_ACTIVE_MOTORS):
                    cmd.motor_cmd[i].q = baseline_q["values"][i]
                    cmd.motor_cmd[i].kp = kp_hold
                    cmd.motor_cmd[i].kd = kd_hold

                # Left leg
                cmd.motor_cmd[G1JointIndex.LeftHipPitch].q = baseline_q["values"][G1JointIndex.LeftHipPitch] + swing
                cmd.motor_cmd[G1JointIndex.LeftHipPitch].kp = kp_leg
                cmd.motor_cmd[G1JointIndex.LeftHipPitch].kd = kd_leg

                cmd.motor_cmd[G1JointIndex.LeftKnee].q = baseline_q["values"][G1JointIndex.LeftKnee] + knee
                cmd.motor_cmd[G1JointIndex.LeftKnee].kp = kp_leg
                cmd.motor_cmd[G1JointIndex.LeftKnee].kd = kd_leg

                cmd.motor_cmd[G1JointIndex.LeftAnklePitch].q = baseline_q["values"][G1JointIndex.LeftAnklePitch] + ankle
                cmd.motor_cmd[G1JointIndex.LeftAnklePitch].kp = kp_leg
                cmd.motor_cmd[G1JointIndex.LeftAnklePitch].kd = kd_leg

                # Right leg (opposite phase)
                cmd.motor_cmd[G1JointIndex.RightHipPitch].q = baseline_q["values"][G1JointIndex.RightHipPitch] - swing
                cmd.motor_cmd[G1JointIndex.RightHipPitch].kp = kp_leg
                cmd.motor_cmd[G1JointIndex.RightHipPitch].kd = kd_leg

                cmd.motor_cmd[G1JointIndex.RightKnee].q = baseline_q["values"][G1JointIndex.RightKnee] - knee
                cmd.motor_cmd[G1JointIndex.RightKnee].kp = kp_leg
                cmd.motor_cmd[G1JointIndex.RightKnee].kd = kd_leg

                cmd.motor_cmd[G1JointIndex.RightAnklePitch].q = baseline_q["values"][G1JointIndex.RightAnklePitch] - ankle
                cmd.motor_cmd[G1JointIndex.RightAnklePitch].kp = kp_leg
                cmd.motor_cmd[G1JointIndex.RightAnklePitch].kd = kd_leg

            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)
            time.sleep(0.005)  # 200 Hz
    except KeyboardInterrupt:
        print("[g1_wiggle_lowcmd] Stopping.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

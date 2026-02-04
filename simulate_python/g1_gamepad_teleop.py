#!/home/meldose/.pyenv/versions/unitree310/bin/python
"""
g1_gamepad_teleop.py

WSL-friendly gamepad teleop for the G1 MuJoCo sim.

Flow:
  Windows: gamepad_udp_sender.py -> WSL: gamepad_udp_receiver.py
  This script subscribes to rt/wirelesscontroller and publishes rt/lowcmd.

Controls (default):
  Left stick Y: forward/back
  Left stick X: turn (yaw)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.utils.crc import CRC

# Ensure we import the local simulate_python/config.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


TOPIC_WIRELESS = "rt/wirelesscontroller"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_LOWCMD = "rt/lowcmd"

G1_ACTIVE_MOTORS_29 = 29
G1_ACTIVE_MOTORS_23 = 23


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


def _detect_dof() -> int:
    scene_path = Path(config.ROBOT_SCENE).expanduser()
    if not scene_path.is_absolute():
        scene_path = (Path(__file__).resolve().parent / scene_path).resolve()
    try:
        text = scene_path.read_text()
    except OSError:
        return G1_ACTIVE_MOTORS_29
    if "g1_23dof.xml" in text or "23dof" in text:
        return G1_ACTIVE_MOTORS_23
    return G1_ACTIVE_MOTORS_29


def _deadzone(val: float, dz: float) -> float:
    if abs(val) < dz:
        return 0.0
    return val


class GamepadTeleop:
    def __init__(self, dof: int):
        self.dof = dof
        self.baseline_q: list[float] | None = None
        self.mode_machine = 0
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.last_wireless = time.time()

        self.pub = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
        self.sub_state = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.sub_wireless = ChannelSubscriber(TOPIC_WIRELESS, WirelessController_)
        self.crc = CRC()

    def _lowstate_handler(self, msg: LowState_):
        if self.baseline_q is None:
            self.baseline_q = [msg.motor_state[i].q for i in range(self.dof)]
        self.mode_machine = msg.mode_machine

    def _wireless_handler(self, msg: WirelessController_):
        self.lx = msg.lx
        self.ly = msg.ly
        self.rx = msg.rx
        self.ry = msg.ry
        self.last_wireless = time.time()

    def start(self):
        self.pub.Init()
        self.sub_state.Init(self._lowstate_handler, 10)
        self.sub_wireless.Init(self._wireless_handler, 10)

        t_wait = time.time()
        while self.baseline_q is None and time.time() - t_wait < 5.0:
            time.sleep(0.05)
        if self.baseline_q is None:
            raise RuntimeError("No lowstate received. Is the sim running?")

    def run(
        self,
        rate_hz: float,
        max_v: float,
        max_yaw: float,
        freq: float,
        hip_offset: float,
        knee_offset: float,
        ankle_offset: float,
        hip_amp: float,
        knee_amp: float,
        ankle_amp: float,
        kp: float,
        kd: float,
        kp_hold: float,
        kd_hold: float,
        deadzone: float,
        wireless_timeout: float,
    ):
        cmd = unitree_hg_msg_dds__LowCmd_()
        for i in range(self.dof):
            cmd.motor_cmd[i].mode = 1
            cmd.motor_cmd[i].q = self.baseline_q[i]
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = kp_hold
            cmd.motor_cmd[i].kd = kd_hold
            cmd.motor_cmd[i].tau = 0.0

        period = 1.0 / max(rate_hz, 1.0)
        start = time.time()

        while True:
            now = time.time()

            if now - self.last_wireless > wireless_timeout:
                vx = 0.0
                yaw = 0.0
            else:
                vx = _deadzone(self.ly, deadzone) * max_v
                yaw = _deadzone(self.lx, deadzone) * max_yaw

            phase = 2.0 * math.pi * freq * (now - start)
            left = math.sin(phase)
            right = math.sin(phase + math.pi)

            cmd.mode_pr = 0
            cmd.mode_machine = self.mode_machine

            for i in range(self.dof):
                cmd.motor_cmd[i].q = self.baseline_q[i]
                cmd.motor_cmd[i].kp = kp_hold
                cmd.motor_cmd[i].kd = kd_hold

            amp_scale = max(0.0, min(1.0, abs(vx)))
            yaw_scale = max(-1.0, min(1.0, yaw / max(max_yaw, 1e-6)))

            left_scale = amp_scale * (1.0 + 0.5 * yaw_scale)
            right_scale = amp_scale * (1.0 - 0.5 * yaw_scale)

            cmd.motor_cmd[G1JointIndex.LeftHipPitch].q = hip_offset + hip_amp * left_scale * left
            cmd.motor_cmd[G1JointIndex.LeftKnee].q = knee_offset + knee_amp * left_scale * math.sin(phase + math.pi / 2.0)
            cmd.motor_cmd[G1JointIndex.LeftAnklePitch].q = ankle_offset + ankle_amp * left_scale * math.sin(phase + math.pi / 2.0)

            cmd.motor_cmd[G1JointIndex.RightHipPitch].q = hip_offset + hip_amp * right_scale * right
            cmd.motor_cmd[G1JointIndex.RightKnee].q = knee_offset + knee_amp * right_scale * math.sin(phase + math.pi / 2.0 + math.pi)
            cmd.motor_cmd[G1JointIndex.RightAnklePitch].q = ankle_offset + ankle_amp * right_scale * math.sin(phase + math.pi / 2.0 + math.pi)

            for j in (
                G1JointIndex.LeftHipPitch,
                G1JointIndex.LeftKnee,
                G1JointIndex.LeftAnklePitch,
                G1JointIndex.RightHipPitch,
                G1JointIndex.RightKnee,
                G1JointIndex.RightAnklePitch,
            ):
                cmd.motor_cmd[j].kp = kp
                cmd.motor_cmd[j].kd = kd

            cmd.crc = self.crc.Crc(cmd)
            self.pub.Write(cmd)
            time.sleep(period)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gamepad teleop for G1 MuJoCo sim.")
    parser.add_argument("--rate", type=float, default=200.0, help="Publish rate (Hz)")
    parser.add_argument("--max-v", type=float, default=0.8, help="Max forward speed scale")
    parser.add_argument("--max-yaw", type=float, default=0.8, help="Max yaw scale")
    parser.add_argument("--freq", type=float, default=1.2, help="Gait frequency (Hz)")
    parser.add_argument("--deadzone", type=float, default=0.08, help="Stick deadzone")
    parser.add_argument("--wireless-timeout", type=float, default=0.5, help="Stop if no gamepad updates (s)")
    parser.add_argument("--dof", type=int, choices=[23, 29], default=0, help="Force DOF (23 or 29)")

    parser.add_argument("--hip-offset", type=float, default=-0.5)
    parser.add_argument("--knee-offset", type=float, default=1.0)
    parser.add_argument("--ankle-offset", type=float, default=-0.5)
    parser.add_argument("--hip-amp", type=float, default=0.3)
    parser.add_argument("--knee-amp", type=float, default=0.4)
    parser.add_argument("--ankle-amp", type=float, default=-0.2)

    parser.add_argument("--kp", type=float, default=160.0)
    parser.add_argument("--kd", type=float, default=8.0)
    parser.add_argument("--kp-hold", type=float, default=60.0)
    parser.add_argument("--kd-hold", type=float, default=3.0)

    args = parser.parse_args()

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)

    dof = args.dof if args.dof in (23, 29) else _detect_dof()
    teleop = GamepadTeleop(G1_ACTIVE_MOTORS_23 if dof == 23 else G1_ACTIVE_MOTORS_29)
    teleop.start()

    print(f"[g1_gamepad_teleop] Running: dof={dof}, rate={args.rate} Hz")
    try:
        teleop.run(
            rate_hz=args.rate,
            max_v=args.max_v,
            max_yaw=args.max_yaw,
            freq=max(0.1, args.freq),
            hip_offset=args.hip_offset,
            knee_offset=args.knee_offset,
            ankle_offset=args.ankle_offset,
            hip_amp=args.hip_amp,
            knee_amp=args.knee_amp,
            ankle_amp=args.ankle_amp,
            kp=args.kp,
            kd=args.kd,
            kp_hold=args.kp_hold,
            kd_hold=args.kd_hold,
            deadzone=args.deadzone,
            wireless_timeout=args.wireless_timeout,
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

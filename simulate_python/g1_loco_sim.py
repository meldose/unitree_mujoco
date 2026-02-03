#!/home/meldose/.pyenv/versions/unitree310/bin/python
import argparse
import math
import time
from pathlib import Path
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

# Ensure we import the local simulate_python/config.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


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


class LocoSim:
    def __init__(self, dof: int):
        self.dof = dof
        self.baseline_q = None
        self.mode_machine = 0
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.crc = CRC()

    def _lowstate_handler(self, msg: LowState_):
        if self.baseline_q is None:
            self.baseline_q = [msg.motor_state[i].q for i in range(self.dof)]
        self.mode_machine = msg.mode_machine

    def start(self):
        self.pub.Init()
        self.sub.Init(self._lowstate_handler, 10)

        t_wait = time.time()
        while self.baseline_q is None and time.time() - t_wait < 5.0:
            time.sleep(0.05)
        if self.baseline_q is None:
            raise RuntimeError("No lowstate received. Is the sim running?")

    def run(self, vx: float, vy: float, yaw: float, freq: float,
            hip_offset: float, knee_offset: float, ankle_offset: float,
            hip_amp: float, knee_amp: float, ankle_amp: float,
            kp: float, kd: float, kp_hold: float, kd_hold: float,
            duration: float):
        cmd = unitree_hg_msg_dds__LowCmd_()
        for i in range(self.dof):
            cmd.motor_cmd[i].mode = 1
            cmd.motor_cmd[i].q = self.baseline_q[i]
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = kp_hold
            cmd.motor_cmd[i].kd = kd_hold
            cmd.motor_cmd[i].tau = 0.0

        vx_scale = max(0.0, min(1.0, abs(vx)))
        start = time.time()
        while True:
            t = time.time() - start
            if duration > 0 and t > duration:
                break

            phase = 2.0 * math.pi * freq * t
            left = math.sin(phase)
            right = math.sin(phase + math.pi)

            cmd.mode_pr = 0
            cmd.mode_machine = self.mode_machine

            # Hold all joints at baseline.
            for i in range(self.dof):
                cmd.motor_cmd[i].q = self.baseline_q[i]
                cmd.motor_cmd[i].kp = kp_hold
                cmd.motor_cmd[i].kd = kd_hold

            # Simple alternating gait: hip/knee/ankle on pitch joints.
            cmd.motor_cmd[G1JointIndex.LeftHipPitch].q = hip_offset + hip_amp * vx_scale * left
            cmd.motor_cmd[G1JointIndex.LeftKnee].q = knee_offset + knee_amp * vx_scale * math.sin(phase + math.pi / 2.0)
            cmd.motor_cmd[G1JointIndex.LeftAnklePitch].q = ankle_offset + ankle_amp * vx_scale * math.sin(phase + math.pi / 2.0)

            cmd.motor_cmd[G1JointIndex.RightHipPitch].q = hip_offset + hip_amp * vx_scale * right
            cmd.motor_cmd[G1JointIndex.RightKnee].q = knee_offset + knee_amp * vx_scale * math.sin(phase + math.pi / 2.0 + math.pi)
            cmd.motor_cmd[G1JointIndex.RightAnklePitch].q = ankle_offset + ankle_amp * vx_scale * math.sin(phase + math.pi / 2.0 + math.pi)

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
            time.sleep(0.005)


def main() -> int:
    parser = argparse.ArgumentParser(description="High-level locomotion shim for G1 sim.")
    parser.add_argument("--vx", type=float, default=0.2, help="Forward velocity [0..1]")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity (unused)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw rate (unused)")
    parser.add_argument("--freq", type=float, default=1.0, help="Gait frequency (Hz)")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run (0 = infinite)")
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
    loco = LocoSim(G1_ACTIVE_MOTORS_23 if dof == 23 else G1_ACTIVE_MOTORS_29)
    loco.start()

    print(f"[g1_loco_sim] Running: dof={dof}, vx={args.vx}")
    try:
        loco.run(
            vx=args.vx,
            vy=args.vy,
            yaw=args.yaw,
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
            duration=args.duration,
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

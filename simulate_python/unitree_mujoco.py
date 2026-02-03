#!/home/meldose/.pyenv/versions/unitree310/bin/python
import argparse
import time
import sys
from pathlib import Path
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

# Ensure we import the local simulate_python/config.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


locker = threading.Lock()

def _parse_args():
    parser = argparse.ArgumentParser(description="Unitree MuJoCo sim runner.")
    parser.add_argument("--stand-z", type=float, default=None, help="Base height for a stand pose.")
    parser.add_argument("--stand-hip", type=float, default=None, help="Hip pitch (rad).")
    parser.add_argument("--stand-knee", type=float, default=None, help="Knee pitch (rad).")
    parser.add_argument("--stand-ankle", type=float, default=None, help="Ankle pitch (rad).")
    parser.add_argument("--pin-base", action="store_true", help="Pin base pose for a few seconds.")
    parser.add_argument("--pin-seconds", type=float, default=2.0, help="Seconds to pin the base pose.")
    parser.add_argument("--startup-pd-hold", action="store_true", help="Apply joint PD hold at startup.")
    parser.add_argument("--pd-hold-seconds", type=float, default=2.0, help="Seconds to apply PD hold.")
    parser.add_argument("--pd-kp", type=float, default=120.0, help="PD hold position gain.")
    parser.add_argument("--pd-kd", type=float, default=6.0, help="PD hold velocity gain.")
    parser.add_argument("--print-base", action="store_true", help="Print pelvis base position periodically.")
    parser.add_argument("--auto-sign", action="store_true", help="Auto-detect leg bend direction for stand pose.")
    parser.add_argument("--auto-stand-z", action="store_true", help="Auto-tune stand z so feet touch ground.")
    return parser.parse_args()


args = _parse_args()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)
pelvis_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
_last_base_print = time.time()
left_foot_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
right_foot_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

def _jnt_id(name: str) -> int:
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)

def _set_joint_qpos(joint_name: str, value: float):
    jid = _jnt_id(joint_name)
    if jid < 0:
        return
    qadr = mj_model.jnt_qposadr[jid]
    mj_data.qpos[qadr] = value

def _apply_leg_pose(hip: float, knee: float, ankle: float):
    _set_joint_qpos("left_hip_pitch_joint", hip)
    _set_joint_qpos("left_knee_joint", knee)
    _set_joint_qpos("left_ankle_pitch_joint", ankle)
    _set_joint_qpos("right_hip_pitch_joint", hip)
    _set_joint_qpos("right_knee_joint", knee)
    _set_joint_qpos("right_ankle_pitch_joint", ankle)

def _pose_score():
    if left_foot_body_id < 0 or right_foot_body_id < 0 or pelvis_body_id < 0:
        return 1e9
    lf = mj_data.xpos[left_foot_body_id][2]
    rf = mj_data.xpos[right_foot_body_id][2]
    pelvis_z = mj_data.xpos[pelvis_body_id][2]
    avg_foot = 0.5 * (lf + rf)
    return abs(avg_foot - 0.0) + max(0.0, (avg_foot + 0.1) - pelvis_z)

# Build actuator -> joint mapping for PD hold
_act_joint = []
for act_id in range(mj_model.nu):
    jnt_id = int(mj_model.actuator_trnid[act_id, 0])
    if jnt_id >= 0:
        _act_joint.append((act_id, jnt_id, mj_model.jnt_qposadr[jnt_id], mj_model.jnt_dofadr[jnt_id]))
    else:
        _act_joint.append((act_id, None, None, None))

# If a keyframe is defined (e.g., a stand pose), initialize the sim to it.
if mj_model.nkey > 0:
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    if pelvis_body_id >= 0:
        pos = mj_data.xpos[pelvis_body_id]
        if pos[2] < 0.3:
            print(f"[unitree_mujoco] Warning: keyframe 0 pelvis z is low ({pos[2]:.3f}).")

# Allow quick override of a simple stand pose (29dof layout) from CLI.
if any(v is not None for v in (args.stand_z, args.stand_hip, args.stand_knee, args.stand_ankle)):
    stand_z = 0.85 if args.stand_z is None else args.stand_z
    hip = -0.5 if args.stand_hip is None else args.stand_hip
    knee = 1.0 if args.stand_knee is None else args.stand_knee
    ankle = -0.5 if args.stand_ankle is None else args.stand_ankle

    # qpos layout: free joint (7) + 29 1-DOF joints.
    mj_data.qpos[:] = 0.0
    mj_data.qpos[2] = stand_z
    mj_data.qpos[3] = 1.0  # identity quaternion

    # Legs (optionally auto-detect sign)
    if args.auto_sign:
        _apply_leg_pose(hip, knee, ankle)
        mujoco.mj_forward(mj_model, mj_data)
        score_a = _pose_score()
        _apply_leg_pose(-hip, -knee, -ankle)
        mujoco.mj_forward(mj_model, mj_data)
        score_b = _pose_score()
        if score_b < score_a:
            hip, knee, ankle = -hip, -knee, -ankle
        _apply_leg_pose(hip, knee, ankle)
    else:
        _apply_leg_pose(hip, knee, ankle)

    mujoco.mj_forward(mj_model, mj_data)
    if args.auto_stand_z and left_foot_body_id >= 0 and right_foot_body_id >= 0:
        lf = mj_data.xpos[left_foot_body_id][2]
        rf = mj_data.xpos[right_foot_body_id][2]
        avg_foot = 0.5 * (lf + rf)
        clearance = 0.02
        dz = -(avg_foot - clearance)
        mj_data.qpos[2] += dz
        mujoco.mj_forward(mj_model, mj_data)
        stand_z = mj_data.qpos[2]
    print(
        f"[unitree_mujoco] Applied stand override: z={stand_z} hip={hip} knee={knee} ankle={ankle} "
        f"(qpos[2]={mj_data.qpos[2]:.3f})"
    )
else:
    print(f"[unitree_mujoco] Stand override not set (qpos[2]={mj_data.qpos[2]:.3f})")

pin_base_until = None
pin_base_qpos = None
pin_base_qvel = None
pd_hold_until = None
pd_hold_qpos = None
if args.pin_base:
    pin_base_until = time.time() + args.pin_seconds
    pin_base_qpos = mj_data.qpos.copy()
    pin_base_qvel = mj_data.qvel.copy()
    print(
        f"[unitree_mujoco] Pinning base and joints for {args.pin_seconds:.2f}s "
        f"(z={pin_base_qpos[2]:.3f})"
    )
if args.startup_pd_hold:
    pd_hold_until = time.time() + args.pd_hold_seconds
    pd_hold_qpos = mj_data.qpos.copy()
    print(f"[unitree_mujoco] Startup PD hold for {args.pd_hold_seconds:.2f}s")


viewer = None
elastic_band = None
band_attached_link = None

if not config.HEADLESS:
    if config.ENABLE_ELASTIC_BAND:
        elastic_band = ElasticBand()
        if config.ROBOT == "h1" or config.ROBOT == "g1":
            band_attached_link = mj_model.body("torso_link").id
        else:
            band_attached_link = mj_model.body("base_link").id
        viewer = mujoco.viewer.launch_passive(
            mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
        )
    else:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model
    global pin_base_until, pin_base_qpos
    global _last_base_print
    global pd_hold_until, pd_hold_qpos

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while True:
        if (viewer is not None) and (not viewer.is_running()):
            break

        step_start = time.perf_counter()

        locker.acquire()

        # Optionally pin all joints (including floating base) for a short time.
        if pin_base_until is not None and pin_base_qpos is not None:
            if time.time() < pin_base_until:
                mj_data.qpos[:] = pin_base_qpos
                mj_data.qvel[:] = 0.0
            else:
                # Disable pin after time window.
                pin_base_qpos = None
                pin_base_qvel = None
                pin_base_until = None

        # Apply startup PD hold after pinning window, if enabled.
        if pd_hold_until is not None and pd_hold_qpos is not None:
            if time.time() < pd_hold_until:
                for act_id, jnt_id, qadr, dadr in _act_joint:
                    if jnt_id is None:
                        continue
                    q = mj_data.qpos[qadr]
                    qd = mj_data.qvel[dadr]
                    q_des = pd_hold_qpos[qadr]
                    tau = args.pd_kp * (q_des - q) - args.pd_kd * qd
                    lo, hi = mj_model.actuator_ctrlrange[act_id]
                    mj_data.ctrl[act_id] = max(lo, min(hi, tau))
            else:
                pd_hold_qpos = None
                pd_hold_until = None

        if config.ENABLE_ELASTIC_BAND and elastic_band is not None:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        if args.print_base and pelvis_body_id >= 0:
            now = time.time()
            if now - _last_base_print > 1.0:
                pos = mj_data.xpos[pelvis_body_id]
                print(f"[unitree_mujoco] pelvis pos: {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
                _last_base_print = now
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    sim_thread = Thread(target=SimulationThread)

    sim_thread.start()
    if viewer is not None:
        viewer_thread = Thread(target=PhysicsViewerThread)
        viewer_thread.start()

    # Keep the main thread alive until simulation ends (or Ctrl+C in headless)
    try:
        sim_thread.join()
    except KeyboardInterrupt:
        pass

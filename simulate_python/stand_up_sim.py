#!/home/meldose/.pyenv/versions/unitree310/bin/python
import argparse
import time
from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np

# Ensure we import the local simulate_python/config.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def _parse_args():
    p = argparse.ArgumentParser(description="Standalone stand-up sim (no DDS).")
    p.add_argument("--stand-z", type=float, default=1.0)
    p.add_argument("--hip", type=float, default=-0.4)
    p.add_argument("--knee", type=float, default=0.8)
    p.add_argument("--ankle", type=float, default=-0.4)
    p.add_argument("--auto-sign", dest="auto_sign", action="store_true", default=True)
    p.add_argument("--no-auto-sign", dest="auto_sign", action="store_false")
    p.add_argument("--auto-stand-z", dest="auto_stand_z", action="store_true", default=True)
    p.add_argument("--no-auto-stand-z", dest="auto_stand_z", action="store_false")
    p.add_argument("--pin-seconds", type=float, default=3.0)
    p.add_argument("--kp", type=float, default=140.0)
    p.add_argument("--kd", type=float, default=8.0)
    p.add_argument("--stable", action="store_true")
    p.add_argument("--lock-stand", dest="lock_stand", action="store_true", default=True)
    p.add_argument("--no-lock-stand", dest="lock_stand", action="store_false")
    p.add_argument("--forward-speed", type=float, default=0.2, help="Drift forward (m/s) when locked or walking.")
    p.add_argument("--gait-walk", dest="gait_walk", action="store_true", default=False)
    p.add_argument("--step-freq", type=float, default=1.0, help="Walking step frequency (Hz).")
    p.add_argument("--hip-amp", type=float, default=0.25, help="Hip pitch amplitude (rad).")
    p.add_argument("--knee-amp", type=float, default=0.35, help="Knee pitch amplitude (rad).")
    p.add_argument("--ankle-amp", type=float, default=0.2, help="Ankle pitch amplitude (rad).")
    p.add_argument("--hip-offset", type=float, default=0.15, help="Forward lean offset for hips (rad).")
    p.add_argument("--swing-knee", type=float, default=0.25, help="Extra knee flex during swing (rad).")
    p.add_argument("--swing-ankle", type=float, default=0.1, help="Extra ankle dorsiflex during swing (rad).")
    p.add_argument("--gait-ds", type=float, default=0.15, help="Double-support fraction (0-0.4).")
    p.add_argument("--gait-swing-scale", type=float, default=0.9, help="Scale swing amplitude (0-1).")
    p.add_argument("--gait-stance-knee", type=float, default=0.05, help="Extra knee flex in stance (rad).")
    p.add_argument("--gait-stance-ankle", type=float, default=0.03, help="Extra ankle push in stance (rad).")
    p.add_argument("--gait-swing-softness", type=float, default=1.5, help="Swing shaping softness.")
    p.add_argument("--gait-pitch-bias", type=float, default=0.15, help="Forward pitch target during gait (rad).")
    p.add_argument("--gait-stab-scale", type=float, default=1.0, help="Scale stabilization gains during gait.")
    p.add_argument("--gait-lat-amp", type=float, default=0.08, help="Hip roll swing amplitude during gait (rad).")
    p.add_argument("--gait-lat-offset", type=float, default=0.05, help="Baseline hip roll offset to widen stance (rad).")
    p.add_argument("--gait-lat-comp", type=float, default=0.3, help="Lateral velocity compensation during gait.")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--print-keys", action="store_true")
    p.add_argument("--keyframe-id", type=int, default=None)
    p.add_argument("--keyframe-name", type=str, default=None)
    p.add_argument("--no-keyframe", dest="no_keyframe", action="store_true", default=True)
    p.add_argument("--keyframe", dest="no_keyframe", action="store_false", help="Use model keyframe on startup.")
    p.add_argument("--scene", type=str, default=None, help="Override scene XML path.")
    p.add_argument("--use-stand-override", dest="use_stand_override", action="store_true", default=True)
    p.add_argument("--no-stand-override", dest="use_stand_override", action="store_false")
    p.add_argument("--stabilize", dest="stabilize", action="store_true", default=True)
    p.add_argument("--no-stabilize", dest="stabilize", action="store_false")
    p.add_argument("--stab-kp", type=float, default=0.6)
    p.add_argument("--stab-kd", type=float, default=0.2)
    p.add_argument("--xy-kp", type=float, default=0.4)
    p.add_argument("--xy-kd", type=float, default=0.1)
    p.add_argument("--xy-max", type=float, default=0.3)
    p.add_argument("--hip-weight", type=float, default=0.6)
    p.add_argument("--ankle-weight", type=float, default=0.4)
    p.add_argument("--stab-ramp", type=float, default=2.0)
    p.add_argument("--fall-reset", dest="fall_reset", action="store_true", default=True)
    p.add_argument("--no-fall-reset", dest="fall_reset", action="store_false")
    p.add_argument("--fall-reset-z", type=float, default=0.35)
    p.add_argument("--fall-reset-pin", type=float, default=1.0)
    p.add_argument("--finger-wiggle", action="store_true", default=False, help="Visual-only finger curl animation.")
    p.add_argument("--finger-freq", type=float, default=0.8, help="Finger curl frequency (Hz).")
    p.add_argument("--finger-amp", type=float, default=0.8, help="Finger curl amplitude (rad).")
    p.add_argument("--finger-offset", type=float, default=0.2, help="Finger curl offset (rad).")
    return p.parse_args()


def _jnt_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _set_joint_qpos(model, data, joint_name: str, value: float):
    jid = _jnt_id(model, joint_name)
    if jid < 0:
        return
    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr] = value


def _apply_leg_pose(model, data, hip: float, knee: float, ankle: float):
    _set_joint_qpos(model, data, "left_hip_pitch_joint", hip)
    _set_joint_qpos(model, data, "left_knee_joint", knee)
    _set_joint_qpos(model, data, "left_ankle_pitch_joint", ankle)
    _set_joint_qpos(model, data, "right_hip_pitch_joint", hip)
    _set_joint_qpos(model, data, "right_knee_joint", knee)
    _set_joint_qpos(model, data, "right_ankle_pitch_joint", ankle)


def _pose_score(model, data, pelvis_id, left_foot_id, right_foot_id):
    if min(pelvis_id, left_foot_id, right_foot_id) < 0:
        return 1e9
    lf = data.xpos[left_foot_id][2]
    rf = data.xpos[right_foot_id][2]
    pelvis_z = data.xpos[pelvis_id][2]
    avg_foot = 0.5 * (lf + rf)
    return abs(avg_foot - 0.0) + max(0.0, (avg_foot + 0.1) - pelvis_z)


def _build_act_joint(model: mujoco.MjModel):
    mapping = []
    for act_id in range(model.nu):
        jnt_id = int(model.actuator_trnid[act_id, 0])
        if jnt_id >= 0:
            mapping.append((act_id, jnt_id, model.jnt_qposadr[jnt_id], model.jnt_dofadr[jnt_id]))
        else:
            mapping.append((act_id, None, None, None))
    return mapping


def _joint_qpos_info(model: mujoco.MjModel, joint_name: str):
    jid = _jnt_id(model, joint_name)
    if jid < 0:
        return None
    qadr = model.jnt_qposadr[jid]
    limited = int(model.jnt_limited[jid]) == 1
    if limited:
        lo, hi = model.jnt_range[jid]
    else:
        lo, hi = -np.inf, np.inf
    return qadr, lo, hi


def _joint_dof_adr(model: mujoco.MjModel, joint_name: str):
    jid = _jnt_id(model, joint_name)
    if jid < 0:
        return None
    return model.jnt_dofadr[jid]


def _clamp(val: float, lo: float, hi: float) -> float:
    return min(max(val, lo), hi)


def _pelvis_rpy(model, data, pelvis_id: int):
    if pelvis_id < 0:
        return 0.0, 0.0, 0.0
    mat = data.xmat[pelvis_id].reshape(3, 3)
    # ZYX (yaw, pitch, roll)
    pitch = np.arcsin(-mat[2, 0])
    roll = np.arctan2(mat[2, 1], mat[2, 2])
    yaw = np.arctan2(mat[1, 0], mat[0, 0])
    return roll, pitch, yaw


def _rotate_world_to_body(yaw: float, vec_xy):
    c = np.cos(yaw)
    s = np.sin(yaw)
    x, y = float(vec_xy[0]), float(vec_xy[1])
    fwd = c * x + s * y
    lat = -s * x + c * y
    return fwd, lat


def _body_linvel_world(data, body_id: int):
    # cvel: [ang(3), lin(3)] in body frame
    v_body = data.cvel[body_id][3:6]
    mat = data.xmat[body_id].reshape(3, 3)
    return mat @ v_body


def main() -> int:
    args = _parse_args()
    if args.stable:
        args.kp = 110.0
        args.kd = 12.0
        args.stab_kp = 0.3
        args.stab_kd = 0.6
        args.xy_kp = 0.2
        args.xy_kd = 0.3
        args.xy_max = 0.15
        args.stab_ramp = 3.0
        args.fall_reset = False
        args.pin_seconds = max(args.pin_seconds, 4.0)
    elif args.gait_walk:
        # Softer gait defaults with stronger stabilization to reduce falls.
        args.kp = max(args.kp, 120.0)
        args.kd = max(args.kd, 10.0)
        args.stab_kp = max(args.stab_kp, 0.7)
        args.stab_kd = max(args.stab_kd, 0.4)
        args.xy_kp = max(args.xy_kp, 0.5)
        args.xy_kd = max(args.xy_kd, 0.2)
        args.xy_max = max(args.xy_max, 0.25)
        args.stab_ramp = max(args.stab_ramp, 2.5)
        args.step_freq = min(args.step_freq, 1.0)
        args.hip_amp = min(args.hip_amp, 0.25)
        args.knee_amp = min(args.knee_amp, 0.35)
        args.ankle_amp = min(args.ankle_amp, 0.2)
        args.hip_offset = min(args.hip_offset, 0.15)
        args.swing_knee = min(args.swing_knee, 0.25)
        args.swing_ankle = min(args.swing_ankle, 0.1)
        args.gait_pitch_bias = min(args.gait_pitch_bias, 0.15)
        args.gait_stab_scale = max(args.gait_stab_scale, 1.0)
        args.gait_lat_amp = min(args.gait_lat_amp, 0.08)
        args.gait_lat_offset = min(args.gait_lat_offset, 0.05)
        args.gait_lat_comp = min(args.gait_lat_comp, 0.3)
        args.gait_ds = _clamp(args.gait_ds, 0.1, 0.25)
        args.gait_swing_scale = _clamp(args.gait_swing_scale, 0.7, 1.0)
        args.gait_stance_knee = _clamp(args.gait_stance_knee, 0.03, 0.08)
        args.gait_stance_ankle = _clamp(args.gait_stance_ankle, 0.02, 0.06)
    scene_path = Path(args.scene) if args.scene else Path(config.ROBOT_SCENE)
    if not scene_path.is_absolute():
        scene_path = (Path(__file__).resolve().parent / scene_path).resolve()

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    if args.print_keys:
        print("nkey =", model.nkey)
        for k in range(model.nkey):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k)
            print("key", k, "name:", name)

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    # Initialize pose (optional keyframe, otherwise defaults)
    use_keyframe = (not args.no_keyframe) and (
        args.keyframe_id is not None or args.keyframe_name is not None or model.nkey > 0
    )
    if use_keyframe:
        if args.keyframe_id is not None:
            key_id = args.keyframe_id
        elif args.keyframe_name is not None:
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe_name)
        else:
            stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
            # If no explicit keyframe and no "stand" key, force manual stand pose when lock is on.
            if stand_id < 0 and args.lock_stand:
                use_keyframe = False
            else:
                key_id = stand_id if stand_id >= 0 else 0
    if use_keyframe:
        if key_id < 0 or key_id >= model.nkey:
            raise RuntimeError(f"Invalid keyframe (id={key_id}, name={args.keyframe_name}).")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        hip, knee, ankle = args.hip, args.knee, args.ankle
        if args.use_stand_override:
            data.qpos[2] = args.stand_z
            data.qpos[3] = 1.0
            _apply_leg_pose(model, data, hip, knee, ankle)
            mujoco.mj_forward(model, data)
    else:
        data.qpos[:] = 0.0
        data.qpos[2] = args.stand_z
        data.qpos[3] = 1.0

        hip, knee, ankle = args.hip, args.knee, args.ankle
        _apply_leg_pose(model, data, hip, knee, ankle)
        mujoco.mj_forward(model, data)

    if args.auto_sign:
        score_a = _pose_score(model, data, pelvis_id, left_foot_id, right_foot_id)
        _apply_leg_pose(model, data, -hip, -knee, -ankle)
        mujoco.mj_forward(model, data)
        score_b = _pose_score(model, data, pelvis_id, left_foot_id, right_foot_id)
        if score_b < score_a:
            hip, knee, ankle = -hip, -knee, -ankle
        _apply_leg_pose(model, data, hip, knee, ankle)
        mujoco.mj_forward(model, data)

    if args.auto_stand_z and left_foot_id >= 0 and right_foot_id >= 0:
        lf = data.xpos[left_foot_id][2]
        rf = data.xpos[right_foot_id][2]
        avg_foot = 0.5 * (lf + rf)
        clearance = 0.02
        dz = -(avg_foot - clearance)
        data.qpos[2] += dz
        mujoco.mj_forward(model, data)

    # Pin all joints initially
    pin_until = time.time() + args.pin_seconds
    pin_qpos = data.qpos.copy()
    stand_qpos = data.qpos.copy()

    act_joint = _build_act_joint(model)
    qpos_des = data.qpos.copy()
    # Precompute qpos slots for stabilization joints (if present)
    joints = {
        "left_hip_pitch_joint": _joint_qpos_info(model, "left_hip_pitch_joint"),
        "right_hip_pitch_joint": _joint_qpos_info(model, "right_hip_pitch_joint"),
        "left_knee_joint": _joint_qpos_info(model, "left_knee_joint"),
        "right_knee_joint": _joint_qpos_info(model, "right_knee_joint"),
        "left_ankle_pitch_joint": _joint_qpos_info(model, "left_ankle_pitch_joint"),
        "right_ankle_pitch_joint": _joint_qpos_info(model, "right_ankle_pitch_joint"),
        "left_hip_roll_joint": _joint_qpos_info(model, "left_hip_roll_joint"),
        "right_hip_roll_joint": _joint_qpos_info(model, "right_hip_roll_joint"),
        "left_ankle_roll_joint": _joint_qpos_info(model, "left_ankle_roll_joint"),
        "right_ankle_roll_joint": _joint_qpos_info(model, "right_ankle_roll_joint"),
    }

    finger_joint_names = [
        "left_thumb_vis_joint",
        "left_index_vis_joint",
        "left_middle_vis_joint",
        "left_ring_vis_joint",
        "left_pinky_vis_joint",
        "right_thumb_vis_joint",
        "right_index_vis_joint",
        "right_middle_vis_joint",
        "right_ring_vis_joint",
        "right_pinky_vis_joint",
    ]
    finger_joints = {}
    for name in finger_joint_names:
        info = _joint_qpos_info(model, name)
        if info is None:
            continue
        dadr = _joint_dof_adr(model, name)
        if dadr is None:
            continue
        qadr, lo, hi = info
        finger_joints[name] = (qadr, lo, hi, dadr)

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)

    last_print = time.time()
    while True:
        if viewer is not None and (not viewer.is_running()):
            break

        if args.lock_stand and not args.gait_walk:
            data.qpos[:] = stand_qpos
            if args.forward_speed != 0.0:
                # Drift the base forward in +X (demo-friendly, non-physical).
                data.qpos[0] += args.forward_speed * model.opt.timestep
            data.qvel[:] = 0.0
        elif time.time() < pin_until:
            data.qpos[:] = pin_qpos
            data.qvel[:] = 0.0
        else:
            # PD hold on joints
            data.ctrl[:] = 0.0
            qpos_cmd = qpos_des.copy()

            if args.gait_walk:
                # Simple open-loop gait: sinusoidal hip/knee/ankle pitch.
                t = time.time() - pin_until
                phase = 2.0 * np.pi * args.step_freq * t
                s = np.sin(phase)
                # Soften extremes to reduce foot slap and pitching.
                s = np.tanh(args.gait_swing_softness * s)
                if pelvis_id >= 0:
                    _, _, yaw = _pelvis_rpy(model, data, pelvis_id)
                else:
                    yaw = 0.0
                # Left/right are out of phase.
                left_s = s
                right_s = -s
                # Swing/stance gating with double-support window.
                ds = _clamp(args.gait_ds, 0.0, 0.4)
                def swing_gate(val):
                    if val <= ds:
                        return 0.0
                    return _clamp((val - ds) / (1.0 - ds), 0.0, 1.0)

                left_swing = swing_gate(left_s)
                right_swing = swing_gate(right_s)
                left_stance = 1.0 - left_swing
                right_stance = 1.0 - right_swing

                def set_joint(jname, base, amp, val):
                    info = joints.get(jname)
                    if info is None:
                        return
                    qadr, lo, hi = info
                    qpos_cmd[qadr] = _clamp(base + amp * val, lo, hi)

                def base_qpos(jname):
                    info = joints.get(jname)
                    if info is None:
                        return None
                    return qpos_des[info[0]]

                # Hips lean forward with offset.
                base_l_hip = base_qpos("left_hip_pitch_joint")
                base_r_hip = base_qpos("right_hip_pitch_joint")
                base_l_knee = base_qpos("left_knee_joint")
                base_r_knee = base_qpos("right_knee_joint")
                base_l_ankle = base_qpos("left_ankle_pitch_joint")
                base_r_ankle = base_qpos("right_ankle_pitch_joint")

                swing_scale = _clamp(args.gait_swing_scale, 0.0, 1.0)
                if base_l_hip is not None:
                    set_joint("left_hip_pitch_joint", base_l_hip, args.hip_amp * swing_scale, left_s)
                if base_r_hip is not None:
                    set_joint("right_hip_pitch_joint", base_r_hip, args.hip_amp * swing_scale, right_s)
                # Add forward lean offset.
                if base_l_hip is not None:
                    set_joint("left_hip_pitch_joint", qpos_cmd[joints["left_hip_pitch_joint"][0]] + args.hip_offset, 0.0, 0.0)
                if base_r_hip is not None:
                    set_joint("right_hip_pitch_joint", qpos_cmd[joints["right_hip_pitch_joint"][0]] + args.hip_offset, 0.0, 0.0)

                # Knees: flex more during swing, extend slightly during stance.
                if base_l_knee is not None:
                    knee_l = base_l_knee + args.knee_amp * (-left_s) + args.swing_knee * left_swing + args.gait_stance_knee * left_stance
                    set_joint("left_knee_joint", knee_l, 0.0, 0.0)
                if base_r_knee is not None:
                    knee_r = base_r_knee + args.knee_amp * (-right_s) + args.swing_knee * right_swing + args.gait_stance_knee * right_stance
                    set_joint("right_knee_joint", knee_r, 0.0, 0.0)
                # Ankles: dorsiflex during swing for toe clearance.
                if base_l_ankle is not None:
                    ankle_l = base_l_ankle + args.ankle_amp * left_s - args.swing_ankle * left_swing - args.gait_stance_ankle * left_stance
                    set_joint("left_ankle_pitch_joint", ankle_l, 0.0, 0.0)
                if base_r_ankle is not None:
                    ankle_r = base_r_ankle + args.ankle_amp * right_s - args.swing_ankle * right_swing - args.gait_stance_ankle * right_stance
                    set_joint("right_ankle_pitch_joint", ankle_r, 0.0, 0.0)

                # Lateral gait: widen stance and counter lateral velocity.
                if pelvis_id >= 0:
                    v_world = _body_linvel_world(data, pelvis_id)
                    _, v_lat = _rotate_world_to_body(yaw, v_world[:2])
                    lat_comp = _clamp(-args.gait_lat_comp * v_lat, -0.12, 0.12)
                else:
                    lat_comp = 0.0
                base_l_hip_roll = base_qpos("left_hip_roll_joint")
                base_r_hip_roll = base_qpos("right_hip_roll_joint")
                if base_l_hip_roll is not None:
                    set_joint(
                        "left_hip_roll_joint",
                        base_l_hip_roll + args.gait_lat_offset + lat_comp,
                        args.gait_lat_amp,
                        left_s,
                    )
                if base_r_hip_roll is not None:
                    set_joint(
                        "right_hip_roll_joint",
                        base_r_hip_roll - args.gait_lat_offset - lat_comp,
                        args.gait_lat_amp,
                        right_s,
                    )
            if args.stabilize and pelvis_id >= 0:
                if args.stab_ramp > 0.0:
                    ramp = (time.time() - pin_until) / args.stab_ramp
                    ramp = _clamp(ramp, 0.0, 1.0)
                else:
                    ramp = 1.0
                roll, pitch, yaw = _pelvis_rpy(model, data, pelvis_id)
                # Use body-frame angular velocity from cvel (first 3 components)
                ang_vel = data.cvel[pelvis_id][:3]
                roll_rate = float(ang_vel[0])
                pitch_rate = float(ang_vel[1])
                stab_scale = args.gait_stab_scale if args.gait_walk else 1.0
                roll_corr = (-args.stab_kp * roll - args.stab_kd * roll_rate) * ramp * stab_scale
                if args.gait_walk:
                    # Track a forward pitch target during gait.
                    pitch_err = pitch - (-args.gait_pitch_bias)
                    pitch_corr = (-args.stab_kp * pitch_err - args.stab_kd * pitch_rate) * ramp * stab_scale
                else:
                    pitch_corr = (-args.stab_kp * pitch - args.stab_kd * pitch_rate) * ramp * stab_scale

                # Position stabilization: keep pelvis over mid-foot in XY
                if left_foot_id >= 0 and right_foot_id >= 0:
                    midfoot = 0.5 * (data.xpos[left_foot_id] + data.xpos[right_foot_id])
                    err_xy = data.xpos[pelvis_id][:2] - midfoot[:2]
                    v_world = _body_linvel_world(data, pelvis_id)
                    err_fwd, err_lat = _rotate_world_to_body(yaw, err_xy)
                    v_fwd, v_lat = _rotate_world_to_body(yaw, v_world[:2])
                    pitch_xy = (-args.xy_kp * err_fwd - args.xy_kd * v_fwd) * ramp
                    roll_xy = (-args.xy_kp * err_lat - args.xy_kd * v_lat) * ramp
                    pitch_corr += _clamp(pitch_xy, -args.xy_max, args.xy_max)
                    roll_corr += _clamp(roll_xy, -args.xy_max, args.xy_max)

                hip_w = args.hip_weight
                ankle_w = args.ankle_weight

                def apply_corr(jname, corr):
                    info = joints.get(jname)
                    if info is None:
                        return
                    qadr, lo, hi = info
                    qpos_cmd[qadr] = _clamp(qpos_cmd[qadr] + corr, lo, hi)

                # Pitch stabilization: push hip/ankle pitch
                apply_corr("left_hip_pitch_joint", hip_w * pitch_corr)
                apply_corr("right_hip_pitch_joint", hip_w * pitch_corr)
                apply_corr("left_ankle_pitch_joint", -ankle_w * pitch_corr)
                apply_corr("right_ankle_pitch_joint", -ankle_w * pitch_corr)

                # Roll stabilization: push hip/ankle roll
                apply_corr("left_hip_roll_joint", hip_w * roll_corr)
                apply_corr("right_hip_roll_joint", hip_w * roll_corr)
                apply_corr("left_ankle_roll_joint", -ankle_w * roll_corr)
                apply_corr("right_ankle_roll_joint", -ankle_w * roll_corr)

            for act_id, jnt_id, qadr, dadr in act_joint:
                if jnt_id is None:
                    continue
                q = data.qpos[qadr]
                qd = data.qvel[dadr]
                q_des = qpos_cmd[qadr]
                tau = args.kp * (q_des - q) - args.kd * qd
                lo, hi = model.actuator_ctrlrange[act_id]
                data.ctrl[act_id] = np.clip(tau, lo, hi)

            if args.gait_walk and args.forward_speed != 0.0:
                # Drift the base forward in +X (demo-friendly, non-physical).
                v = _clamp(args.forward_speed, -0.3, 0.3)
                data.qpos[0] += v * model.opt.timestep

        if args.fall_reset and pelvis_id >= 0 and time.time() >= pin_until:
            if data.xpos[pelvis_id][2] < args.fall_reset_z:
                stand_qpos = stand_qpos.copy()
                data.qpos[:] = stand_qpos
                data.qvel[:] = 0.0
                qpos_des = stand_qpos.copy()
                pin_qpos = stand_qpos.copy()
                pin_until = time.time() + args.fall_reset_pin

        if args.finger_wiggle and finger_joints:
            phase = 2.0 * np.pi * args.finger_freq * time.time()
            base = args.finger_offset + args.finger_amp * 0.5 * (np.sin(phase) + 1.0)
            for name, (qadr, lo, hi, dadr) in finger_joints.items():
                scale = 1.0
                if "thumb" in name:
                    scale = 0.7
                elif "pinky" in name:
                    scale = 0.8
                elif "ring" in name:
                    scale = 0.9
                q = _clamp(base * scale, lo, hi)
                data.qpos[qadr] = q
                data.qvel[dadr] = 0.0

        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()

        if time.time() - last_print > 1.0 and pelvis_id >= 0:
            pos = data.xpos[pelvis_id]
            print(f"[stand_up_sim] pelvis pos: {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
            last_print = time.time()

        time.sleep(model.opt.timestep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

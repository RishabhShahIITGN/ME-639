#!/usr/bin/env python3
"""
ik_pick_place.py — Task 9: Pick-and-place with Mink IK
========================================================
Implements a 5-phase state machine for pick-and-place using Mink
(differential IK) with collision avoidance between gripper, cube,
table, and tray.

Phases:
  1. APPROACH  — Move EE above the cube (pre-grasp pose, gripper open)
  2. DESCEND   — Lower EE to the grasp height
  3. GRASP     — Close gripper around the cube
  4. LIFT      — Lift the cube vertically
  5. TRANSIT   — Move to above the tray (place target)
  6. LOWER     — Lower to the tray surface
  7. RELEASE   — Open gripper and retreat upward

Usage:
    cd ME-639/lab/lab-03
    source ../../venv/bin/activate
    python3 scripts/ik_pick_place.py                     # single episode, viewer
    python3 scripts/ik_pick_place.py --episodes 5 --no-gui
    python3 scripts/ik_pick_place.py --seed 42 --no-gui  # deterministic
"""

from __future__ import annotations

import argparse
import json
import time
from enum import Enum, auto
from pathlib import Path

import mujoco
import numpy as np

import mink

# ---------------------------------------------------------------------------
# Import the environment builder from Task 7
# ---------------------------------------------------------------------------
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env_pick_place import build_model, ARM_ACTUATORS  # noqa: E402

LAB = HERE.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DT = 0.002               # MuJoCo timestep
IK_DT = 0.1              # IK integration timestep (virtual)
MAX_IK_ITERS = 600       # max Mink iterations per phase
POS_TOL = 0.005           # 5 mm position convergence
ORI_TOL = 0.05            # orientation convergence (rad-ish)
GRIPPER_OPEN_CMD = 0.0
GRIPPER_CLOSE_CMD = 255.0

GRIPPER_OFFSET = 0.14     # Gripper pads are ~14cm below EE site
PRE_GRASP_HEIGHT = 0.12   # pad height above table surface for pre-grasp
GRASP_HEIGHT = 0.025      # pad height above table surface at grasp (cube center)
LIFT_HEIGHT = 0.15        # pad height above table for lifted cube
SETTLE_STEPS = 1000       # sim steps to let physics settle after grip change
CUBE_HALF = 0.025

# Table design
with open(LAB / "table_design.json") as _f:
    _design = json.load(_f)
TABLE_H = _design["table_height_m"]


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
class Phase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    TRANSIT = auto()
    LOWER = auto()
    RELEASE = auto()
    DONE = auto()


# ---------------------------------------------------------------------------
# Orientation: EE pointing straight down
# ---------------------------------------------------------------------------
def rot_down_yaw(yaw: float = 0.0) -> mink.SO3:
    """Rotation matrix: EE z-axis pointing down, with a yaw twist."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [ c,  s, 0],
        [ s, -c, 0],
        [ 0,  0, -1],
    ])
    return mink.SO3.from_matrix(R)

ROT_DOWN = rot_down_yaw(0.0)


# ---------------------------------------------------------------------------
# IK and Simulation Integration
# ---------------------------------------------------------------------------
def pd_control_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_q: np.ndarray,
    target_v: np.ndarray,
    gripper_cmd: float,
    kp: float = 500.0,
    kd: float = 50.0,
    steps: int = 1,
    viewer=None,
):
    """Apply PD control to track target joint positions and step the physics."""
    grip_act_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator"
    )
    for _ in range(steps):
        for name in ARM_ACTUATORS:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                continue
            dof = model.actuator_trnid[aid][0]
            q = data.qpos[dof]
            v = data.qvel[dof]
            tau = kp * (target_q[dof] - q) + kd * (target_v[dof] - v) + data.qfrc_bias[dof]
            data.ctrl[aid] = tau
            
        if grip_act_id >= 0:
            data.ctrl[grip_act_id] = gripper_cmd
            
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def move_to_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: mink.Configuration,
    task: mink.FrameTask,
    target: mink.SE3,
    gripper_cmd: float,
    limits: list | None = None,
    max_iters: int = MAX_IK_ITERS,
    pos_tol: float = POS_TOL,
    viewer=None,
) -> tuple[bool, int, float]:
    """Drive the configuration toward target smoothly using PD control."""
    start_pose = config.get_transform_frame_to_world(task.frame_name, task.frame_type)
    
    # We want to move at roughly 0.03 m/s.
    dist = np.linalg.norm(target.translation() - start_pose.translation())
    duration = max(dist / 0.03, 1.0)  # at least 1.0s
    ik_dt = 0.01  # IK solver dt
    num_steps = int(duration / ik_dt)
    
    # Physics steps per IK step
    physics_steps = int(ik_dt / DT)
    
    for i in range(num_steps):
        alpha = min(1.0, (i + 1) / num_steps)
        interp_target = start_pose.interpolate(target, alpha)
        task.set_target(interp_target)

        vel = mink.solve_ik(
            config, [task], ik_dt,
            solver="daqp",
            damping=1e-4,
            limits=limits if limits else [],
        )
        config.integrate_inplace(vel, ik_dt)
        
        pd_control_step(
            model, data, config.q, vel, gripper_cmd,
            steps=physics_steps, viewer=viewer
        )
    
    # Final convergence check
    T_cur = config.get_transform_frame_to_world(task.frame_name, task.frame_type)
    err = np.linalg.norm(T_cur.translation() - target.translation())
    
    # Give it a few more steps to settle exactly if needed
    if err > pos_tol:
        task.set_target(target)
        for i in range(50):
            vel = mink.solve_ik(
                config, [task], ik_dt, solver="daqp", damping=1e-4, limits=limits if limits else []
            )
            config.integrate_inplace(vel, ik_dt)
            
            pd_control_step(
                model, data, config.q, vel, gripper_cmd,
                steps=physics_steps, viewer=viewer
            )
            
            T_cur = config.get_transform_frame_to_world(task.frame_name, task.frame_type)
            err = np.linalg.norm(T_cur.translation() - target.translation())
            if err < pos_tol:
                break
                
    return (err < pos_tol), num_steps, float(err)


# ---------------------------------------------------------------------------
# Collision checking
# ---------------------------------------------------------------------------
def check_collisions(model, data, cube_body_id):
    """Check for undesirable collisions. Returns a list of issues."""
    issues = []
    mujoco.mj_forward(model, data)
    for k in range(data.ncon):
        c = data.contact[k]
        g1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        b1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
        b2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
        pair = frozenset([b1_name, b2_name])

        # Arm links hitting the table (not gripper pads, those are OK near cube)
        arm_bodies = {"link_1", "link_2", "link_3", "link_4", "link_5"}
        if pair & arm_bodies and "table" in pair:
            issues.append(f"arm-table collision: {b1_name} <-> {b2_name}")

        # Gripper base (not pads) hitting table
        if "gripper/base" in pair and "table" in pair:
            issues.append(f"gripper_base-table collision")

    return issues


def check_cube_in_tray(data, cube_body_id, tray_body_id):
    """Check if the cube center is within the tray bounds."""
    cube_pos = data.xpos[cube_body_id]
    tray_pos = data.xpos[tray_body_id]
    dx = abs(cube_pos[0] - tray_pos[0])
    dy = abs(cube_pos[1] - tray_pos[1])
    # Tray half-sizes: 0.06 m each
    return dx < 0.06 and dy < 0.06 and cube_pos[2] > TABLE_H


def check_cube_grasped(data, cube_body_id):
    """Check if cube is lifted above the table surface."""
    return data.xpos[cube_body_id][2] > TABLE_H + CUBE_HALF + 0.02


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------
def run_episode(
    seed: int | None = None,
    viewer=None,
    verbose: bool = True,
) -> dict:
    """Run one pick-and-place episode.

    Returns:
        dict with keys: success, reason, phases, cube_start, cube_end, ...
    """
    model, data = build_model(seed=seed)
    mujoco.mj_forward(model, data)

    # IDs
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    tray_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")

    cube_start = data.xpos[cube_body_id].copy()
    if verbose:
        print(f"  Cube start: ({cube_start[0]:.3f}, {cube_start[1]:.3f}, {cube_start[2]:.3f})")

    # Mink configuration
    config = mink.Configuration(model)
    config.update()

    # Let the cube settle on the table first
    pd_control_step(model, data, config.q, np.zeros(6), GRIPPER_OPEN_CMD, steps=200, viewer=viewer)
    mujoco.mj_forward(model, data)
    cube_pos_settled = data.xpos[cube_body_id].copy()

    # Tray / place target
    tray_pos = data.xpos[tray_body_id].copy()
    place_target_xy = np.array([tray_pos[0], tray_pos[1]])

    # IK task: track the EE site
    ee_task = mink.FrameTask(
        frame_name="right_center",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        gain=1.0,
        lm_damping=1e-3,
    )

    # Collision avoidance limits
    gripper_geoms = ["gripper/left_pad1", "gripper/left_pad2",
                     "gripper/right_pad1", "gripper/right_pad2"]
    table_geoms = ["table_top"]
    tray_geoms = ["tray_floor", "tray_wall_px", "tray_wall_mx",
                   "tray_wall_py", "tray_wall_my"]
    arm_collision_geoms = ["link_2_collision", "link_3_collision",
                           "link_4_collision", "link_5_collision",
                           "end_effector_collision"]

    try:
        collision_limit = mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[
                (arm_collision_geoms + gripper_geoms, table_geoms),
                (arm_collision_geoms + gripper_geoms, tray_geoms),
            ],
            gain=0.85,
            minimum_distance_from_collisions=0.01,
            collision_detection_distance=0.05,
        )
        ik_limits = [collision_limit]
    except Exception as e:
        if verbose:
            print(f"  [WARN] Collision avoidance setup failed: {e}, proceeding without")
        ik_limits = []

    # Phase log
    phase_log = {}
    success = False
    reason = "incomplete"
    gripper_cmd = GRIPPER_OPEN_CMD

    # ----- Phase targets -----
    cx, cy, cz = cube_pos_settled
    
    # Target Z for the right_center site to achieve desired pad height
    def target_z(pad_height):
        return TABLE_H + pad_height + GRIPPER_OFFSET
        
    approach_pos = np.array([cx, cy, target_z(PRE_GRASP_HEIGHT)])
    descend_pos = np.array([cx, cy, target_z(GRASP_HEIGHT)])
    lift_pos = np.array([cx, cy, target_z(LIFT_HEIGHT)])
    transit_pos = np.array([place_target_xy[0], place_target_xy[1], target_z(LIFT_HEIGHT)])
    lower_pos = np.array([place_target_xy[0], place_target_xy[1], target_z(GRASP_HEIGHT)])
    retreat_pos = np.array([place_target_xy[0], place_target_xy[1], target_z(PRE_GRASP_HEIGHT)])

    phases_sequence = [
        (Phase.APPROACH, approach_pos),
        (Phase.DESCEND,  descend_pos),
        (Phase.GRASP,    None),         # no motion, just close gripper
        (Phase.LIFT,     lift_pos),
        (Phase.TRANSIT,  transit_pos),
        (Phase.LOWER,    lower_pos),
        (Phase.RELEASE,  None),         # open gripper + retreat
    ]

    for phase, target_pos in phases_sequence:
        if verbose:
            print(f"  Phase: {phase.name}", end="")

        t0 = time.time()

        if phase == Phase.GRASP:
            # Close gripper
            gripper_cmd = GRIPPER_CLOSE_CMD
            pd_control_step(model, data, config.q, np.zeros(6), gripper_cmd, steps=SETTLE_STEPS, viewer=viewer)
            mujoco.mj_forward(model, data)

            # Verify grasp
            if not check_cube_grasped(data, cube_body_id):
                cube_now = data.xpos[cube_body_id]
                if abs(cube_now[0] - cx) > 0.05 or abs(cube_now[1] - cy) > 0.05:
                    reason = "grasp_failed: cube slipped during close"
                    phase_log[phase.name] = {"ok": False, "reason": reason}
                    if verbose:
                        print(f" — FAILED ({reason})")
                    break

            phase_log[phase.name] = {"ok": True, "time": time.time() - t0}
            if verbose:
                print(f" — OK ({time.time()-t0:.2f}s)")
            continue

        elif phase == Phase.RELEASE:
            gripper_cmd = GRIPPER_OPEN_CMD
            pd_control_step(model, data, config.q, np.zeros(6), gripper_cmd, steps=SETTLE_STEPS, viewer=viewer)

            # Retreat up
            target = mink.SE3.from_rotation_and_translation(ROT_DOWN, retreat_pos)
            converged, iters, err = move_to_target(
                model, data, config, ee_task, target, gripper_cmd, limits=ik_limits, viewer=viewer
            )

            mujoco.mj_forward(model, data)
            in_tray = check_cube_in_tray(data, cube_body_id, tray_body_id)
            phase_log[phase.name] = {
                "ok": in_tray, "converged": converged, "iters": iters, "err": err,
                "time": time.time() - t0,
            }
            if in_tray:
                success = True
                reason = "success"
            else:
                cube_final = data.xpos[cube_body_id]
                reason = (f"place_failed: cube at ({cube_final[0]:.3f}, "
                          f"{cube_final[1]:.3f}, {cube_final[2]:.3f}), "
                          f"not in tray")
            if verbose:
                print(f" — {'OK' if in_tray else 'FAILED'} ({reason})")
            break

        else:
            # Motion phases: APPROACH, DESCEND, LIFT, TRANSIT, LOWER
            target = mink.SE3.from_rotation_and_translation(ROT_DOWN, target_pos)
            converged, iters, err = move_to_target(
                model, data, config, ee_task, target, gripper_cmd, limits=ik_limits, viewer=viewer
            )

            # Collision check after motion
            mujoco.mj_forward(model, data)
            col_issues = check_collisions(model, data, cube_body_id)

            phase_log[phase.name] = {
                "ok": converged and not col_issues,
                "converged": converged,
                "iters": iters,
                "pos_err": float(err),
                "collisions": col_issues,
                "time": time.time() - t0,
            }

            if verbose:
                status = "OK" if converged else f"FAIL(err={err:.4f})"
                col_str = f", collisions={col_issues}" if col_issues else ""
                print(f" — {status} ({iters} iters, {err:.4f}m{col_str})")

            if not converged:
                reason = f"ik_diverged in {phase.name} (err={err:.4f}m)"
                break

            # After LIFT, verify cube is still held
            if phase == Phase.LIFT:
                mujoco.mj_forward(model, data)
                if not check_cube_grasped(data, cube_body_id):
                    reason = "cube_dropped during lift"
                    phase_log[phase.name]["ok"] = False
                    if verbose:
                        print(f"    ⚠ Cube dropped! z={data.xpos[cube_body_id][2]:.3f}")
                    break

    cube_end = data.xpos[cube_body_id].copy()

    result = {
        "success": success,
        "reason": reason,
        "cube_start": cube_start.tolist(),
        "cube_end": cube_end.tolist(),
        "phases": phase_log,
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Task 9: IK pick-and-place with Mink")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    results = []
    n_success = 0

    log_dir = LAB / "logs"
    log_dir.mkdir(exist_ok=True)

    for ep in range(args.episodes):
        ep_seed = int(rng.integers(0, 2**31))
        print(f"\n{'='*50}")
        print(f"Episode {ep+1}/{args.episodes}  (seed={ep_seed})")
        print(f"{'='*50}")

        result = run_episode(seed=ep_seed, viewer=None, verbose=True)
        result["episode"] = ep
        result["seed"] = ep_seed
        results.append(result)

        if result["success"]:
            n_success += 1
            print(f"  ✓ SUCCESS")
        else:
            print(f"  ✗ FAILURE: {result['reason']}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Summary: {n_success}/{args.episodes} successful "
          f"({100*n_success/max(args.episodes,1):.0f}%)")
    print(f"{'='*50}")

    # Save log
    log_file = log_dir / "ik_pick_place_log.json"
    with open(log_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved log → {log_file}")


if __name__ == "__main__":
    main()

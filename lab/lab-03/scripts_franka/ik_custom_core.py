#!/usr/bin/env python3
"""Shared pick-and-place episode loop for the custom (non-Mink) IK solvers.

DLS and QP live in their own runnable files so each method can be batched
independently. This module only provides Jacobian/error helpers and the
7-phase episode machine that calls a solver callback every IK step.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ik_pick_place import (  # noqa: E402
    CARRY_DRIFT_TOL,
    CARRY_PHASES,
    CUBE_HALF,
    EE_SITE,
    Failure,
    GRASP_HEIGHT,
    GRIPPER_CLOSE_CMD,
    GRIPPER_OFFSET,
    GRIPPER_OPEN_CMD,
    IK_DT,
    LIFT_FRACTION,
    LIFT_HEIGHT,
    LOWER_CLEARANCE,
    MIN_GRASP_FORCE,
    MIN_MOVE_TIME,
    MOVE_SPEED,
    PHASE_ORDER,
    PLACE_SETTLE_STEPS,
    POS_TOL,
    PRE_GRASP_HEIGHT,
    Phase,
    Scene,
    SETTLE_STEPS,
    TABLE_H_DESIGN,
    TAIL_IK_ITERS,
    TRANSIT_CLEARANCE,
    check_collisions,
    check_cube_placed,
    cube_rel_to_ee,
    cube_yaw,
    geom_top_z,
    grasp_yaw_for,
    measure_min_clearance,
    pads_hold_cube,
    pd_control_step,
    rot_down_yaw,
    site_pos,
    _jsonable,
)
from env_pick_place import ARM_ACTUATORS, build_model  # noqa: E402

SolveIK = Callable[[mujoco.MjModel, mujoco.MjData, np.ndarray, np.ndarray, str], Optional[np.ndarray]]


def get_ee_jacobian(model, data, site_name=EE_SITE):
    """Spatial Jacobian for a site: 6 x nv, [linear; angular]."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack([jacp, jacr])


def compute_error(model, data, target_pos, target_quat, site_name=EE_SITE):
    """6D pose error [dx, dy, dz, drx, dry, drz]. target_quat is [w, x, y, z]."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    pos_err = target_pos - data.site_xpos[site_id]

    site_mat = data.site_xmat[site_id].reshape(3, 3)
    target_mat = np.zeros(9)
    mujoco.mju_quat2Mat(target_mat, target_quat)
    target_mat = target_mat.reshape(3, 3)
    R_err = target_mat @ site_mat.T

    quat_err = np.zeros(4)
    mujoco.mju_mat2Quat(quat_err, R_err.flatten())
    axis = np.zeros(3)
    mujoco.mju_quat2Vel(axis, quat_err, 1.0)
    return np.concatenate([pos_err, axis])


def arm_dof_indices(model: mujoco.MjModel) -> np.ndarray:
    """Velocity indices of the 6 HEAL arm actuators (exclude cube / fingers)."""
    dofs = []
    for name in ARM_ACTUATORS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            continue
        jnt = int(model.actuator_trnid[aid][0])
        dofs.append(int(model.jnt_dofadr[jnt]))
    return np.asarray(dofs, dtype=int)


def pack_arm_velocity(model: mujoco.MjModel, dq_arm: np.ndarray) -> np.ndarray:
    dq = np.zeros(model.nv)
    dq[arm_dof_indices(model)] = dq_arm
    return dq


def mask_arm_velocity(dq: np.ndarray, scene: Scene, nv: int) -> np.ndarray:
    """Keep only arm DoFs so the cube / fingers are not used as IK variables."""
    out = np.zeros(nv)
    for _, _, dadr in scene.arm_act:
        out[dadr] = dq[dadr]
    return out


def integrate_arm_qpos(q: np.ndarray, dq: np.ndarray, dt: float, model, scene: Scene) -> np.ndarray:
    q_new = q.copy()
    for _, qadr, dadr in scene.arm_act:
        q_new[qadr] = q[qadr] + dq[dadr] * dt
    for _, qadr, dadr in scene.arm_act:
        jnt = None
        for j in range(model.njnt):
            if int(model.jnt_qposadr[j]) == qadr:
                jnt = j
                break
        if jnt is not None and model.jnt_limited[jnt]:
            lo, hi = model.jnt_range[jnt]
            q_new[qadr] = float(np.clip(q_new[qadr], lo, hi))
    return q_new


def mat_to_quat(mat3: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat3.reshape(9))
    return quat


def slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / max(np.linalg.norm(q), 1e-12)
    theta = np.arccos(dot)
    return (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


class MoveResult:
    def __init__(self, converged, steps, err, failure=None, planning_time_s=0.0):
        self.converged = converged
        self.steps = steps
        self.err = err
        self.failure = failure
        self.planning_time_s = planning_time_s


def hold_still(model, data, scene, q_ref, gripper_cmd, steps, viewer=None):
    pd_control_step(model, data, scene, q_ref, np.zeros(model.nv),
                    gripper_cmd, steps=steps, viewer=viewer)


def move_to_target(
    model,
    data,
    scene: Scene,
    q_ref: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    gripper_cmd: float,
    phase: Phase,
    solve_ik: SolveIK,
    monitor=None,
    pos_tol: float = POS_TOL,
    viewer=None,
) -> tuple[MoveResult, np.ndarray]:
    """Straight-line interpolation in task space using a custom velocity IK solver."""
    ik_dt = IK_DT
    physics_steps = max(1, int(round(ik_dt / model.opt.timestep)))
    planning_time_s = 0.0
    q_ref = q_ref.copy()

    def real_err() -> float:
        return float(np.linalg.norm(site_pos(model, data, scene) - target_pos))

    start_pos = site_pos(model, data, scene)
    start_quat = mat_to_quat(data.site_xmat[scene.site])
    dist = float(np.linalg.norm(target_pos - start_pos))
    duration = max(dist / MOVE_SPEED, MIN_MOVE_TIME)
    num_steps = max(1, int(duration / ik_dt))

    def one_step(pos, quat):
        nonlocal planning_time_s, q_ref
        solve_started = time.perf_counter()
        dq = solve_ik(model, data, pos, quat, EE_SITE)
        planning_time_s += time.perf_counter() - solve_started
        if dq is None:
            raise RuntimeError("IK solver returned no solution")
        dq = mask_arm_velocity(np.asarray(dq, dtype=float), scene, model.nv)
        q_ref = integrate_arm_qpos(q_ref, dq, ik_dt, model, scene)
        pd_control_step(model, data, scene, q_ref, dq, gripper_cmd,
                        steps=physics_steps, viewer=viewer)

    try:
        q_ref = data.qpos.copy()
        for i in range(num_steps):
            alpha = min(1.0, (i + 1) / num_steps)
            pos = start_pos + alpha * (target_pos - start_pos)
            quat = slerp_quat(start_quat, target_quat, alpha)
            one_step(pos, quat)
            if monitor is not None:
                fail = monitor()
                if fail:
                    return MoveResult(False, i + 1, real_err(), fail, planning_time_s), q_ref

        err = real_err()
        it = 0
        while err > pos_tol and it < TAIL_IK_ITERS:
            q_ref = data.qpos.copy()
            one_step(target_pos, target_quat)
            if monitor is not None:
                fail = monitor()
                if fail:
                    return MoveResult(False, num_steps + it + 1, real_err(), fail, planning_time_s), q_ref
            err = real_err()
            it += 1
        return MoveResult(err <= pos_tol, num_steps + it, err,
                          planning_time_s=planning_time_s), q_ref
    except RuntimeError as e:
        return MoveResult(False, 0, real_err(),
                          Failure("ik_failed", phase.name, str(e)),
                          planning_time_s), q_ref


def make_run_episode(solve_ik: SolveIK, solver_name: str):
    """Build a run_episode(seed, gui, verbose) function bound to one IK solver."""

    def run_episode(seed: int | None = None, gui: bool = False, verbose: bool = True) -> dict:
        model, data = build_model(seed=seed)
        mujoco.mj_forward(model, data)

        result: dict = {
            "success": False, "reason": "incomplete", "failure_code": None,
            "failure_phase": None, "cube_start": None, "cube_end": None,
            "initial_pose": None, "grasp_yaw": None, "phases": {}, "placement": None,
            "solver": solver_name,
            "minimum_clearance": {"distance_m": None, "geom_pair": None, "phase": None, "samples": 0},
        }

        def fail_result(f: Failure) -> dict:
            result["success"] = False
            result["reason"] = f.reason
            result["failure_code"] = f.code
            result["failure_phase"] = f.phase
            return result

        try:
            scene = Scene(model)
        except KeyError as e:
            return fail_result(Failure("setup_error", "SETUP", f"model lookup failed: {e}"))

        viewer = None
        if gui:
            from mujoco import viewer as mj_viewer
            viewer = mj_viewer.launch_passive(model, data)

        try:
            _execute(model, data, scene, viewer, verbose, result, fail_result, solve_ik)
        finally:
            result["cube_end"] = data.xpos[scene.cube_body].tolist()
            if viewer is not None:
                if viewer.is_running():
                    time.sleep(2.0)
                viewer.close()
        return _jsonable(result)

    return run_episode


def _verify_lift(model, data, scene, cz_before, descend_z, lift_z):
    mujoco.mj_forward(model, data)
    cube_dz = float(data.xpos[scene.cube_body][2] - cz_before)
    planned_dz = float(lift_z - descend_z)
    if cube_dz < LIFT_FRACTION * planned_dz:
        return Failure("grasp_failed", Phase.LIFT.name,
                       f"cube did not rise with the gripper (rose {cube_dz*100:.1f} cm, "
                       f"expected ~{planned_dz*100:.1f} cm)")
    holding, forces = pads_hold_cube(model, data, scene)
    if not holding:
        return Failure("grasp_failed", Phase.LIFT.name,
                       f"pads lost the cube after lift (F_left={forces['L']:.2f} N, "
                       f"F_right={forces['R']:.2f} N)")
    return None


def _execute(model, data, scene: Scene, viewer, verbose: bool, result: dict, fail_result, solve_ik: SolveIK):
    cube_id = scene.cube_body
    result["cube_start"] = data.xpos[cube_id].copy().tolist()
    result["initial_pose"] = {
        "position_m": data.xpos[cube_id].copy().tolist(),
        "quaternion_wxyz": data.xquat[cube_id].copy().tolist(),
    }
    if verbose:
        cs = result["cube_start"]
        print(f"  Cube start: ({cs[0]:.3f}, {cs[1]:.3f}, {cs[2]:.3f})")

    q_ref = data.qpos.copy()
    hold_still(model, data, scene, q_ref, GRIPPER_OPEN_CMD, 200, viewer)
    mujoco.mj_forward(model, data)
    cx, cy, cz = data.xpos[cube_id].copy()
    pick_surface = cz - CUBE_HALF
    if TABLE_H_DESIGN is not None and abs(pick_surface - TABLE_H_DESIGN) > 0.01 and verbose:
        print(f"  [WARN] measured table height {pick_surface:.3f} m differs from "
              f"table_design.json ({TABLE_H_DESIGN:.3f} m)")

    floor = scene.floor_gid
    tray_info = {
        "center": data.geom_xpos[floor].copy(),
        "floor_top": geom_top_z(model, data, floor),
        "inner_half": model.geom_size[floor][:2].copy(),
    }
    wall_top = max(geom_top_z(model, data, g) for g in scene.wall_gids)

    yaw = grasp_yaw_for(cube_yaw(data, cube_id))
    result["grasp_yaw"] = yaw
    R_grasp = rot_down_yaw(yaw).as_matrix()
    target_quat = mat_to_quat(R_grasp)

    def site_z(surface_z: float, pad_height: float) -> float:
        return surface_z + pad_height + GRIPPER_OFFSET

    carry_pad_h = max(LIFT_HEIGHT, wall_top - pick_surface + CUBE_HALF + TRANSIT_CLEARANCE)
    approach_pos = np.array([cx, cy, site_z(pick_surface, PRE_GRASP_HEIGHT)])
    descend_pos = np.array([cx, cy, site_z(pick_surface, GRASP_HEIGHT)])
    lift_pos = np.array([cx, cy, site_z(pick_surface, carry_pad_h)])

    state = {"rel0": None}

    def make_monitor(phase: Phase):
        def monitor():
            record_min_clearance(phase)
            issues = check_collisions(model, data, scene, phase)
            if issues:
                return Failure("collision", phase.name, ", ".join(issues))
            if phase in CARRY_PHASES and state["rel0"] is not None:
                drift = float(np.linalg.norm(cube_rel_to_ee(model, data, scene) - state["rel0"]))
                if drift > CARRY_DRIFT_TOL:
                    return Failure("cube_dropped", phase.name,
                                   f"cube moved {drift*100:.1f} cm relative to the gripper")
            return None
        return monitor

    def record_min_clearance(phase: Phase):
        distance, pair = measure_min_clearance(model, data, scene, phase)
        metric = result["minimum_clearance"]
        metric["samples"] += 1
        if distance is not None and (metric["distance_m"] is None or distance < metric["distance_m"]):
            metric["distance_m"] = distance
            metric["geom_pair"] = pair
            metric["phase"] = phase.name

    gripper_cmd = GRIPPER_OPEN_CMD
    plan = {
        Phase.APPROACH: approach_pos, Phase.DESCEND: descend_pos, Phase.LIFT: lift_pos,
    }
    cz_after_grasp = cz

    for phase in PHASE_ORDER:
        if verbose:
            print(f"  Phase: {phase.name}", end="")
        t0 = time.time()

        if phase == Phase.GRASP:
            gripper_cmd = GRIPPER_CLOSE_CMD
            q_ref = data.qpos.copy()
            hold_still(model, data, scene, q_ref, gripper_cmd, SETTLE_STEPS, viewer)
            mujoco.mj_forward(model, data)
            record_min_clearance(phase)
            holding, forces = pads_hold_cube(model, data, scene)
            issues = check_collisions(model, data, scene, phase, final=True)
            result["phases"][phase.name] = {
                "ok": holding and not issues, "pad_forces_N": forces,
                "collisions": issues, "time": time.time() - t0,
            }
            if issues:
                f = Failure("collision", phase.name, ", ".join(issues))
            elif not holding:
                f = Failure("grasp_failed", phase.name,
                            f"no two-sided pad contact on cube "
                            f"(F_left={forces['L']:.2f} N, F_right={forces['R']:.2f} N, "
                            f"need >= {MIN_GRASP_FORCE} N each)")
            else:
                f = None
            if f:
                if verbose:
                    print(f" — FAILED ({f.reason})")
                fail_result(f)
                return
            state["rel0"] = cube_rel_to_ee(model, data, scene)
            cz_after_grasp = float(data.xpos[cube_id][2])
            off_xy = state["rel0"][:2]
            place_xy = tray_info["center"][:2] - off_xy
            plan[Phase.TRANSIT] = np.array([place_xy[0], place_xy[1], lift_pos[2]])
            plan[Phase.LOWER] = np.array([place_xy[0], place_xy[1],
                                          site_z(tray_info["floor_top"], GRASP_HEIGHT + LOWER_CLEARANCE)])
            plan[Phase.RELEASE] = np.array([place_xy[0], place_xy[1],
                                            site_z(tray_info["floor_top"], PRE_GRASP_HEIGHT)])
            if verbose:
                print(f" — OK (F_L={forces['L']:.1f} N, F_R={forces['R']:.1f} N, "
                      f"{time.time()-t0:.2f}s)")
            continue

        if phase == Phase.RELEASE:
            gripper_cmd = GRIPPER_OPEN_CMD
            q_ref = data.qpos.copy()
            hold_still(model, data, scene, q_ref, gripper_cmd, SETTLE_STEPS, viewer)
            mujoco.mj_forward(model, data)
            record_min_clearance(phase)
            still_holding, forces = pads_hold_cube(model, data, scene)
            if forces["L"] >= MIN_GRASP_FORCE or forces["R"] >= MIN_GRASP_FORCE:
                f = Failure("place_failed", phase.name,
                            f"cube still pressed by the pads after opening "
                            f"(F_left={forces['L']:.2f} N, F_right={forces['R']:.2f} N)")
                result["phases"][phase.name] = {"ok": False, "pad_forces_N": forces}
                if verbose:
                    print(f" — FAILED ({f.reason})")
                fail_result(f)
                return

            mv, q_ref = move_to_target(
                model, data, scene, q_ref, plan[phase], target_quat, gripper_cmd,
                phase, solve_ik, make_monitor(phase), viewer=viewer,
            )
            hold_still(model, data, scene, q_ref, gripper_cmd, PLACE_SETTLE_STEPS, viewer)
            placement = check_cube_placed(model, data, scene, tray_info)
            result["placement"] = placement
            ok_place = all(placement[k] for k in
                           ("inside_footprint", "on_floor", "upright", "at_rest", "released"))
            result["phases"][phase.name] = {
                "ok": ok_place and mv.failure is None and mv.converged,
                "converged": mv.converged, "steps": mv.steps, "pos_err": mv.err,
                "planning_time_s": mv.planning_time_s,
                "placement": placement, "time": time.time() - t0,
            }
            if mv.failure:
                fail_result(mv.failure)
                if verbose:
                    print(f" — FAILED ({mv.failure.reason})")
                return
            if not ok_place:
                bad = [k for k in ("inside_footprint", "on_floor", "upright", "at_rest", "released")
                       if not placement[k]]
                cp = placement["cube_pos"]
                f = Failure("place_failed", phase.name,
                            f"failed checks {bad}; cube at ({cp[0]:.3f}, {cp[1]:.3f}, {cp[2]:.3f})")
                if verbose:
                    print(f" — FAILED ({f.reason})")
                fail_result(f)
                return
            if not mv.converged:
                f = Failure("ik_diverged", phase.name, f"retreat err={mv.err:.4f} m")
                if verbose:
                    print(f" — FAILED ({f.reason})")
                fail_result(f)
                return
            result["success"] = True
            result["reason"] = "success"
            if verbose:
                print(f" — OK (cube in tray, offset "
                      f"{placement['offset_xy'][0]*100:.1f}/{placement['offset_xy'][1]*100:.1f} cm, "
                      f"{time.time()-t0:.2f}s)")
            return

        mv, q_ref = move_to_target(
            model, data, scene, q_ref, plan[phase], target_quat, gripper_cmd,
            phase, solve_ik, make_monitor(phase), viewer=viewer,
        )
        mujoco.mj_forward(model, data)
        record_min_clearance(phase)
        end_issues = check_collisions(model, data, scene, phase, final=True)

        entry = {
            "ok": mv.converged and mv.failure is None and not end_issues,
            "converged": mv.converged, "steps": mv.steps, "pos_err": mv.err,
            "planning_time_s": mv.planning_time_s,
            "collisions": end_issues, "time": time.time() - t0,
        }
        result["phases"][phase.name] = entry

        failure = mv.failure
        if failure is None and phase == Phase.LIFT:
            failure = _verify_lift(model, data, scene, cz_after_grasp,
                                   plan[Phase.DESCEND][2], plan[Phase.LIFT][2])
        if failure is None and end_issues:
            failure = Failure("collision", phase.name, ", ".join(end_issues))
        if failure is None and not mv.converged:
            failure = Failure("ik_diverged", phase.name, f"EE position error {mv.err:.4f} m")

        if verbose:
            status = "OK" if failure is None else "FAILED"
            print(f" — {status} ({mv.steps} steps, err={mv.err:.4f} m)", end="")
            print(f" -> {failure.reason}" if failure else "")
        if failure:
            entry["ok"] = False
            entry["failure"] = failure.reason
            fail_result(failure)
            return

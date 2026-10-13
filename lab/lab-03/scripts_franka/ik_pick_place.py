#!/usr/bin/env python3
"""
ik_pick_place.py — Task 9 / IK Phase 1: Pick-and-place with Mink (off-the-shelf IK)
==================================================================================
Implements a 7-phase state machine for pick-and-place using Mink (differential
IK), with phase-aware collision avoidance and collision checking between the
gripper, cube, table and tray, and with explicit logging of success / failure
and the reason for it.

Phases:
  1. APPROACH — Move EE above the cube (pre-grasp pose, gripper open)
  2. DESCEND  — Lower EE to the grasp height
  3. GRASP    — Close gripper around the cube (verify two-sided pad contact)
  4. LIFT     — Lift the cube vertically (verify it really comes along)
  5. TRANSIT  — Move to above the tray (place target)
  6. LOWER    — Lower to the tray surface
  7. RELEASE  — Open gripper, retreat upward, verify the cube is placed in the tray

What is checked / logged
  * IK collision limits (Mink) are rebuilt per phase so that intended contacts
    (pads <-> cube while grasping, cube/pads <-> tray floor while placing) are
    never "avoided" by the solver.
  * Contacts in the physics simulation are checked after every IK step (not only
    at the end of a phase) against a phase-aware rule table. Any disallowed
    contact aborts the episode with a specific failure code.
  * Grasp: both pads must press on the cube; after LIFT the cube must have risen
    with the gripper and must stay at a constant offset from the end-effector.
  * Place: cube footprint inside the tray, resting on the tray floor, upright,
    at rest, and no longer touched by the pads.
  * Every episode logs: success, reason, failure_code, failure_phase, per-phase
    results and the placement checks -> logs/ik_pick_place_log.json

Usage:
    cd ME-639/lab/lab-03
    source ../../venv/bin/activate
    python3 scripts/ik_pick_place.py                     # single episode, viewer
    python3 scripts/ik_pick_place.py --episodes 5 --no-gui
    python3 scripts/ik_pick_place.py --seed 42 --no-gui  # deterministic
    (On macOS the viewer needs `mjpython` instead of `python3`.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

import mujoco
import numpy as np

import mink

# ---------------------------------------------------------------------------
# Import the environment builder from Task 7
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env_pick_place import build_model, ARM_ACTUATORS  # noqa: E402

LAB = HERE.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IK_DT = 0.01              # IK integration timestep (s)
MOVE_SPEED = 0.03         # EE translation speed (m/s)
MOVE_ROT_SPEED = 0.5      # EE rotation speed (rad/s), only matters for yawed cubes
MIN_MOVE_TIME = 1.0       # s
TAIL_IK_ITERS = 150       # extra closed-loop IK steps to settle on the target
POS_TOL = 0.005           # 5 mm position convergence (measured on the real arm)
GRIPPER_OPEN_CMD = 0.0
GRIPPER_CLOSE_CMD = 255.0

EE_SITE = "right_center"
GRIPPER_OFFSET = 0.14     # Gripper pads are ~14 cm below EE site
PRE_GRASP_HEIGHT = 0.12   # pad height above surface for pre-grasp
CUBE_HALF = 0.025
GRASP_HEIGHT = CUBE_HALF + 0.01  # pad height above surface at grasp (= cube centre)
LIFT_HEIGHT = 0.15        # minimum pad height above the pick surface when carrying
TRANSIT_CLEARANCE = 0.03  # cube bottom must clear the tray walls by this much
LOWER_CLEARANCE = 0.003   # stop this far above the tray floor, then release
SETTLE_STEPS = 1000       # sim steps to let physics settle after a grip change
PLACE_SETTLE_STEPS = 500  # sim steps to let the cube come to rest after release

# Verification thresholds
CONTACT_DIST_TOL = 0.002  # m: contacts farther apart than this are ignored
MIN_GRASP_FORCE = 1.0     # N: normal force each pad side must exert on the cube
LIFT_FRACTION = 0.8       # cube must rise at least this fraction of the planned lift
CARRY_DRIFT_TOL = 0.03    # m: max change of cube-to-EE offset while carrying
CUBE_REST_SPEED = 0.02    # m/s, rad/s: "settled" threshold
ON_FLOOR_TOL = 0.01       # m: cube centre vs. (floor top + half cube)
UPRIGHT_MIN = 0.98        # cos of max tilt of the cube

# IK: all non-arm DoFs (cube free joint, finger joints) are frozen via a
# posture-task cost so the solver cannot "satisfy" a collision constraint by
# moving the cube or the fingers instead of the arm.
FREEZE_COST = 100.0

# Geometry names (from the Task-7 model)
PAD_GEOMS = ("gripper/left_pad1", "gripper/left_pad2",
             "gripper/right_pad1", "gripper/right_pad2")
ARM_COLLISION_GEOMS = ("link_2_collision", "link_3_collision", "link_4_collision",
                       "link_5_collision", "end_effector_collision")
TABLE_GEOMS = ("table_top",)
TRAY_FLOOR_GEOM = "tray_floor"
TRAY_WALL_GEOMS = ("tray_wall_px", "tray_wall_mx", "tray_wall_py", "tray_wall_my")

# Table design (only used for a sanity check against the measured table height)
try:
    with open(LAB / "table_design.json") as _f:
        TABLE_H_DESIGN: Optional[float] = json.load(_f)["table_height_m"]
except Exception:  # noqa: BLE001
    TABLE_H_DESIGN = None


# ---------------------------------------------------------------------------
# Phases and failures
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


PHASE_ORDER = [Phase.APPROACH, Phase.DESCEND, Phase.GRASP, Phase.LIFT,
               Phase.TRANSIT, Phase.LOWER, Phase.RELEASE]
CARRY_PHASES = (Phase.LIFT, Phase.TRANSIT, Phase.LOWER)


@dataclass
class Failure:
    """Why an episode failed. `code` is a short machine-readable category."""
    code: str      # setup_error | ik_failed | ik_diverged | collision | grasp_failed |
                   # cube_dropped | place_failed
    phase: str
    detail: str

    @property
    def reason(self) -> str:
        return f"{self.code}: {self.detail} (phase={self.phase})"


# ---------------------------------------------------------------------------
# Orientation: EE pointing straight down
# ---------------------------------------------------------------------------
def rot_down_yaw(yaw: float = 0.0) -> mink.SO3:
    """Rotation matrix: EE z-axis pointing down, with a yaw twist."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [c,   s,  0],
        [s,  -c,  0],
        [0,   0, -1],
    ])
    return mink.SO3.from_matrix(R)


def cube_yaw(data: mujoco.MjData, cube_body_id: int) -> float:
    R = data.xmat[cube_body_id].reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def grasp_yaw_for(cube_yaw_rad: float) -> float:
    """Gripper yaw that closes across a cube face, wrapped to [-45deg, 45deg].

    With rot_down_yaw(psi) the closing axis (EE y) is (sin psi, -cos psi), which is
    a face normal of a cube rotated by `psi`. Cube faces repeat every 90 deg.
    """
    return float((cube_yaw_rad + np.pi / 4) % (np.pi / 2) - np.pi / 4)


# ---------------------------------------------------------------------------
# Scene: ids, geom roles, actuator mapping
# ---------------------------------------------------------------------------
def _id(model: mujoco.MjModel, objtype, name: str) -> int:
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise KeyError(f"'{name}' not found in model")
    return int(i)


def build_roles(model: mujoco.MjModel) -> list[str]:
    """Classify every geom: pad / gripper / arm / cube / table / tray_floor / tray_wall / other."""
    roles = []
    for g in range(model.ngeom):
        gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        if gn in PAD_GEOMS:
            role = "pad"
        elif gn.startswith("gripper/") or bn.startswith("gripper/"):
            role = "gripper"
        elif bn == "pick_cube":
            role = "cube"
        elif gn in TABLE_GEOMS or bn == "table":
            role = "table"
        elif gn == TRAY_FLOOR_GEOM:
            role = "tray_floor"
        elif gn in TRAY_WALL_GEOMS or gn.startswith("tray_wall"):
            role = "tray_wall"
        elif re.fullmatch(r"link_[1-5]", bn) or bn == "end_effector":
            role = "arm"
        else:
            role = "other"
        roles.append(role)
    return roles


class Scene:
    """All model ids / lookups needed by the episode, resolved once."""

    def __init__(self, model: mujoco.MjModel):
        obj = mujoco.mjtObj
        self.cube_body = _id(model, obj.mjOBJ_BODY, "pick_cube")
        self.tray_body = _id(model, obj.mjOBJ_BODY, "tray")
        self.site = _id(model, obj.mjOBJ_SITE, EE_SITE)

        self.roles = build_roles(model)
        self.cube_geoms = [g for g in range(model.ngeom) if self.roles[g] == "cube"]
        self.pad_gids = [_id(model, obj.mjOBJ_GEOM, n) for n in PAD_GEOMS]
        self.pad_side = {g: ("L" if "left" in PAD_GEOMS[i] else "R")
                         for i, g in enumerate(self.pad_gids)}
        self.arm_gids = [_id(model, obj.mjOBJ_GEOM, n) for n in ARM_COLLISION_GEOMS]
        self.table_gids = [_id(model, obj.mjOBJ_GEOM, n) for n in TABLE_GEOMS]
        self.floor_gid = _id(model, obj.mjOBJ_GEOM, TRAY_FLOOR_GEOM)
        self.wall_gids = [_id(model, obj.mjOBJ_GEOM, n) for n in TRAY_WALL_GEOMS]
        self.tray_gids = [self.floor_gid] + self.wall_gids

        # Cube free joint
        if model.body_jntnum[self.cube_body] != 1:
            raise KeyError("pick_cube must have exactly one (free) joint")
        jid = int(model.body_jntadr[self.cube_body])
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            raise KeyError("pick_cube joint is not a free joint")
        self.cube_qadr = int(model.jnt_qposadr[jid])
        self.cube_dadr = int(model.jnt_dofadr[jid])

        # Arm actuators -> (actuator id, qpos address, dof address)
        self.arm_act: list[tuple[int, int, int]] = []
        for name in ARM_ACTUATORS:
            aid = mujoco.mj_name2id(model, obj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                continue
            jnt = int(model.actuator_trnid[aid][0])
            self.arm_act.append((int(aid), int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt])))
        if not self.arm_act:
            raise KeyError("no arm actuators found")
        self.grip_act = int(mujoco.mj_name2id(model, obj.mjOBJ_ACTUATOR, "gripper/fingers_actuator"))

        arm_q = {q for _, q, _ in self.arm_act}
        arm_d = {d for _, _, d in self.arm_act}
        self.passive_q = np.array([i for i in range(model.nq) if i not in arm_q])
        self.passive_d = np.array([i for i in range(model.nv) if i not in arm_d])


# ---------------------------------------------------------------------------
# Small geometry / state helpers
# ---------------------------------------------------------------------------
def site_pos(model, data, scene) -> np.ndarray:
    mujoco.mj_kinematics(model, data)
    return data.site_xpos[scene.site].copy()


def cube_rel_to_ee(model, data, scene) -> np.ndarray:
    """Cube centre minus EE site position (world frame)."""
    mujoco.mj_kinematics(model, data)
    return data.xpos[scene.cube_body] - data.site_xpos[scene.site]


def geom_top_z(model, data, gid: int) -> float:
    """Highest world-z point of a geom (exact for boxes, bounding sphere otherwise)."""
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
        R = data.geom_xmat[gid].reshape(3, 3)
        return float(data.geom_xpos[gid][2] + np.abs(R[2]) @ model.geom_size[gid])
    return float(data.geom_xpos[gid][2] + model.geom_rbound[gid])


def grasp_contact_forces(model, data, scene) -> dict[str, float]:
    """Summed pad->cube normal force for the left and right pads."""
    forces = {"L": 0.0, "R": 0.0}
    buf = np.zeros(6)
    for k in range(data.ncon):
        c = data.contact[k]
        for gp, gc in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            side = scene.pad_side.get(int(gp))
            if side and scene.roles[int(gc)] == "cube":
                mujoco.mj_contactForce(model, data, k, buf)
                forces[side] += abs(float(buf[0]))
    return forces


def pads_hold_cube(model, data, scene) -> tuple[bool, dict[str, float]]:
    f = grasp_contact_forces(model, data, scene)
    return (f["L"] >= MIN_GRASP_FORCE and f["R"] >= MIN_GRASP_FORCE), f


# ---------------------------------------------------------------------------
# Collision checking (simulation contacts, phase-aware)
# ---------------------------------------------------------------------------
def contact_violation(phase: Phase, a: str, b: str, final: bool) -> Optional[str]:
    """Return a label if the contact between roles `a` and `b` is NOT allowed in `phase`.

    Intended contacts (not violations):
      pad<->cube            from DESCEND on (grasp, carry, release)
      cube<->table          until the cube has been lifted
      cube/pad<->tray_floor while placing (LOWER, RELEASE)
    `final=True` is used for the end-of-phase check; it additionally requires that
    the cube is no longer touching the table once LIFT is complete.
    """
    if a == b:
        return None
    s = {a, b}

    def has(x, *others):
        return x in s and any(o in s for o in others)

    def other(x):
        rest = s - {x}
        return rest.pop() if rest else x

    # The arm links and the end-effector must never touch anything in the workspace.
    if has("arm", "table", "tray_floor", "tray_wall", "cube"):
        return f"arm-{other('arm')}"
    # Gripper body/fingers (non-pad) must not touch table or tray ...
    if has("gripper", "table", "tray_floor", "tray_wall"):
        return f"gripper-{other('gripper')}"
    # ... and may only touch the cube once the gripper is around it.
    if has("gripper", "cube") and phase == Phase.APPROACH:
        return "gripper-cube (unintended)"
    # Pads: never on table or tray walls; tray floor only while placing.
    if has("pad", "table"):
        return "pad-table"
    if has("pad", "tray_wall"):
        return "pad-tray_wall"
    if has("pad", "tray_floor") and phase not in (Phase.LOWER, Phase.RELEASE):
        return "pad-tray_floor"
    if has("pad", "cube") and phase == Phase.APPROACH:
        return "pad-cube (unintended)"
    # Cube: dragged over table / bumping the tray while carried
    if has("cube", "table"):
        if phase in (Phase.TRANSIT, Phase.LOWER) or (phase == Phase.LIFT and final):
            return "cube dragged on table"
    if has("cube", "tray_floor") and phase in (Phase.LIFT, Phase.TRANSIT):
        return "cube-tray_floor (too low)"
    if has("cube", "tray_wall") and phase in (Phase.LIFT, Phase.TRANSIT, Phase.LOWER):
        return "cube-tray_wall"
    return None


def check_collisions(model, data, scene: Scene, phase: Phase, final: bool = False) -> list[str]:
    """Return the list of disallowed contacts currently active in the simulation."""
    if final:
        mujoco.mj_forward(model, data)
    issues = set()
    for k in range(data.ncon):
        c = data.contact[k]
        if c.dist > CONTACT_DIST_TOL:
            continue
        v = contact_violation(phase, scene.roles[int(c.geom1)], scene.roles[int(c.geom2)], final)
        if v:
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
            issues.add(f"{v} [{n1} <-> {n2}]")
    return sorted(issues)


def measure_min_clearance(model, data, scene: Scene, phase: Phase) -> tuple[Optional[float], Optional[str]]:
    """Measure the smallest signed distance for phase-disallowed obstacle pairs."""
    moving_roles = {"arm", "gripper", "pad", "cube"}
    obstacle_roles = {"cube", "table", "tray_floor", "tray_wall"}
    candidates = []
    for g1, role1 in enumerate(scene.roles):
        if role1 not in moving_roles:
            continue
        for g2, role2 in enumerate(scene.roles):
            if g1 == g2 or role2 not in obstacle_roles:
                continue
            if role1 == role2 == "cube":
                continue
            if contact_violation(phase, role1, role2, final=False) is None:
                continue
            candidates.append((g1, g2))

    best = float("inf")
    best_pair = None
    fromto = np.zeros(6, dtype=np.float64)
    for g1, g2 in candidates:
        distance = float(mujoco.mj_geomDistance(model, data, g1, g2, 1.0, fromto))
        if distance < best:
            best = distance
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}"
            best_pair = f"{n1} <-> {n2}"
    return (None, None) if best_pair is None else (best, best_pair)


# ---------------------------------------------------------------------------
# Placement verification
# ---------------------------------------------------------------------------
def check_cube_placed(model, data, scene: Scene, tray_info: dict) -> dict:
    """Strict check that the cube ended up resting inside the tray, released."""
    mujoco.mj_forward(model, data)
    cube = data.xpos[scene.cube_body]
    R = data.xmat[scene.cube_body].reshape(3, 3)

    yaw = np.arctan2(R[1, 0], R[0, 0])
    ext = CUBE_HALF * (abs(np.cos(yaw)) + abs(np.sin(yaw)))   # footprint half-extent
    dx = abs(cube[0] - tray_info["center"][0])
    dy = abs(cube[1] - tray_info["center"][1])
    inside = bool(dx + ext < tray_info["inner_half"][0] and dy + ext < tray_info["inner_half"][1])

    on_floor = bool(abs(cube[2] - (tray_info["floor_top"] + CUBE_HALF)) < ON_FLOOR_TOL)
    upright = bool(R[2, 2] > UPRIGHT_MIN)
    speed = float(np.linalg.norm(data.qvel[scene.cube_dadr:scene.cube_dadr + 6]))
    at_rest = bool(speed < CUBE_REST_SPEED)
    f = grasp_contact_forces(model, data, scene)
    released = bool(f["L"] < MIN_GRASP_FORCE * 0.1 and f["R"] < MIN_GRASP_FORCE * 0.1)

    return {
        "inside_footprint": inside, "on_floor": on_floor, "upright": upright,
        "at_rest": at_rest, "released": released,
        "cube_pos": cube.tolist(), "offset_xy": [float(dx), float(dy)], "speed": speed,
    }


# ---------------------------------------------------------------------------
# IK context and low-level control
# ---------------------------------------------------------------------------
@dataclass
class IKContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    scene: Scene
    config: mink.Configuration
    ee_task: mink.FrameTask
    posture: mink.PostureTask

    def sync_all(self):
        """Reset the IK configuration to the real simulated state."""
        self.config.update(self.data.qpos.copy())

    def sync_passive(self):
        """Copy cube pose and finger joints from the sim into the IK configuration
        (keeps the arm's smooth open-loop reference), so that collision limits see
        the real cube/pad positions."""
        q = self.config.q.copy()
        q[self.scene.passive_q] = self.data.qpos[self.scene.passive_q]
        self.config.update(q)


def pd_control_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: Scene,
    target_q: np.ndarray,
    target_v: np.ndarray,
    gripper_cmd: float,
    kp: float = 500.0,
    kd: float = 50.0,
    steps: int = 1,
    viewer=None,
):
    """Apply PD (+ gravity/bias compensation) to track target joint positions and step physics."""
    for _ in range(steps):
        for aid, qadr, dadr in scene.arm_act:
            data.ctrl[aid] = (kp * (target_q[qadr] - data.qpos[qadr])
                              + kd * (target_v[dadr] - data.qvel[dadr])
                              + data.qfrc_bias[dadr])
        if scene.grip_act >= 0:
            data.ctrl[scene.grip_act] = gripper_cmd
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def hold_still(ctx: IKContext, gripper_cmd: float, steps: int, viewer=None):
    """Hold the current arm configuration while the gripper acts / the cube settles."""
    pd_control_step(ctx.model, ctx.data, ctx.scene, ctx.config.q,
                    np.zeros(ctx.model.nv), gripper_cmd, steps=steps, viewer=viewer)


def build_ik_limits(model: mujoco.MjModel, scene: Scene, phase: Phase) -> list:
    """Joint limits + collision avoidance, chosen per phase.

    The pad<->cube pair is deliberately absent from DESCEND onward so the solver
    does not fight the intended grasp. The cube pose is synced from the simulation
    every IK step, so cube pairs stay valid while carrying.
    """
    arm, pads = scene.arm_gids, scene.pad_gids
    table, tray = scene.table_gids, scene.tray_gids
    walls = scene.wall_gids
    cube = scene.cube_geoms

    if phase == Phase.APPROACH:
        pairs = [(arm + pads, table + tray + cube)]
    elif phase in (Phase.DESCEND, Phase.GRASP):
        pairs = [(arm, table + tray + cube), (pads, table + tray)]
    elif phase in (Phase.LIFT, Phase.TRANSIT):
        # the cube starts in contact with the table, so only the tray is a target for it
        pairs = [(arm + pads, table + tray), (cube, tray)]
    elif phase == Phase.LOWER:
        # pads and cube must be allowed to approach the tray floor
        pairs = [(arm, table + tray), (pads, table + walls), (cube, walls)]
    else:  # RELEASE (retreat)
        pairs = [(arm, table + tray), (pads, table + walls)]

    collision_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=pairs,
        gain=0.85,
        minimum_distance_from_collisions=0.01,
        collision_detection_distance=0.05,
    )
    return [mink.ConfigurationLimit(model), collision_limit]


@dataclass
class MoveResult:
    converged: bool
    steps: int
    err: float                       # real EE-site position error at the end (m)
    failure: Optional[Failure] = None
    planning_time_s: float = 0.0      # accumulated wall time in mink.solve_ik calls


Monitor = Callable[[], Optional[Failure]]


def move_to_target(
    ctx: IKContext,
    target: mink.SE3,
    gripper_cmd: float,
    limits: list,
    phase: Phase,
    monitor: Optional[Monitor] = None,
    pos_tol: float = POS_TOL,
    viewer=None,
) -> MoveResult:
    """Drive the EE to `target` along a straight-line interpolation using Mink + PD control.

    `monitor` is called after every IK step; if it returns a Failure the motion stops
    immediately and the failure is returned.
    """
    model, data, scene = ctx.model, ctx.data, ctx.scene
    ik_dt = IK_DT
    physics_steps = max(1, int(round(ik_dt / model.opt.timestep)))
    planning_time_s = 0.0

    def one_step(tgt: mink.SE3):
        nonlocal planning_time_s
        ctx.sync_passive()
        ctx.posture.set_target_from_configuration(ctx.config)
        ctx.ee_task.set_target(tgt)
        solve_started = time.perf_counter()
        vel = mink.solve_ik(ctx.config, [ctx.ee_task, ctx.posture], ik_dt,
                            solver="daqp", damping=1e-4, limits=limits)
        planning_time_s += time.perf_counter() - solve_started
        ctx.config.integrate_inplace(vel, ik_dt)
        pd_control_step(model, data, scene, ctx.config.q, vel, gripper_cmd,
                        steps=physics_steps, viewer=viewer)

    def real_err() -> float:
        return float(np.linalg.norm(site_pos(model, data, scene) - target.translation()))

    try:
        # Start every motion from the real arm state (no drift between phases)
        ctx.sync_all()
        start_pose = ctx.config.get_transform_frame_to_world(ctx.ee_task.frame_name,
                                                             ctx.ee_task.frame_type)
        dist = float(np.linalg.norm(target.translation() - start_pose.translation()))
        rot = float(np.linalg.norm((start_pose.rotation().inverse() @ target.rotation()).log()))
        duration = max(dist / MOVE_SPEED, rot / MOVE_ROT_SPEED, MIN_MOVE_TIME)
        num_steps = max(1, int(duration / ik_dt))

        for i in range(num_steps):
            alpha = min(1.0, (i + 1) / num_steps)
            one_step(start_pose.interpolate(target, alpha))
            if monitor is not None:
                fail = monitor()
                if fail:
                    return MoveResult(False, i + 1, real_err(), fail, planning_time_s)

        # Closed-loop settle: re-sync with the real arm so residual error is corrected
        err = real_err()
        it = 0
        while err > pos_tol and it < TAIL_IK_ITERS:
            ctx.sync_all()
            one_step(target)
            if monitor is not None:
                fail = monitor()
                if fail:
                    return MoveResult(False, num_steps + it + 1, real_err(), fail, planning_time_s)
            err = real_err()
            it += 1
        return MoveResult(err <= pos_tol, num_steps + it, err,
                          planning_time_s=planning_time_s)

    except mink.NoSolutionFound as e:
        return MoveResult(False, 0, real_err(),
                          Failure("ik_failed", phase.name, f"IK QP infeasible ({e})"),
                          planning_time_s)


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def run_episode(
    seed: int | None = None,
    gui: bool = False,
    verbose: bool = True,
) -> dict:
    """Run one pick-and-place episode.

    Returns a dict with keys: success, reason, failure_code, failure_phase,
    cube_start, cube_end, grasp_yaw, phases, placement.
    """
    model, data = build_model(seed=seed)
    mujoco.mj_forward(model, data)

    result: dict = {
        "success": False, "reason": "incomplete", "failure_code": None,
        "failure_phase": None, "cube_start": None, "cube_end": None,
        "initial_pose": None, "grasp_yaw": None, "phases": {}, "placement": None,
        "minimum_clearance": {"distance_m": None, "geom_pair": None, "phase": None, "samples": 0},
    }

    def fail_result(f: Failure) -> dict:
        result["success"] = False
        result["reason"] = f.reason
        result["failure_code"] = f.code
        result["failure_phase"] = f.phase
        return result

    # --- scene ids -----------------------------------------------------
    try:
        scene = Scene(model)
    except KeyError as e:
        return fail_result(Failure("setup_error", "SETUP", f"model lookup failed: {e}"))

    viewer = None
    if gui:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    try:
        _execute(model, data, scene, viewer, verbose, result, fail_result)
    finally:
        result["cube_end"] = data.xpos[scene.cube_body].tolist()
        if viewer is not None:
            if viewer.is_running():
                time.sleep(2.0)
            viewer.close()
    return _jsonable(result)


def _execute(model, data, scene: Scene, viewer, verbose: bool, result: dict, fail_result):
    cube_id = scene.cube_body
    result["cube_start"] = data.xpos[cube_id].copy().tolist()
    result["initial_pose"] = {
        "position_m": data.xpos[cube_id].copy().tolist(),
        "quaternion_wxyz": data.xquat[cube_id].copy().tolist(),
    }
    if verbose:
        cs = result["cube_start"]
        print(f"  Cube start: ({cs[0]:.3f}, {cs[1]:.3f}, {cs[2]:.3f})")

    # --- Mink configuration (start from the REAL simulated state) -------
    config = mink.Configuration(model)
    config.update(data.qpos.copy())

    ee_task = mink.FrameTask(
        frame_name=EE_SITE, frame_type="site",
        position_cost=1.0, orientation_cost=1.0, gain=1.0, lm_damping=1e-3,
    )
    cost = np.zeros(model.nv)
    cost[scene.passive_d] = FREEZE_COST
    posture = mink.PostureTask(model, cost=cost)
    ctx = IKContext(model, data, scene, config, ee_task, posture)

    # --- let the cube settle on the table ---------------------------------
    hold_still(ctx, GRIPPER_OPEN_CMD, 200, viewer)
    mujoco.mj_forward(model, data)
    cx, cy, cz = data.xpos[cube_id].copy()
    pick_surface = cz - CUBE_HALF
    if TABLE_H_DESIGN is not None and abs(pick_surface - TABLE_H_DESIGN) > 0.01 and verbose:
        print(f"  [WARN] measured table height {pick_surface:.3f} m differs from "
              f"table_design.json ({TABLE_H_DESIGN:.3f} m)")

    # --- tray geometry (measured from the model, not hard-coded) ----------
    floor = scene.floor_gid
    tray_info = {
        "center": data.geom_xpos[floor].copy(),
        "floor_top": geom_top_z(model, data, floor),
        "inner_half": model.geom_size[floor][:2].copy(),
    }
    wall_top = max(geom_top_z(model, data, g) for g in scene.wall_gids)

    # --- grasp orientation aligned with the cube faces --------------------
    yaw = grasp_yaw_for(cube_yaw(data, cube_id))
    result["grasp_yaw"] = yaw
    R_grasp = rot_down_yaw(yaw)

    def site_z(surface_z: float, pad_height: float) -> float:
        return surface_z + pad_height + GRIPPER_OFFSET

    # Pad height used while carrying: clear the tray walls with the cube's bottom
    carry_pad_h = max(LIFT_HEIGHT,
                      wall_top - pick_surface + CUBE_HALF + TRANSIT_CLEARANCE)

    approach_pos = np.array([cx, cy, site_z(pick_surface, PRE_GRASP_HEIGHT)])
    descend_pos = np.array([cx, cy, site_z(pick_surface, GRASP_HEIGHT)])
    lift_pos = np.array([cx, cy, site_z(pick_surface, carry_pad_h)])

    state = {"rel0": None}   # cube-to-EE offset right after the grasp

    def make_monitor(phase: Phase) -> Monitor:
        def monitor() -> Optional[Failure]:
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
    plan: dict[Phase, np.ndarray] = {
        Phase.APPROACH: approach_pos, Phase.DESCEND: descend_pos, Phase.LIFT: lift_pos,
    }
    cz_after_grasp = cz
    site_z_after_grasp = None

    for phase in PHASE_ORDER:
        if verbose:
            print(f"  Phase: {phase.name}", end="")
        t0 = time.time()

        # ============================ GRASP ==================================
        if phase == Phase.GRASP:
            gripper_cmd = GRIPPER_CLOSE_CMD
            hold_still(ctx, gripper_cmd, SETTLE_STEPS, viewer)
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
            site_z_after_grasp = float(site_pos(model, data, scene)[2])

            # Place targets: put the CUBE (not the EE) over the tray centre
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

        # ============================ RELEASE ================================
        if phase == Phase.RELEASE:
            gripper_cmd = GRIPPER_OPEN_CMD
            hold_still(ctx, gripper_cmd, SETTLE_STEPS, viewer)
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

            # Retreat upward
            target = mink.SE3.from_rotation_and_translation(R_grasp, plan[phase])
            try:
                limits = build_ik_limits(model, scene, phase)
            except Exception as e:  # noqa: BLE001
                f = Failure("setup_error", phase.name, f"collision-limit setup failed: {e}")
                fail_result(f)
                return
            mv = move_to_target(ctx, target, gripper_cmd, limits, phase, make_monitor(phase), viewer=viewer)

            hold_still(ctx, gripper_cmd, PLACE_SETTLE_STEPS, viewer)   # let the cube come to rest
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

        # ======================= MOTION PHASES ===============================
        target = mink.SE3.from_rotation_and_translation(R_grasp, plan[phase])
        try:
            limits = build_ik_limits(model, scene, phase)
        except Exception as e:  # noqa: BLE001
            # Collision checking is part of the task: do not silently continue without it.
            f = Failure("setup_error", phase.name, f"collision-limit setup failed: {e}")
            if verbose:
                print(f" — FAILED ({f.reason})")
            fail_result(f)
            return

        mv = move_to_target(ctx, target, gripper_cmd, limits, phase, make_monitor(phase), viewer=viewer)
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
            failure = _verify_lift(model, data, scene, cz_after_grasp, site_z_after_grasp,
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


def _verify_lift(model, data, scene, cz_before, site_z_before, descend_z, lift_z) -> Optional[Failure]:
    """After LIFT: the cube must have risen with the gripper and still be gripped."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_batch(
    run_episode_fn,
    *,
    episodes: int,
    seed: int,
    no_gui: bool,
    batch_prefix: str = "ik_batch",
    aggregate_name: str = "ik_pick_place_log.json",
    solver_name: str = "mink",
):
    """Run many pick-and-place episodes and write per-episode batch logs."""
    rng = np.random.default_rng(seed)

    results = []
    n_success = 0

    log_dir = LAB / "logs"
    log_dir.mkdir(exist_ok=True)
    batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = log_dir / f"{batch_prefix}_{batch_stamp}_seed{seed}"
    batch_dir.mkdir()

    for ep in range(episodes):
        ep_seed = int(rng.integers(0, 2**31))
        print(f"\n{'='*60}")
        print(f"Episode {ep+1}/{episodes}  (seed={ep_seed}, solver={solver_name})")
        print(f"{'='*60}")

        episode_started = time.perf_counter()
        result = run_episode_fn(seed=ep_seed, gui=not no_gui, verbose=True)
        result["solver"] = solver_name
        result["time_to_solve_s"] = time.perf_counter() - episode_started
        result["planning_time_s"] = sum(
            float(phase.get("planning_time_s", 0.0))
            for phase in result.get("phases", {}).values()
        )
        result["ik_iterations"] = sum(
            int(phase.get("steps", 0)) for phase in result.get("phases", {}).values()
        )
        result["episode"] = ep
        result["seed"] = ep_seed
        results.append(result)

        # Persist each demonstration immediately so an interrupted batch keeps completed episodes.
        with open(batch_dir / f"episode_{ep + 1:03d}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        with open(batch_dir / "episodes.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        if result["success"]:
            n_success += 1
            print(f"  ✓ SUCCESS (IK planning {result['planning_time_s']:.3f}s, "
                f"episode {result['time_to_solve_s']:.2f}s, "
                  f"{result['ik_iterations']} IK iterations)")
        else:
            print(f"  ✗ FAILURE: {result['reason']} (IK planning {result['planning_time_s']:.3f}s, "
                f"episode {result['time_to_solve_s']:.2f}s, {result['ik_iterations']} IK iterations)")

    print(f"\n{'='*60}")
    print(f"Summary: {n_success}/{episodes} successful "
          f"({100*n_success/max(episodes,1):.0f}%)")
    failures = Counter((r["failure_code"], r["failure_phase"]) for r in results if not r["success"])
    for (code, ph), n in failures.most_common():
        print(f"  {n}x {code} in {ph}")
    print(f"{'='*60}")

    planning_times = [r["planning_time_s"] for r in results]
    wall_times = [r["time_to_solve_s"] for r in results]
    clearances = [r["minimum_clearance"]["distance_m"] for r in results
                  if r.get("minimum_clearance", {}).get("distance_m") is not None]
    summary = {
        "solver": solver_name,
        "episodes": len(results),
        "successes": n_success,
        "success_rate": n_success / max(len(results), 1),
        "mean_ik_planning_time_s": float(np.mean(planning_times)) if planning_times else 0.0,
        "mean_episode_wall_time_s": float(np.mean(wall_times)) if wall_times else 0.0,
        "total_ik_iterations": sum(r["ik_iterations"] for r in results),
        "minimum_clearance_m": float(min(clearances)) if clearances else None,
        "seed": seed,
        "batch_directory": str(batch_dir),
    }
    with open(batch_dir / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Mean IK planning time: {summary['mean_ik_planning_time_s']:.4f}s")
    print(f"Mean episode wall time: {summary['mean_episode_wall_time_s']:.2f}s")
    print(f"Minimum clearance observed: {summary['minimum_clearance_m']} m")

    log_file = log_dir / aggregate_name
    with open(log_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved per-episode batch logs → {batch_dir}")
    print(f"Updated aggregate log → {log_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Task 9: IK pick-and-place with Mink")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gui", action="store_true", help="run headless (no MuJoCo viewer)")
    args = parser.parse_args()
    run_batch(
        run_episode,
        episodes=args.episodes,
        seed=args.seed,
        no_gui=args.no_gui,
        batch_prefix="ik_batch",
        aggregate_name="ik_pick_place_log.json",
        solver_name="mink",
    )


if __name__ == "__main__":
    main()
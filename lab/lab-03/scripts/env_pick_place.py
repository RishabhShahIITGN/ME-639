#!/usr/bin/env python3
"""
env_pick_place.py — Task 7: Full pick-and-place environment setup
=================================================================
Spawns the complete scene in MuJoCo:

  Scene
  ├── HEAL 6-DOF arm  +  Robotiq 2F-85 gripper  (via MjSpec attach)
  ├── Table            (workspace-fitted from table_design.json)
  ├── Red cube         (pick object, free joint, randomized spawn on table)
  └── Green tray       (placement target zone with shallow walls)

The cube spawns at a random position on the pick side of the table
(+Y half) each time the script runs. Gravity compensation is applied
to the arm joints so the robot holds its pose while the cube settles.

Usage:
    cd ME-639/lab/lab-03
    source ../../venv/bin/activate
    python3 scripts/env_pick_place.py          # launch viewer
    python3 scripts/env_pick_place.py --no-gui # headless sanity check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # lab-03/scripts/
LAB  = HERE.parent                              # lab-03/
DESC = LAB.parent / "ITR_mujoco_fk_lab" / "robot_descriptions"

ARM_XML  = DESC / "single_arm_heal_effort_actuation_rs_mj.xml"
GRIP_XML = DESC / "robotiq_2f85_v4" / "2f85.xml"
DESIGN   = LAB / "table_design.json"

# ---------------------------------------------------------------------------
# Table dimensions (from workspace analysis — Task 6)
# ---------------------------------------------------------------------------
with open(DESIGN) as f:
    _d = json.load(f)

TABLE_H      = _d["table_height_m"]          # 0.20
TABLE_THICK  = _d["table_thickness_m"]        # 0.04
TABLE_X_MIN  = _d["table_x_min_m"]           # 0.24
TABLE_X_MAX  = _d["table_x_max_m"]           # 0.58
TABLE_Y_MIN  = _d["table_y_min_m"]           # -0.30
TABLE_Y_MAX  = _d["table_y_max_m"]           # 0.30
TABLE_CX     = _d["table_center_xy_m"][0]    # 0.41
TABLE_CY     = _d["table_center_xy_m"][1]    # 0.00

TABLE_HX = 0.5 * (TABLE_X_MAX - TABLE_X_MIN)   # half-length X
TABLE_HY = 0.5 * (TABLE_Y_MAX - TABLE_Y_MIN)   # half-width  Y
TABLE_HZ = 0.5 * TABLE_THICK                    # half-thickness

# Cube
CUBE_HALF = 0.025  # 5 cm cube (half-size = 2.5 cm)
CUBE_MASS = 0.08   # 80 g

# Tray (placement target — shallow open-top box on the −Y half of the table)
TRAY_CX   = TABLE_CX
TRAY_CY   = -0.12                # centre of the tray (−Y = place side)
TRAY_HX   = 0.06                 # 12 cm wide
TRAY_HY   = 0.06                 # 12 cm deep
TRAY_WALL = 0.015                # wall height
TRAY_FLOOR = 0.002               # floor thickness
TRAY_Z    = TABLE_H + TRAY_FLOOR # top of the tray floor just above the table

# ---------------------------------------------------------------------------
# Build the scene model (robot + gripper + furniture)
# ---------------------------------------------------------------------------

def build_model(seed: int | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Construct the full pick-and-place MuJoCo model.

    1.  Load HEAL arm MjSpec and Robotiq 2F-85 MjSpec.
    2.  Attach the gripper at the ``right_center`` site.
    3.  Add the scene XML (table, cube, tray, floor, lights, cameras)
        via ``spec.worldbody.add_*`` calls.
    4.  Compile and return (model, data).
    """
    rng = np.random.default_rng(seed)

    # ---- 1. Robot + gripper via MjSpec attach ----
    arm  = mujoco.MjSpec.from_file(str(ARM_XML))
    grip = mujoco.MjSpec.from_file(str(GRIP_XML))
    arm.attach(grip, prefix="gripper/", site=arm.site("right_center"))

    wb = arm.worldbody

    # The base HEAL XML already defines a floor geom and one light,
    # so we only add a secondary fill light here.
    wb.add_light(
        name="fill_light",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[-0.5, 0.8, 1.0], dir=[0.5, -0.8, -0.5],
        diffuse=[0.3, 0.3, 0.3],
    )

    # Camera: slightly elevated, looking at the table centre
    arm.worldbody.add_camera(
        name="overhead", pos=[TABLE_CX, 0, 1.2],
        quat=[1, 0, 0, 0],  # looking straight down
        fovy=60,
    )

    # ---- 3. Table body ----
    body_z = TABLE_H - TABLE_HZ  # body origin = centre of the top slab
    leg_half = 0.5 * max(TABLE_H - TABLE_THICK, 0.02)
    leg_z = -TABLE_HZ - leg_half

    table = wb.add_body(name="table", pos=[TABLE_CX, TABLE_CY, body_z])
    table.add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[TABLE_HX, TABLE_HY, TABLE_HZ],
        friction=[1, 0.005, 0.0001],
        rgba=[0.72, 0.56, 0.35, 1],
    )
    table.add_site(name="table_surface", pos=[0, 0, TABLE_HZ], size=[0.001])

    # Four legs
    for lname, sx, sy in [
        ("leg_fl", -1, +1), ("leg_fr", +1, +1),
        ("leg_rl", -1, -1), ("leg_rr", +1, -1),
    ]:
        table.add_geom(
            name=lname, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.02, leg_half],
            pos=[sx * (TABLE_HX - 0.03), sy * (TABLE_HY - 0.03), leg_z],
            rgba=[0.35, 0.35, 0.35, 1],
        )

    # ---- 4. Red cube (pick object) — randomised on +Y half of table ----
    spawn_x = rng.uniform(TABLE_X_MIN + 0.04, TABLE_X_MAX - 0.04)
    spawn_y = rng.uniform(0.04, TABLE_Y_MAX - 0.04)      # +Y half
    spawn_z = TABLE_H + CUBE_HALF + 0.005                 # slight drop gap

    box = wb.add_body(name="pick_cube", pos=[spawn_x, spawn_y, spawn_z])
    box.mass = CUBE_MASS
    box.ipos = [0, 0, 0]
    inertia_val = CUBE_MASS * 2 * CUBE_HALF**2 / 3.0
    box.inertia = [inertia_val, inertia_val, inertia_val]
    box.add_freejoint(name="cube_free")
    box.add_geom(
        name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
        rgba=[0.9, 0.15, 0.15, 1],
        friction=[1, 0.5, 0.5],
        condim=4, mass=CUBE_MASS,
        contype=2, conaffinity=1,
    )

    # ---- 5. Green placement tray (shallow box with walls) ----
    tray_base_z = TABLE_H + TRAY_FLOOR
    tray = wb.add_body(name="tray", pos=[TRAY_CX, TRAY_CY, tray_base_z])
    # floor of the tray
    tray.add_geom(
        name="tray_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[TRAY_HX, TRAY_HY, TRAY_FLOOR],
        rgba=[0.15, 0.65, 0.25, 0.85],
        friction=[1, 0.5, 0.5],
    )
    # four walls
    wall_h = TRAY_WALL
    wall_t = 0.004   # wall thickness
    walls = [
        ("tray_wall_px", [+TRAY_HX, 0, wall_h], [wall_t, TRAY_HY, wall_h]),
        ("tray_wall_mx", [-TRAY_HX, 0, wall_h], [wall_t, TRAY_HY, wall_h]),
        ("tray_wall_py", [0, +TRAY_HY, wall_h], [TRAY_HX, wall_t, wall_h]),
        ("tray_wall_my", [0, -TRAY_HY, wall_h], [TRAY_HX, wall_t, wall_h]),
    ]
    for wname, wpos, wsize in walls:
        tray.add_geom(
            name=wname, type=mujoco.mjtGeom.mjGEOM_BOX,
            size=wsize, pos=wpos,
            rgba=[0.12, 0.55, 0.20, 0.85],
        )
    # Visual marker: small site at the tray centre for IK targeting
    tray.add_site(
        name="place_target", pos=[0, 0, TRAY_FLOOR + CUBE_HALF],
        size=[0.005], rgba=[0, 1, 0, 0.8],
    )

    # ---- 6. Compile ----
    model = arm.compile()
    data  = mujoco.MjData(model)

    return model, data


# ---------------------------------------------------------------------------
# Gravity compensation (arm only, skip gripper actuators)
# ---------------------------------------------------------------------------

def gravity_comp(model: mujoco.MjModel, data: mujoco.MjData,
                 arm_act_names: list[str]) -> None:
    """Apply qfrc_bias as control for the arm motors (not the gripper)."""
    for name in arm_act_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            continue
        dof = model.actuator_trnid[aid][0]
        data.ctrl[aid] = data.qfrc_bias[dof]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ARM_ACTUATORS = ["turret", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def print_scene_info(model: mujoco.MjModel, data: mujoco.MjData):
    """Print a summary of the loaded scene for verification."""
    print("\n" + "=" * 60)
    print("  HEAL Pick-and-Place Environment  —  Task 7")
    print("=" * 60)
    print(f"  Bodies       : {model.nbody}")
    print(f"  Joints       : {model.njnt}")
    print(f"  Actuators    : {model.nu}")
    print(f"  Geoms        : {model.ngeom}")
    print(f"  DOFs (nv)    : {model.nv}")
    print(f"  Timestep     : {model.opt.timestep:.4f} s")
    print()

    # Table
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    if tid >= 0:
        mujoco.mj_forward(model, data)
        print(f"  Table pos    : {data.xpos[tid]}")
    # Cube
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    if cid >= 0:
        print(f"  Cube  pos    : {data.xpos[cid]}")
    # Tray
    trid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")
    if trid >= 0:
        print(f"  Tray  pos    : {data.xpos[trid]}")
    # End effector
    eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
    if eid >= 0:
        print(f"  EE    pos    : {data.xpos[eid]}")

    print()
    # List actuators
    print("  Actuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"    [{i:2d}]  {name}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Task 7: HEAL pick-and-place env")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for cube spawn (None = random)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Headless: build scene, print info, exit")
    args = parser.parse_args()

    model, data = build_model(seed=args.seed)
    mujoco.mj_forward(model, data)
    print_scene_info(model, data)

    if args.no_gui:
        # Run 500 steps to let the cube settle, then report
        for _ in range(500):
            gravity_comp(model, data, ARM_ACTUATORS)
            mujoco.mj_step(model, data)
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
        print(f"After 500 steps → cube at {data.xpos[cid]}")
        print("Headless check passed ✓")
        return

    # Interactive viewer
    from mujoco.viewer import launch_passive

    with launch_passive(model, data) as viewer:
        while viewer.is_running():
            gravity_comp(model, data, ARM_ACTUATORS)
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()

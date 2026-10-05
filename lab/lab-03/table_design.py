"""
table_design.py — HEAL pick-and-place table from workspace analysis
===================================================================
Uses the same downward-approach samples as the task / dexterous occupancy maps
to choose a table footprint and height.

Height is selected so that:
  - a thick slab of downward TCP poses exists just above the surface
    (comfortable top-down approach)
  - there is vertical room for the Robotiq 2F-85 (~0.16 m flange-to-pad)
    plus a lift clearance above the object
  - the surface sits in a high-manipulability band (not at the workspace rim)

XY size is an axis-aligned rectangle fitted to the dexterous occupancy
(with a small inset so pick/place targets stay away from the boundary)
and offset in +X so the table does not collide with the robot base.

Outputs (saved next to this script):
  table_design_occupancy.png   task vs dexterous map + chosen table
  table_design_height.png      dexterous area vs candidate table height
  table_design_xz.png          XZ workspace with table surface and approach band
  table_design.json            numeric design used by the scene XML

Usage:
  cd ME-639/lab/lab-03
  source ../../venv/bin/activate
  python3 table_design.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mujoco
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
XML_PATH = (
    HERE.parent
    / "ITR_mujoco_fk_lab"
    / "robot_descriptions"
    / "single_arm_heal_effort_actuation_rs.xml"
)

N_SAMPLES = 200_000
SEED = 1
GRID = 0.05  # m
YAW_BINS = 6
MIN_YAW_BINS = 4  # dexterous if this many yaw sectors are reached
DOWN_DOT = 0.80  # ee z-axis · (0,0,-1)
SLAB = 0.10  # m, half-thickness of the TCP slab used as "at table approach height"
GRIPPER_LEN = 0.16  # Robotiq 2F-85 flange to pad, approx
LIFT_CLEARANCE = 0.10  # extra vertical room to lift an object
BASE_KEEP_OUT = 0.18  # m, table must not overlap the base
INSET = 0.03  # shrink fitted box so targets are not on the rim
MIN_X_FRONT = 0.20  # table starts in front of the robot

CANDIDATE_HEIGHTS = np.round(np.arange(0.08, 0.46, 0.04), 3)


def sample_downward(model, data, ee_id, qadr, limits, rng, n):
    pos = np.zeros((n, 3))
    z_axis = np.zeros((n, 3))
    x_axis = np.zeros((n, 3))
    for i in range(n):
        data.qpos[qadr] = rng.uniform(limits[:, 0], limits[:, 1])
        mujoco.mj_kinematics(model, data)
        pos[i] = data.xpos[ee_id]
        R = data.xmat[ee_id].reshape(3, 3)
        x_axis[i] = R[:, 0]
        z_axis[i] = R[:, 2]
    down = z_axis[:, 2] < -DOWN_DOT
    return pos[down], x_axis[down]


def occupancy(xy, yaw, x_edges, y_edges):
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    ix = np.clip(np.digitize(xy[:, 0], x_edges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(xy[:, 1], y_edges) - 1, 0, ny - 1)
    task = np.zeros((nx, ny), dtype=bool)
    dex = np.zeros((nx, ny), dtype=bool)
    bins = np.linspace(-np.pi, np.pi, YAW_BINS + 1)
    occupied = {}
    for i, j, a in zip(ix, iy, yaw):
        occupied.setdefault((i, j), []).append(a)
    for (i, j), angs in occupied.items():
        task[i, j] = True
        reached = np.unique(np.clip(np.digitize(angs, bins) - 1, 0, YAW_BINS - 1))
        if len(reached) >= MIN_YAW_BINS:
            dex[i, j] = True
    return task, dex


def cell_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def fit_table_xy(mask, x_c, y_c):
    """Front-of-robot rectangle, mirrored about y=0 (HEAL workspace is symmetric)."""
    ii, jj = np.where(mask)
    if len(ii) == 0:
        raise RuntimeError("No occupied cells to fit a table.")
    pts = np.column_stack([x_c[ii], y_c[jj]])
    front = pts[:, 0] >= MIN_X_FRONT
    pts = pts[front] if front.any() else pts

    x0, x1 = np.percentile(pts[:, 0], [12, 88])
    y_half = float(np.percentile(np.abs(pts[:, 1]), 80))
    x0 = max(x0, MIN_X_FRONT) + INSET
    x1 -= INSET
    y_half = max(0.20, y_half - INSET)
    # 1 cm CAD rounding, and clamp width so the rim stays inside the annulus
    x0, x1, y_half = np.round([x0, x1, min(y_half, 0.30)], 2)
    if x1 <= x0 or y_half <= 0:
        raise RuntimeError("Fitted table degenerated; check occupancy.")
    return float(x0), float(x1), float(-y_half), float(y_half)


def main():
    print("Loading HEAL model...")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}") for i in range(1, 7)]
    qadr = model.jnt_qposadr[jids]
    limits = np.array(
        [model.jnt_range[j] if model.jnt_limited[j] else (-np.pi, np.pi) for j in jids]
    )

    rng = np.random.default_rng(SEED)
    print(f"Sampling {N_SAMPLES} configs (keeping downward TCP poses)...")
    pos, x_axis = sample_downward(model, data, ee_id, qadr, limits, rng, N_SAMPLES)
    yaw = np.arctan2(x_axis[:, 1], x_axis[:, 0])
    print(f"  Downward poses: {len(pos)}")

    # Workspace point cloud from the earlier Monte Carlo (for XZ context)
    ws = np.load(HERE / "heal_workspace_data.npz")
    ws_pos = ws["pos"]

    scores = []
    for h in CANDIDATE_HEIGHTS:
        # TCP at grasp: table + gripper length. Approach band: that ± SLAB,
        # plus lift clearance still inside reach.
        z_grasp = h + GRIPPER_LEN
        z_lo, z_hi = z_grasp - SLAB, z_grasp + SLAB + LIFT_CLEARANCE
        m = (pos[:, 2] >= z_lo) & (pos[:, 2] <= z_hi)
        if m.sum() < 50:
            scores.append((h, 0.0, 0.0, int(m.sum())))
            continue
        pad = 0.05
        x_e = np.arange(pos[m, 0].min() - pad, pos[m, 0].max() + pad + GRID, GRID)
        y_e = np.arange(pos[m, 1].min() - pad, pos[m, 1].max() + pad + GRID, GRID)
        task, dex = occupancy(pos[m, :2], yaw[m], x_e, y_e)
        front = cell_centers(x_e) >= MIN_X_FRONT
        dex_front = dex[front, :] if front.any() else dex
        scores.append((h, task.mean() * task.size * GRID**2, dex_front.sum() * GRID**2, int(m.sum())))

    scores = np.array(scores, dtype=float)
    print("\nCandidate table heights (surface z in robot frame):")
    print("   h [m]   task area [m^2]   dex. area in +X [m^2]   n_down")
    for h, a_t, a_d, n in scores:
        print(f"  {h:5.2f}      {a_t:8.3f}           {a_d:8.3f}          {int(n)}")

    # Max dexterous (+X) area, with a preference for heights whose grasp pose
    # (table + gripper) sits in the high-manipulability core around z ≈ 0.35 m.
    grasp_z = scores[:, 0] + GRIPPER_LEN
    best = int(np.argmax(scores[:, 2] + 0.35 * scores[:, 1] - 0.8 * np.abs(grasp_z - 0.35)))
    table_h = float(scores[best, 0])
    print(f"\nChosen table height: {table_h:.2f} m (robot-base frame)")

    z_grasp = table_h + GRIPPER_LEN
    z_lo, z_hi = z_grasp - SLAB, z_grasp + SLAB + LIFT_CLEARANCE
    m = (pos[:, 2] >= z_lo) & (pos[:, 2] <= z_hi)
    pad = 0.08
    x_e = np.arange(pos[m, 0].min() - pad, pos[m, 0].max() + pad + GRID, GRID)
    y_e = np.arange(pos[m, 1].min() - pad, pos[m, 1].max() + pad + GRID, GRID)
    task, dex = occupancy(pos[m, :2], yaw[m], x_e, y_e)
    x_c, y_c = cell_centers(x_e), cell_centers(y_e)
    fit_mask = dex if dex.any() else task
    x0, x1, y0, y1 = fit_table_xy(fit_mask, x_c, y_c)

    # Keep table clear of the base cylinder
    x0 = max(x0, BASE_KEEP_OUT)
    length = x1 - x0
    width = y1 - y0
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    thickness = 0.04

    design = {
        "table_height_m": table_h,
        "table_thickness_m": thickness,
        "table_x_min_m": x0,
        "table_x_max_m": x1,
        "table_y_min_m": y0,
        "table_y_max_m": y1,
        "table_length_x_m": round(length, 3),
        "table_width_y_m": round(width, 3),
        "table_center_xy_m": [round(cx, 3), round(cy, 3)],
        "approach_band_z_m": [round(z_lo, 3), round(z_hi, 3)],
        "gripper_length_m": GRIPPER_LEN,
        "lift_clearance_m": LIFT_CLEARANCE,
        "base_keep_out_m": BASE_KEEP_OUT,
        "justification": {
            "height": (
                f"Surface at z={table_h:.2f} m puts downward TCP grasps near "
                f"z={z_grasp:.2f} m, inside the high-manipulability core of the "
                f"reachable sphere (not the z≈0.8 m rim). The band "
                f"[{z_lo:.2f}, {z_hi:.2f}] m covers the 2F-85 length "
                f"({GRIPPER_LEN:.2f} m) plus {LIFT_CLEARANCE:.2f} m lift."
            ),
            "footprint": (
                f"Rectangle {length:.2f}×{width:.2f} m in +X covers the dexterous "
                f"occupancy (arbitrary yaw) with a {INSET:.2f} m inset and "
                f"{BASE_KEEP_OUT:.2f} m clearance from the base."
            ),
        },
    }
    with open(HERE / "table_design.json", "w") as f:
        json.dump(design, f, indent=2)
    print("\nTable design:")
    print(json.dumps(design, indent=2))

    write_pick_place_xml(design)

    # ---- Plot 1: occupancy + table rectangle ----
    fig, ax = plt.subplots(figsize=(8.5, 7))
    xx, yy = np.meshgrid(x_c, y_c, indexing="ij")
    ax.scatter(xx[task], yy[task], c="lightsteelblue", s=36, marker="s",
               label="Task workspace (reachable, pointing down)", zorder=2)
    ax.scatter(xx[dex], yy[dex], c="crimson", s=36, marker="s",
               label="Dexterous workspace (arbitrary yaw)", zorder=3)
    ax.add_patch(Rectangle((x0, y0), length, width, fill=False, lw=2.2,
                           edgecolor="forestgreen", label="Chosen table"))
    ax.add_patch(mpatches.Circle((0, 0), BASE_KEEP_OUT, fill=False, ls="--",
                                 color="0.35", label="Base keep-out"))
    ax.plot(0, 0, "k^", ms=10, label="Robot base")
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Table footprint on task / dexterous occupancy\n"
        f"(surface z = {table_h:.2f} m, approach TCP z ∈ [{z_lo:.2f}, {z_hi:.2f}] m)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "table_design_occupancy.png", dpi=300)
    print("Saved table_design_occupancy.png")

    # ---- Plot 2: height sweep ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(scores[:, 0], scores[:, 1], "o-", color="steelblue", label="Task area")
    ax2.plot(scores[:, 0], scores[:, 2], "s-", color="crimson", label="Dexterous area (+X)")
    ax2.axvline(table_h, color="forestgreen", ls="--", label=f"Chosen h = {table_h:.2f} m")
    ax2.set_xlabel("Table surface height (m)")
    ax2.set_ylabel("Occupancy area (m²)")
    ax2.set_title("Feasible pick-and-place area vs table height")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(HERE / "table_design_height.png", dpi=300)
    print("Saved table_design_height.png")

    # ---- Plot 3: XZ with table and approach band ----
    sl = np.abs(ws_pos[:, 1]) < 0.03
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    ax3.scatter(ws_pos[sl, 0], ws_pos[sl, 2], s=3, c="0.75", alpha=0.5,
                label="Reachable workspace (|y|<3 cm)")
    ax3.add_patch(Rectangle((x0, table_h - thickness), length, thickness,
                            facecolor="saddlebrown", edgecolor="k", alpha=0.85,
                            label="Table (XZ slice)"))
    ax3.axhspan(z_lo, z_hi, color="forestgreen", alpha=0.15,
                label="Approach / lift band")
    ax3.axhline(table_h, color="saddlebrown", lw=1)
    ax3.set_aspect("equal")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_title("Table height vs reachable workspace (comfortable approach)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(HERE / "table_design_xz.png", dpi=300)
    print("Saved table_design_xz.png")

    print("\nDone.")


def write_pick_place_xml(design):
    """Write heal_pick_place.xml so the scene matches the fitted table."""
    h = design["table_height_m"]
    t = design["table_thickness_m"]
    x0, x1 = design["table_x_min_m"], design["table_x_max_m"]
    y0, y1 = design["table_y_min_m"], design["table_y_max_m"]
    cx, cy = design["table_center_xy_m"]
    hx = 0.5 * (x1 - x0)
    hy = 0.5 * (y1 - y0)
    hz = 0.5 * t
    body_z = h - hz
    cube = 0.03
    pick_y = min(0.12, 0.6 * hy)
    pick = (round(cx, 3), round(pick_y, 3), round(h + cube, 3))
    place = (round(cx, 3), round(-pick_y, 3), round(h + 0.001, 3))
    leg_half = 0.5 * max(h - t, 0.02)
    leg_local_z = -hz - leg_half
    xml = f"""<mujoco model="heal_pick_place">
  <!-- Table sized from task/dexterous workspace (see table_design.py / table_design.json).
       Surface z = {h:.2f} m, footprint {design['table_length_x_m']:.2f} x {design['table_width_y_m']:.2f} m
       centered at ({cx:.2f}, {cy:.2f}) m, in front of the base keep-out.
       Load this file from robot_descriptions/ so HEAL meshes resolve (see scripts/load_pick_place.py). -->
  <include file="single_arm_heal_effort_actuation_rs.xml"/>

  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05" pos="0 0 0" rgba="0.75 0.75 0.75 1"/>
    <light directional="true" pos="0.4 0 1.2" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>

    <body name="table" pos="{cx:.3f} {cy:.3f} {body_z:.3f}">
      <geom name="table_collision" type="box" size="{hx:.3f} {hy:.3f} {hz:.3f}"
            friction="1 0.005 0.0001" rgba="0.72 0.56 0.35 1"/>
      <site name="table_top" pos="0 0 {hz:.3f}" size="0.001"/>
      <geom name="leg_fl" type="cylinder" size="0.02 {leg_half:.3f}"
            pos="{-hx + 0.03:.3f} {hy - 0.03:.3f} {leg_local_z:.3f}" rgba="0.35 0.35 0.35 1"/>
      <geom name="leg_fr" type="cylinder" size="0.02 {leg_half:.3f}"
            pos="{hx - 0.03:.3f} {hy - 0.03:.3f} {leg_local_z:.3f}" rgba="0.35 0.35 0.35 1"/>
      <geom name="leg_rl" type="cylinder" size="0.02 {leg_half:.3f}"
            pos="{-hx + 0.03:.3f} {-hy + 0.03:.3f} {leg_local_z:.3f}" rgba="0.35 0.35 0.35 1"/>
      <geom name="leg_rr" type="cylinder" size="0.02 {leg_half:.3f}"
            pos="{hx - 0.03:.3f} {-hy + 0.03:.3f} {leg_local_z:.3f}" rgba="0.35 0.35 0.35 1"/>
    </body>

    <body name="box" pos="{pick[0]} {pick[1]} {pick[2]}">
      <inertial pos="0 0 0" mass="0.1" diaginertia="6e-5 6e-5 6e-5"/>
      <joint name="cube" type="free"/>
      <geom type="box" name="box_geom" size="{cube} {cube} {cube}"
            condim="3" friction="1 0.5 0.5" rgba="1 0 0 1"
            contype="2" conaffinity="1" mass="0.1"/>
    </body>
    <site name="place_target" pos="{place[0]} {place[1]} {place[2]}"
          type="box" size="0.05 0.05 0.001" rgba="0 1 0 0.5"/>
  </worldbody>
</mujoco>
"""
    desc = HERE.parent / "ITR_mujoco_fk_lab" / "robot_descriptions"
    out_desc = desc / "heal_pick_place.xml"
    out_desc.write_text(xml)
    print(f"Wrote {out_desc}")
    # Lab copy for the report; meshes resolve when loaded via scripts/load_pick_place.py
    out_lab = HERE / "heal_pick_place.xml"
    out_lab.write_text(xml)
    print(f"Wrote {out_lab}")


if __name__ == "__main__":
    main()

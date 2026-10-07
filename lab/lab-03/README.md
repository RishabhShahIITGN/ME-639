# Lab 03 — HEAL pick and place

## 5) Workspace analysis

Task workspace (reachable, gripper pointing down) and dexterous workspace (arbitrary yaw) are computed in `scripts/workspace_task.py` and `table_design.py`. Occupancy maps: `table_design_occupancy.png`.

## 6) Table design

Chosen from the occupancy maps (not the default 1.2 m × 1.2 m, 0.80 m robosuite table, which sits on the workspace rim and overlaps the base).

| Quantity | Value | Why |
|---|---|---|
| Surface height | **0.20 m** | Downward TCP grasps then sit near **z = 0.36 m**, in the high-manipulability core of `heal_workspace_xz.png` / `heal_workspace_vox.png`. |
| Footprint (X × Y) | **0.34 m × 0.60 m** | Rectangle fitted to the **dexterous** cells in front of the robot (`x ∈ [0.24, 0.58]`, `y ∈ [−0.30, 0.30]`), inset 3 cm from the rim. |
| Placement | center **(0.41, 0.00) m** | +X work area; 0.18 m keep-out around the base so the table does not collide with the turret. |
| Thickness | 0.04 m | Thin top; legs down to the floor. |

**Height / approach / clearance**

- Robotiq 2F-85 is about **0.16 m** flange-to-pad. With the surface at 0.20 m, a top-down grasp TCP is at ~0.36 m.
- The approach/lift band **z ∈ [0.26, 0.56] m** is still well inside the reachable sphere (max z ≈ 1.07 m, poor manipulability above ~0.8 m). That leaves ~10 cm to lift an object without stretching toward the workspace boundary, and enough vertical room for a slightly tilted approach rather than a perfectly vertical one.
- Lower tables (0.08–0.12 m) have slightly more occupancy area, but they squeeze the gripper against the surface. Taller tables (0.36–0.44 m) shrink both task and dexterous area and push grasps toward the rim. 0.20 m is the compromise: large front dexterous area and a comfortable approach band. See `table_design_height.png` and `table_design_xz.png`.

Regenerate numbers, plots, and `heal_pick_place.xml`:

```bash
cd lab/lab-03
source ../../venv/bin/activate
python3 table_design.py
```

## 7) Environment setup

Full pick-and-place scene built programmatically via MjSpec in `scripts/env_pick_place.py`.

| Component | Description |
|---|---|
| **Robot** | HEAL 6-DOF arm (`single_arm_heal_effort_actuation_rs_mj.xml`) with Robotiq 2F-85 gripper attached at the `right_center` site via `MjSpec.attach()` |
| **Table** | 0.34 × 0.60 m surface at z = 0.20 m, centered at (0.41, 0.00) m — from Task 6 (`table_design.json`) |
| **Pick cube** | 5 cm red cube (80 g, free joint) spawned at a random position on the +Y half of the table each run |
| **Placement tray** | 12 × 12 cm green tray with 1.5 cm raised walls on the −Y half of the table, centered at (0.41, −0.12) m |
| **Gravity comp** | `qfrc_bias` feedforward on the 6 arm motors; gripper actuator is left free for explicit control |

```bash
# Launch interactive viewer (cube position randomized each run)
cd lab/lab-03
source ../../venv/bin/activate
python3 scripts/env_pick_place.py

# Headless sanity check with fixed seed
python3 scripts/env_pick_place.py --no-gui --seed 42
```

## 8) Episode generation and randomization

To run multi-episode data collection and logging:
```bash
cd lab/lab-03
source ../../venv/bin/activate
python3 scripts/run_episodes.py --episodes 10 --steps 200
```
- A loop iterates over `N` episodes using `mujoco.mj_resetData`.
- The cube pose is sampled inside the designed table surface on the pick side (+Y).
- Samples $x \in [X_{min}, X_{max}]$, $y \in [Y_{min}, Y_{max}]$, and $yaw \in [-\pi, \pi]$ using uniform randomization.
- Logs cube position, cube quaternion, end-effector position, and time at each step into `.npz` files in `lab/lab-03/logs/`.

## 9) IK Phase 1: Off‑the‑shelf IK (Mink)

Differential IK pipeline using [Mink](https://github.com/kevinzakka/mink) for end-effector trajectory planning combined with joint-space PD control and collision avoidance.

### Execution Phases:
1. **APPROACH**: Pre-grasp pose directly above randomized cube position ($z = \text{table\_h} + 0.12\text{m} + \text{offset}$).
2. **DESCEND**: Lower end-effector to grasp height ($z = \text{table\_h} + 0.025\text{m} + \text{offset}$).
3. **GRASP**: Close 2F-85 gripper fingers around cube.
4. **LIFT**: Retract vertically to lift height ($z = \text{table\_h} + 0.15\text{m} + \text{offset}$).
5. **TRANSIT**: Transfer end-effector to the tray target site on the $-Y$ table section.
6. **LOWER**: Lower cube into target tray.
7. **RELEASE**: Open gripper, drop object into tray, and retreat upward.

### Collision Avoidance & Safety:
- Implements `mink.CollisionAvoidanceLimit` between arm links/gripper pads and the table/tray surfaces.
- Step-by-step collision inspection between arm/gripper base and furniture.

### Usage:
```bash
cd lab/lab-03
source ../../venv/bin/activate

# Interactive viewer (run a single pick-and-place episode)
python3 scripts/ik_pick_place.py

# Run batch test headless (e.g., 5 episodes)
python3 scripts/ik_pick_place.py --episodes 5 --no-gui

# Deterministic seed test
python3 scripts/ik_pick_place.py --episodes 1 --seed 42 --no-gui
```
Detailed execution logs and per-phase stats are saved to `lab/lab-03/logs/ik_pick_place_log.json`.


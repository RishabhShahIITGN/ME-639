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

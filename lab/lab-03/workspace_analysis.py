"""
workspace_analysis.py — HEAL Manipulator Workspace & Dexterity Analysis
========================================================================
Monte Carlo sampling of the HEAL robot's reachable workspace. For each valid
random joint configuration it records the end-effector position and several
dexterity measures:

  - w_pos : position-only Yoshikawa measure  sqrt(det(Jp Jp^T))   [m^3]
  - w_full: |det(J)| of the full 6x6 Jacobian (mixes m and rad units)
  - s_min : smallest singular value of J (distance to singularity)
  - cond  : condition number of J (lower = more isotropic)

Outputs (saved next to this script):
  heal_workspace.png      3D scatter, colored by log position manipulability
  heal_workspace_xz.png   XZ cross-section (thin slab around y = 0)
  heal_workspace_vox.png  XZ projection of per-voxel max manipulability
  heal_workspace_data.npz raw samples for the report

Usage:
  cd ME-639/lab/lab-03
  source ../../venv/bin/activate
  python3 workspace_analysis.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mujoco

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
XML_PATH = (HERE.parent / "ITR_mujoco_fk_lab" / "robot_descriptions"
            / "single_arm_heal_effort_actuation_rs.xml")

EE_BODY_NAME = "end_effector"
EE_SITE_NAME = None          # set to a site name if you have a TCP site
JOINT_NAMES = [f"joint_{i + 1}" for i in range(6)]

N_SAMPLES = 50_000
SEED = 0
REJECT_SELF_COLLISION = True
N_PLOT_3D = 15_000           # subsample size for the 3D scatter
SLICE_HALF_WIDTH = 0.02      # m, half thickness of the y = 0 slab
VOXEL_SIZE = 0.02            # m
EPS = 1e-12                  # floor for log-scale plotting

# ----------------------------------------------------------------------------
# Model setup
# ----------------------------------------------------------------------------
print("Loading HEAL model...")
if not XML_PATH.exists():
    raise FileNotFoundError(f"Model XML not found: {XML_PATH}")

# from_xml_path resolves meshes relative to the XML, so no os.chdir needed
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

# End effector: body origin, or a site if one is specified
if EE_SITE_NAME:
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    assert ee_id >= 0, f"Site '{EE_SITE_NAME}' not found"
else:
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    assert ee_id >= 0, f"Body '{EE_BODY_NAME}' not found"

# Joints: resolve IDs, then the qpos / dof addresses (do not assume order)
jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
missing = [n for n, j in zip(JOINT_NAMES, jids) if j < 0]
assert not missing, f"Joint(s) not found: {missing}"

qadr = model.jnt_qposadr[jids]
dadr = model.jnt_dofadr[jids]
limits = np.array([
    model.jnt_range[j] if model.jnt_limited[j] else (-np.pi, np.pi)
    for j in jids
])
print("Joint limits (rad):")
for n, (lo, hi) in zip(JOINT_NAMES, limits):
    print(f"  {n}: [{lo:+.3f}, {hi:+.3f}]")

# ----------------------------------------------------------------------------
# Collision filter
# ----------------------------------------------------------------------------
def has_self_collision(m, d):
    """True if any contact is between non-adjacent bodies (after mj_forward)."""
    for k in range(d.ncon):
        c = d.contact[k]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        if b1 == b2:
            continue
        # ignore parent-child pairs (neighbouring links always touch at joints)
        if m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1:
            continue
        return True
    return False

# ----------------------------------------------------------------------------
# Monte Carlo sampling
# ----------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

pos = np.zeros((N_SAMPLES, 3))
w_pos = np.zeros(N_SAMPLES)
w_full = np.zeros(N_SAMPLES)
s_min = np.zeros(N_SAMPLES)
cond = np.zeros(N_SAMPLES)
valid = np.ones(N_SAMPLES, dtype=bool)

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

print(f"Sampling {N_SAMPLES} random joint configurations...")
for i in range(N_SAMPLES):
    data.qpos[qadr] = rng.uniform(limits[:, 0], limits[:, 1])
    mujoco.mj_forward(model, data)   # kinematics, COM pos, and collisions

    if REJECT_SELF_COLLISION and has_self_collision(model, data):
        valid[i] = False
        continue

    if EE_SITE_NAME:
        pos[i] = data.site_xpos[ee_id]
        mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
    else:
        pos[i] = data.xpos[ee_id]
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)

    Jp = jacp[:, dadr]                      # 3x6 translational
    J = np.vstack((Jp, jacr[:, dadr]))      # 6x6 full

    s = np.linalg.svd(J, compute_uv=False)
    w_full[i] = np.prod(s)                  # |det J|
    s_min[i] = s[-1]
    cond[i] = s[0] / max(s[-1], EPS)
    w_pos[i] = np.sqrt(max(np.linalg.det(Jp @ Jp.T), 0.0))

    if (i + 1) % 10_000 == 0:
        print(f"  Processed {i + 1} samples...")

n_valid = int(valid.sum())
print(f"Sampling complete. Valid: {n_valid}/{N_SAMPLES} "
      f"({100 * n_valid / N_SAMPLES:.1f}%)")

pos, w_pos, w_full, s_min, cond = (a[valid] for a in (pos, w_pos, w_full, s_min, cond))

# Summary stats (useful for the report)
print("\nWorkspace extents (m):")
for ax_name, k in zip("XYZ", range(3)):
    print(f"  {ax_name}: [{pos[:, k].min():+.3f}, {pos[:, k].max():+.3f}]")
print(f"Max reach from origin: {np.linalg.norm(pos, axis=1).max():.3f} m")
print(f"Near-singular configs (s_min < 1e-3): {100 * np.mean(s_min < 1e-3):.1f}%")
print(f"Median w_pos: {np.median(w_pos):.3e} m^3, "
      f"median cond(J): {np.median(cond):.1f}")

np.savez_compressed(HERE / "heal_workspace_data.npz",
                    pos=pos, w_pos=w_pos, w_full=w_full, s_min=s_min, cond=cond)

# ----------------------------------------------------------------------------
# Plot 1: 3D scatter (subsampled, log color scale)
# ----------------------------------------------------------------------------
print("\nPlotting...")
w_plot = np.maximum(w_pos, EPS)
lo_c, hi_c = np.percentile(w_plot, [1, 99])
norm = LogNorm(vmin=max(lo_c, EPS), vmax=hi_c)

idx = rng.choice(len(pos), size=min(N_PLOT_3D, len(pos)), replace=False)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(pos[idx, 0], pos[idx, 1], pos[idx, 2],
                c=w_plot[idx], cmap="viridis", norm=norm, s=2, alpha=0.6)
plt.colorbar(sc, ax=ax, pad=0.1, label="Position manipulability (m³, log)")
ax.set_title("HEAL Manipulator Reachable Workspace & Dexterity")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
ax.set_box_aspect((np.ptp(pos[:, 0]), np.ptp(pos[:, 1]), np.ptp(pos[:, 2])))
fig.savefig(HERE / "heal_workspace.png", dpi=300)
print("Saved heal_workspace.png")

# ----------------------------------------------------------------------------
# Plot 2: XZ cross-section (thin slab around y = 0)
# ----------------------------------------------------------------------------
sl = np.abs(pos[:, 1]) < SLICE_HALF_WIDTH
fig2, ax2 = plt.subplots(figsize=(7, 7))
sc2 = ax2.scatter(pos[sl, 0], pos[sl, 2], c=w_plot[sl],
                  cmap="viridis", norm=norm, s=3)
ax2.set_aspect("equal")
ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z (m)")
ax2.set_title(f"XZ cross-section (|y| < {SLICE_HALF_WIDTH} m, {sl.sum()} pts)")
plt.colorbar(sc2, ax=ax2, label="Position manipulability (m³, log)")
fig2.savefig(HERE / "heal_workspace_xz.png", dpi=300)
print("Saved heal_workspace_xz.png")

# ----------------------------------------------------------------------------
# Plot 3: Voxel map, max manipulability per voxel, projected onto XZ
# ----------------------------------------------------------------------------
mins = pos.min(axis=0)
vox = np.floor((pos - mins) / VOXEL_SIZE).astype(int)
dims = vox.max(axis=0) + 1

grid = np.zeros(dims)
np.maximum.at(grid, (vox[:, 0], vox[:, 1], vox[:, 2]), w_pos)
proj = grid.max(axis=1)                      # max over Y -> (X, Z)
proj_masked = np.ma.masked_less_equal(proj, 0)

fig3, ax3 = plt.subplots(figsize=(7, 7))
extent = [mins[0], mins[0] + dims[0] * VOXEL_SIZE,
          mins[2], mins[2] + dims[2] * VOXEL_SIZE]
im = ax3.imshow(proj_masked.T, origin="lower", extent=extent, cmap="viridis",
                norm=LogNorm(vmin=max(proj[proj > 0].min(), EPS),
                             vmax=proj.max()))
ax3.set_xlabel("X (m)"); ax3.set_ylabel("Z (m)")
ax3.set_title(f"Max manipulability per voxel ({VOXEL_SIZE * 100:.0f} cm), XZ projection")
plt.colorbar(im, ax=ax3, label="Position manipulability (m³, log)")
fig3.savefig(HERE / "heal_workspace_vox.png", dpi=300)
print("Saved heal_workspace_vox.png")

try:
    plt.show()
except Exception:
    print("Could not display plots interactively; check the saved PNG files.")
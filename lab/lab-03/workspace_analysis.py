"""
workspace_analysis.py — HEAL Manipulator Workspace & Dexterity Analysis
========================================================================
This script uses Monte Carlo sampling to generate the reachable workspace
of the HEAL robot. It computes the end-effector position for random joint
configurations and calculates the Manipulability Index (Yoshikawa) to 
visualize the "dexterous" areas of the workspace.

Usage:
  cd ME-639/lab/lab-03
  source ../../venv/bin/activate
  python3 workspace_analysis.py
"""

import mujoco
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def setup_working_directory():
    """Ensure we are in robot_descriptions/ so MuJoCo resolves relative mesh paths."""
    candidates = [
        "../ITR_mujoco_fk_lab/robot_descriptions",
        "../../lab/ITR_mujoco_fk_lab/robot_descriptions",
    ]
    for c in candidates:
        if os.path.exists(c):
            os.chdir(c)
            return
    raise FileNotFoundError("Could not find 'robot_descriptions'. Run from lab/lab-03.")

print("Loading HEAL model...")
setup_working_directory()
model = mujoco.MjModel.from_xml_path("single_arm_heal_effort_actuation_rs.xml")
data = mujoco.MjData(model)

EE_BODY_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
NUM_JOINTS = 6

# Get joint limits
joint_limits = []
for i in range(NUM_JOINTS):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i+1}")
    if model.jnt_limited[jid]:
        joint_limits.append((model.jnt_range[jid, 0], model.jnt_range[jid, 1]))
    else:
        joint_limits.append((-np.pi, np.pi))

print(f"Joint limits: {joint_limits}")

# Monte Carlo Sampling
N_SAMPLES = 50000
print(f"Sampling {N_SAMPLES} random joint configurations...")

positions = np.zeros((N_SAMPLES, 3))
manipulability = np.zeros(N_SAMPLES)

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

for i in range(N_SAMPLES):
    # Sample random joints within limits
    q_rand = [np.random.uniform(lo, hi) for lo, hi in joint_limits]
    data.qpos[:NUM_JOINTS] = q_rand
    
    # Compute Forward Kinematics
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data) # Required before jacobian
    
    positions[i] = data.xpos[EE_BODY_ID]
    
    # Compute Jacobian to find Dexterity / Manipulability
    mujoco.mj_jacBody(model, data, jacp, jacr, EE_BODY_ID)
    
    # Full 6x6 Jacobian (translation and rotation)
    J = np.vstack((jacp[:, :NUM_JOINTS], jacr[:, :NUM_JOINTS]))
    
    # Yoshikawa Manipulability Measure: w = sqrt(det(J * J^T))
    # Note: J is 6x6, so det(J*J^T) = det(J)^2. 
    # We add a small epsilon to avoid negative values due to floating point before sqrt.
    det_J = np.linalg.det(J)
    manipulability[i] = np.abs(det_J)

    if i % 10000 == 0 and i > 0:
        print(f"  Processed {i} samples...")

print("Sampling complete. Plotting workspace...")

# Filter out singular configurations for better color scaling
m_min, m_max = np.percentile(manipulability, [5, 95])
manipulability = np.clip(manipulability, m_min, m_max)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot colored by manipulability
sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                c=manipulability, cmap='viridis', s=2, alpha=0.6)

cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Manipulability Index (Dexterity)')

ax.set_title("HEAL Manipulator Reachable Workspace & Dexterity")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# Keep axes scaled equally
max_range = np.array([positions[:, 0].max()-positions[:, 0].min(), 
                      positions[:, 1].max()-positions[:, 1].min(), 
                      positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Save plot instead of just showing it (useful if running headless)
plt.savefig("heal_workspace.png", dpi=300)
print("Workspace plot saved as 'heal_workspace.png'")

try:
    plt.show()
except Exception as e:
    print("Could not display plot interactively. Please check 'heal_workspace.png'.")

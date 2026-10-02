import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os

print("Setting up HEAL kinematics...")
os.chdir('../ITR_mujoco_fk_lab/robot_descriptions')
model = mujoco.MjModel.from_xml_path('single_arm_heal_effort_actuation_rs.xml')
data = mujoco.MjData(model)
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'end_effector')

# Fast numpy FK logic (reusing our analytical_fk_heal logic for millions of points)
def rotmat_z(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

# Let's just use mujoco in a loop for 200,000 points. It takes ~2 seconds in python.
N = 300000
print(f"Sampling {N} points...")
q_samples = np.random.uniform(-np.pi, np.pi, (N, 6))

positions = np.zeros((N, 3))
z_axis_dir = np.zeros((N, 3))

for i in range(N):
    data.qpos[:6] = q_samples[i]
    mujoco.mj_kinematics(model, data)
    positions[i] = data.xpos[ee_id]
    z_axis_dir[i] = data.xmat[ee_id].reshape(3,3)[:, 2] # The Z axis of the end effector in world frame

# 1. We only care about points near the table surface (Task workspace height)
# Let's define the table surface as Z in [-0.2, 0.2] relative to robot base.
table_z_min, table_z_max = -0.1, 0.3
z_mask = (positions[:, 2] >= table_z_min) & (positions[:, 2] <= table_z_max)

# 2. We only care about points where the end-effector is pointing "down" 
# (e.g. dot product with [0,0,-1] > 0.8)
pointing_down_mask = z_axis_dir[:, 2] < -0.8

valid_mask = z_mask & pointing_down_mask
valid_positions = positions[valid_mask]
valid_q = q_samples[valid_mask]

print(f"Found {len(valid_positions)} valid points pointing down at table height.")

# Grid the table surface to find Task Workspace vs Dexterous Workspace
grid_resolution = 0.05 # 5cm voxels
min_xy = np.min(valid_positions[:, :2], axis=0) - grid_resolution
max_xy = np.max(valid_positions[:, :2], axis=0) + grid_resolution

nx = int(np.ceil((max_xy[0] - min_xy[0]) / grid_resolution))
ny = int(np.ceil((max_xy[1] - min_xy[1]) / grid_resolution))

# We will store the yaw angles reached in each voxel
# Yaw is calculated from the X axis of the end-effector projected onto the world XY plane
valid_x_axis = np.zeros((len(valid_positions), 3))
for i, q in enumerate(valid_q):
    data.qpos[:6] = q
    mujoco.mj_kinematics(model, data)
    valid_x_axis[i] = data.xmat[ee_id].reshape(3,3)[:, 0]

# Yaw angle in [-pi, pi]
yaw_angles = np.arctan2(valid_x_axis[:, 1], valid_x_axis[:, 0])

grid = [[[] for _ in range(ny)] for _ in range(nx)]

for pos, yaw in zip(valid_positions, yaw_angles):
    ix = int((pos[0] - min_xy[0]) / grid_resolution)
    iy = int((pos[1] - min_xy[1]) / grid_resolution)
    if 0 <= ix < nx and 0 <= iy < ny:
        grid[ix][iy].append(yaw)

task_workspace_pts = []
dexterous_workspace_pts = []

# A voxel is dexterous if it covers at least 6 out of 8 octants of yaw
NUM_YAW_BINS = 8
for ix in range(nx):
    for iy in range(ny):
        if len(grid[ix][iy]) > 0:
            x = min_xy[0] + (ix + 0.5) * grid_resolution
            y = min_xy[1] + (iy + 0.5) * grid_resolution
            
            task_workspace_pts.append([x, y])
            
            # Check yaw diversity
            yaws = np.array(grid[ix][iy])
            bins_reached = len(np.unique(np.digitize(yaws, np.linspace(-np.pi, np.pi, NUM_YAW_BINS+1))))
            
            if bins_reached >= NUM_YAW_BINS - 1: # Highly dexterous (covers almost all yaw angles)
                dexterous_workspace_pts.append([x, y])

task_workspace_pts = np.array(task_workspace_pts)
dexterous_workspace_pts = np.array(dexterous_workspace_pts)

print(f"Task workspace voxels: {len(task_workspace_pts)}")
print(f"Dexterous workspace voxels: {len(dexterous_workspace_pts)}")

# --- PLOTTING ---
plt.figure(figsize=(10, 8))
if len(task_workspace_pts) > 0:
    plt.scatter(task_workspace_pts[:, 0], task_workspace_pts[:, 1], c='lightblue', s=50, label='Task Workspace (Reachable)')
if len(dexterous_workspace_pts) > 0:
    plt.scatter(dexterous_workspace_pts[:, 0], dexterous_workspace_pts[:, 1], c='red', s=50, label='Dexterous Workspace (Arbitrary Yaw)')

plt.title('2D Occupancy Map: Task vs Dexterous Workspace\n(HEAL Manipulator, Top-Down Table View)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Ensure directories exist
os.chdir('../../lab-03')
plt.savefig('workspace_task_vs_dexterous.png', dpi=300)
print("Saved occupancy map to workspace_task_vs_dexterous.png")

# --- 3D Point Cloud ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# We can plot the raw points directly to show 3D volume
if len(valid_positions) > 0:
    ax.scatter(valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2], 
               c='lightblue', s=1, alpha=0.1, label='Task Workspace')

# For dexterous points, we need to extract raw points that fall into dexterous voxels
dex_pts_3d = []
for i, pos in enumerate(valid_positions):
    ix = int((pos[0] - min_xy[0]) / grid_resolution)
    iy = int((pos[1] - min_xy[1]) / grid_resolution)
    if [min_xy[0] + (ix + 0.5) * grid_resolution, min_xy[1] + (iy + 0.5) * grid_resolution] in dexterous_workspace_pts.tolist():
        dex_pts_3d.append(pos)

dex_pts_3d = np.array(dex_pts_3d)
if len(dex_pts_3d) > 0:
    ax.scatter(dex_pts_3d[:, 0], dex_pts_3d[:, 1], dex_pts_3d[:, 2], 
               c='red', s=1, alpha=0.5, label='Dexterous Workspace')

ax.set_title("3D Point Cloud: Task vs Dexterous Workspace")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# Keep axes scaled equally
max_range = np.array([valid_positions[:, 0].max()-valid_positions[:, 0].min(), 
                      valid_positions[:, 1].max()-valid_positions[:, 1].min(), 
                      valid_positions[:, 2].max()-valid_positions[:, 2].min()]).max() / 2.0
mid_x = (valid_positions[:, 0].max()+valid_positions[:, 0].min()) * 0.5
mid_y = (valid_positions[:, 1].max()+valid_positions[:, 1].min()) * 0.5
mid_z = (valid_positions[:, 2].max()+valid_positions[:, 2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.legend()
plt.savefig('workspace_task_vs_dexterous_3d.png', dpi=300)
print("Saved 3D plot to workspace_task_vs_dexterous_3d.png")

"""
spawn_franka.py — Franka Emika Panda (7-DOF) Interactive MuJoCo Simulation
===========================================================================
Features:
  • Tkinter SLIDER GUI — one slider per joint + gripper
  • PD controller with gravity compensation
  • ANALYTICAL Forward Kinematics (from body chain transforms)
  • Comparison: Analytical FK vs MuJoCo Simulation coordinates
  • Position error (mm) and orientation error (deg) displayed live
  • Jacobian computation + end-effector velocity
  • Body/world frame axes visualization
  • Telemetry HUD overlay
  • Predefined poses + demo trajectory mode

Usage:
  cd ME-639/lab && source ../venv/bin/activate && python3 spawn_franka.py
"""

import mujoco
import mujoco.viewer
import time
import os
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk


# ═══════════════════════════════════════════════════════════════════
# 1. LOAD MODEL
# ═══════════════════════════════════════════════════════════════════
def setup_working_directory():
    cwd = os.getcwd()
    if os.path.basename(cwd) == "robot_descriptions":
        return
    for c in ["ITR_mujoco_fk_lab/robot_descriptions",
              "lab/ITR_mujoco_fk_lab/robot_descriptions"]:
        if os.path.exists(c):
            os.chdir(c)
            return
    raise FileNotFoundError("Could not find 'robot_descriptions'.")


setup_working_directory()
model = mujoco.MjModel.from_xml_path("franka_clean.xml")
data = mujoco.MjData(model)

# ═══════════════════════════════════════════════════════════════════
# 2. ROBOT STRUCTURE
# ═══════════════════════════════════════════════════════════════════
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
JOINT_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
EE_BODY_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
LINK_BODY_NAMES = [f"link{i}" for i in range(8)] + ["hand"]
LINK_BODY_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in LINK_BODY_NAMES]
NUM_JOINTS = 7
NUM_ACTUATORS = model.nu

JOINT_LIMITS = []
for jid in JOINT_IDS:
    lo = model.jnt_range[jid, 0] if model.jnt_limited[jid] else -3.14
    hi = model.jnt_range[jid, 1] if model.jnt_limited[jid] else 3.14
    JOINT_LIMITS.append((lo, hi))

JOINT_LABELS = ["J1 (Base)", "J2 (Shoulder)", "J3 (Elbow 1)", "J4 (Elbow 2)",
                "J5 (Wrist 1)", "J6 (Wrist 2)", "J7 (Wrist 3)"]

# ═══════════════════════════════════════════════════════════════════
# 3. ANALYTICAL FK — Transformation Chain from XML Body Hierarchy
# ═══════════════════════════════════════════════════════════════════
#
# MuJoCo body chain: T_world_body = T_parent * Trans(pos) * Rot(quat) * Rot_joint(q)
#
# Chain extracted directly from franka/mjx_panda.xml:
#   link0:  pos=[0,0,0]                quat=[1,0,0,0]                       (no joint)
#   link1:  pos=[0,0,0.333]            quat=[1,0,0,0]                       joint1 axis=[0,0,1]
#   link2:  pos=[0,0,0]                quat=[1,-1,0,0] → Rx(-90°)           joint2 axis=[0,0,1]
#   link3:  pos=[0,-0.316,0]           quat=[1,1,0,0]  → Rx(+90°)           joint3 axis=[0,0,1]
#   link4:  pos=[0.0825,0,0]           quat=[1,1,0,0]  → Rx(+90°)           joint4 axis=[0,0,1]
#   link5:  pos=[-0.0825,0.384,0]      quat=[1,-1,0,0] → Rx(-90°)           joint5 axis=[0,0,1]
#   link6:  pos=[0,0,0]                quat=[1,1,0,0]  → Rx(+90°)           joint6 axis=[0,0,1]
#   link7:  pos=[0.088,0,0]            quat=[1,1,0,0]  → Rx(+90°)           joint7 axis=[0,0,1]
#   hand:   pos=[0,0,0.107]            quat=[0.924,0,0,-0.383] → Rz(-45°)   (no joint)
#

def quat_to_rotmat(q):
    """Quaternion [w, x, y, z] → 3x3 rotation matrix."""
    q = np.array(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])


def rot_axis(axis, angle):
    """Rodrigues' formula: rotation matrix about an arbitrary axis by angle (rad)."""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def make_transform(pos, rot3x3):
    """Build a 4x4 homogeneous transformation from position + rotation."""
    T = np.eye(4)
    T[:3, :3] = rot3x3
    T[:3, 3] = pos
    return T


# Define the kinematic chain: list of (translation, rotation_quat, joint_axis_or_None)
# joint_axis = None means this is a fixed body (no joint variable consumed)
FRANKA_CHAIN = [
    # link0 (base, fixed)
    ([0.0, 0.0, 0.0],     [1, 0, 0, 0],                    None),
    # link1 → joint1
    ([0, 0, 0.333],        [1, 0, 0, 0],                    [0, 0, 1]),
    # link2 → joint2
    ([0, 0, 0],            [1, -1, 0, 0],                   [0, 0, 1]),
    # link3 → joint3
    ([0, -0.316, 0],       [1, 1, 0, 0],                    [0, 0, 1]),
    # link4 → joint4
    ([0.0825, 0, 0],       [1, 1, 0, 0],                    [0, 0, 1]),
    # link5 → joint5
    ([-0.0825, 0.384, 0],  [1, -1, 0, 0],                   [0, 0, 1]),
    # link6 → joint6
    ([0, 0, 0],            [1, 1, 0, 0],                    [0, 0, 1]),
    # link7 → joint7
    ([0.088, 0, 0],        [1, 1, 0, 0],                    [0, 0, 1]),
    # hand (fixed flange transform, no joint)
    ([0, 0, 0.107],        [0.9238795, 0, 0, -0.3826834],   None),
]


def analytical_fk_franka(q):
    """
    Compute 4x4 homogeneous transform of the Franka end-effector (hand body)
    using the kinematic chain extracted from the MJCF XML.

    Args:
        q: array of 7 joint angles [q1, ..., q7] in radians.

    Returns:
        T: 4x4 numpy array, homogeneous transform from world to hand.
    """
    T = np.eye(4)
    joint_idx = 0

    for (pos, quat, axis) in FRANKA_CHAIN:
        # Fixed body transform: translate then rotate
        R_body = quat_to_rotmat(quat)
        T_body = make_transform(pos, R_body)
        T = T @ T_body

        # Joint rotation (if this body has a joint)
        if axis is not None:
            R_joint = rot_axis(axis, q[joint_idx])
            T_joint = make_transform([0, 0, 0], R_joint)
            T = T @ T_joint
            joint_idx += 1

    return T


def compute_fk_error(T_analytical, sim_pos, sim_rotmat):
    """
    Compute position error (mm) and orientation error (deg)
    between analytical FK and MuJoCo simulation.
    """
    # Position error
    p_fk = T_analytical[:3, 3]
    pos_error_m = np.linalg.norm(sim_pos - p_fk)
    pos_error_mm = pos_error_m * 1000.0

    # Orientation error (axis-angle magnitude)
    R_fk = T_analytical[:3, :3]
    R_err = R_fk.T @ sim_rotmat
    trace_val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    rot_error_rad = np.arccos(trace_val)
    rot_error_deg = np.degrees(rot_error_rad)

    return p_fk, pos_error_mm, rot_error_deg


# ═══════════════════════════════════════════════════════════════════
# 4. POSES & CONTROLLER
# ═══════════════════════════════════════════════════════════════════
POSES = {
    "home":  np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
    "zero":  np.zeros(7),
    "ready": np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.0]),
}

KP = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
KD = np.array([50.0,  50.0,  50.0,  50.0,  20.0,  15.0,  5.0])
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.0

# ═══════════════════════════════════════════════════════════════════
# 5. SHARED STATE
# ═══════════════════════════════════════════════════════════════════
q_target = POSES["home"].copy()
gripper_target = GRIPPER_OPEN
follow_camera = False
demo_mode = False
gui_running = True
last_print_time = 0.0
PRINT_INTERVAL = 0.5

# Error state (updated in main loop, read by GUI)
fk_error_pos_mm = 0.0
fk_error_rot_deg = 0.0
fk_analytical_pos = np.zeros(3)


# ═══════════════════════════════════════════════════════════════════
# 6. VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════
BODY_AXIS_LEN, BODY_AXIS_WIDTH = 0.15, 0.005
WORLD_AXIS_LEN, WORLD_AXIS_WIDTH = 0.4, 0.003
BODY_COLORS = {"x": np.array([1,0,0,1.]), "y": np.array([0,1,0,1.]), "z": np.array([0,0,1,1.])}
WORLD_COLORS = {"x": np.array([.6,.2,.2,.5]), "y": np.array([.2,.6,.2,.5]), "z": np.array([.2,.2,.6,.5])}
AXES = [("x", np.array([1.,0.,0.])), ("y", np.array([0.,1.,0.])), ("z", np.array([0.,0.,1.]))]


def draw_arrow(scn, origin, direction, length, width, rgba):
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_ARROW, size=np.zeros(3),
                        pos=np.zeros(3), mat=np.eye(3).flatten(), rgba=rgba.astype(np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, origin, origin + direction * length)
    scn.ngeom += 1


def draw_label(scn, pos, text):
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([.001,0,0]),
                        pos=pos, mat=np.eye(3).flatten(), rgba=np.array([1,1,1,1], dtype=np.float32))
    g.label = text
    scn.ngeom += 1


def rotation_matrix_to_euler(R):
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


# ═══════════════════════════════════════════════════════════════════
# 7. TKINTER SLIDER GUI
# ═══════════════════════════════════════════════════════════════════
def run_slider_gui():
    global q_target, gripper_target, follow_camera, demo_mode, gui_running

    root = tk.Tk()
    root.title("Franka Panda — Joint Control + FK Error Analysis")
    root.geometry("520x850")
    root.configure(bg="#2b2b2b")
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root))

    style = ttk.Style(); style.theme_use("clam")
    style.configure("TScale", background="#2b2b2b")
    style.configure("TLabel", background="#2b2b2b", foreground="#e0e0e0", font=("Consolas", 10))
    style.configure("TButton", font=("Consolas", 10))
    style.configure("Header.TLabel", font=("Consolas", 13, "bold"), foreground="#4fc3f7")
    style.configure("Error.TLabel", background="#2b2b2b", foreground="#ff8a65", font=("Consolas", 10, "bold"))
    style.configure("Good.TLabel", background="#2b2b2b", foreground="#66bb6a", font=("Consolas", 10, "bold"))

    ttk.Label(root, text="Franka Panda (7-DOF)", style="Header.TLabel").pack(pady=6)

    slider_vars, angle_labels = [], []
    jf = ttk.Frame(root); jf.pack(fill="x", padx=10)
    for i in range(NUM_JOINTS):
        lo_d, hi_d = np.degrees(JOINT_LIMITS[i][0]), np.degrees(JOINT_LIMITS[i][1])
        f = ttk.Frame(jf); f.pack(fill="x", pady=2)
        ttk.Label(f, text=f"{JOINT_LABELS[i]}:", width=16, anchor="w").pack(side="left")
        v = tk.DoubleVar(value=np.degrees(q_target[i])); slider_vars.append(v)
        ttk.Scale(f, from_=lo_d, to=hi_d, orient="horizontal", variable=v, length=220).pack(side="left", padx=5)
        l = ttk.Label(f, text=f"{np.degrees(q_target[i]):+7.1f}°", width=9); l.pack(side="left"); angle_labels.append(l)

    gf = ttk.Frame(jf); gf.pack(fill="x", pady=2)
    ttk.Label(gf, text="Gripper:", width=16, anchor="w").pack(side="left")
    grip_var = tk.DoubleVar(value=GRIPPER_OPEN * 1000)
    ttk.Scale(gf, from_=0, to=40, orient="horizontal", variable=grip_var, length=220).pack(side="left", padx=5)
    grip_label = ttk.Label(gf, text=f"{GRIPPER_OPEN*1000:.0f} mm", width=9); grip_label.pack(side="left")

    # --- FK & Error display ---
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=10, pady=6)
    ttk.Label(root, text="FK Comparison: Analytical vs Simulation", style="Header.TLabel").pack()

    sim_pos_var = tk.StringVar(value="Sim EE:  (---, ---, ---)")
    fk_pos_var = tk.StringVar(value="FK  EE:  (---, ---, ---)")
    fk_rpy_var = tk.StringVar(value="EE RPY:  (---, ---, ---)")
    pos_err_var = tk.StringVar(value="Pos Error: --- mm")
    rot_err_var = tk.StringVar(value="Rot Error: --- deg")
    vel_var = tk.StringVar(value="EE |v|:  --- m/s")

    ttk.Label(root, textvariable=sim_pos_var).pack(anchor="w", padx=15)
    ttk.Label(root, textvariable=fk_pos_var).pack(anchor="w", padx=15)
    ttk.Label(root, textvariable=fk_rpy_var).pack(anchor="w", padx=15)
    ttk.Label(root, textvariable=vel_var).pack(anchor="w", padx=15)

    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=10, pady=4)
    pos_err_label = ttk.Label(root, textvariable=pos_err_var, style="Good.TLabel")
    pos_err_label.pack(anchor="w", padx=15)
    rot_err_label = ttk.Label(root, textvariable=rot_err_var, style="Good.TLabel")
    rot_err_label.pack(anchor="w", padx=15)

    # --- Buttons ---
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=10, pady=6)
    bf = ttk.Frame(root); bf.pack(fill="x", padx=10)
    def set_pose(n):
        global demo_mode; demo_mode = False
        for i in range(NUM_JOINTS): slider_vars[i].set(np.degrees(POSES[n][i]))
    ttk.Button(bf, text="🏠 Home", command=lambda: set_pose("home")).pack(side="left", padx=4, expand=True, fill="x")
    ttk.Button(bf, text="0️⃣ Zero", command=lambda: set_pose("zero")).pack(side="left", padx=4, expand=True, fill="x")
    ttk.Button(bf, text="✋ Ready", command=lambda: set_pose("ready")).pack(side="left", padx=4, expand=True, fill="x")

    bf2 = ttk.Frame(root); bf2.pack(fill="x", padx=10, pady=5)
    demo_var, follow_var = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
    def toggle_demo():
        global demo_mode; demo_mode = demo_var.get()
    def toggle_follow():
        global follow_camera; follow_camera = follow_var.get()
    ttk.Checkbutton(bf2, text="Demo Trajectory", variable=demo_var, command=toggle_demo).pack(side="left", padx=10)
    ttk.Checkbutton(bf2, text="Follow Camera", variable=follow_var, command=toggle_follow).pack(side="left", padx=10)

    def update():
        global q_target, gripper_target
        if not gui_running: root.destroy(); return
        if not demo_mode:
            for i in range(NUM_JOINTS): q_target[i] = np.radians(slider_vars[i].get())
        else:
            for i in range(NUM_JOINTS): slider_vars[i].set(np.degrees(q_target[i]))
        gripper_target = grip_var.get() / 1000.0
        for i in range(NUM_JOINTS): angle_labels[i].config(text=f"{slider_vars[i].get():+7.1f}°")
        grip_label.config(text=f"{grip_var.get():.0f} mm")
        try:
            ee_pos = data.xpos[EE_BODY_ID]
            ee_rot = data.xmat[EE_BODY_ID].reshape(3, 3)
            r, p, y = rotation_matrix_to_euler(ee_rot)
            sim_pos_var.set(f"Sim EE:  ({ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f})")
            fk_pos_var.set(f"FK  EE:  ({fk_analytical_pos[0]:+.4f}, {fk_analytical_pos[1]:+.4f}, {fk_analytical_pos[2]:+.4f})")
            fk_rpy_var.set(f"EE RPY:  ({np.degrees(r):+.1f}°, {np.degrees(p):+.1f}°, {np.degrees(y):+.1f}°)")
            jacp = np.zeros((3, model.nv)); mujoco.mj_jacBody(model, data, jacp, None, EE_BODY_ID)
            vel_var.set(f"EE |v|:  {np.linalg.norm(jacp @ data.qvel):.4f} m/s")

            pos_err_var.set(f"Position Error:    {fk_error_pos_mm:.4f} mm")
            rot_err_var.set(f"Orientation Error:  {fk_error_rot_deg:.4f} deg")
            s = "Good.TLabel" if fk_error_pos_mm < 0.1 else "Error.TLabel"
            pos_err_label.configure(style=s)
            rot_err_label.configure(style="Good.TLabel" if fk_error_rot_deg < 0.1 else "Error.TLabel")
        except Exception:
            pass
        root.after(33, update)

    def on_close(r):
        global gui_running; gui_running = False

    root.after(100, update); root.mainloop()


# ═══════════════════════════════════════════════════════════════════
# 8. LAUNCH GUI + MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════════
threading.Thread(target=run_slider_gui, daemon=True).start()
print("\nSlider GUI launched! Drag sliders to control joints.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and gui_running:
        step_start = time.time()
        t = data.time

        if demo_mode:
            q_target[0] = POSES["home"][0] + 0.8 * np.sin(0.5 * t)
            q_target[1] = POSES["home"][1] + 0.3 * np.sin(0.4 * t + 1.0)
            q_target[3] = POSES["home"][3] + 0.4 * np.sin(0.6 * t + 0.5)
            q_target[5] = POSES["home"][5] + 0.5 * np.sin(0.7 * t + 2.0)

        # PD + gravity comp
        gc = data.qfrc_bias[:NUM_JOINTS].copy()
        for i in range(NUM_JOINTS):
            data.ctrl[i] = KP[i] * (q_target[i] - data.qpos[i]) + KD[i] * (-data.qvel[i]) + gc[i]
        if NUM_ACTUATORS > NUM_JOINTS:
            data.ctrl[NUM_JOINTS] = gripper_target

        mujoco.mj_step(model, data)

        # --- Simulation FK (ground truth) ---
        ee_pos_sim = data.xpos[EE_BODY_ID].copy()
        ee_rot_sim = data.xmat[EE_BODY_ID].reshape(3, 3).copy()
        roll, pitch, yaw = rotation_matrix_to_euler(ee_rot_sim)

        # --- Analytical FK ---
        q_current = data.qpos[:NUM_JOINTS].copy()
        T_fk = analytical_fk_franka(q_current)
        fk_analytical_pos[:] = T_fk[:3, 3]
        fk_p, fk_error_pos_mm, fk_error_rot_deg = compute_fk_error(T_fk, ee_pos_sim, ee_rot_sim)

        # Jacobian
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, EE_BODY_ID)
        ee_vel = jacp @ data.qvel

        # --- Visualization ---
        viewer.user_scn.ngeom = 0
        for an, av in AXES:
            draw_arrow(viewer.user_scn, ee_pos_sim, ee_rot_sim @ av, BODY_AXIS_LEN, BODY_AXIS_WIDTH, BODY_COLORS[an])
            draw_arrow(viewer.user_scn, np.zeros(3), av, WORLD_AXIS_LEN, WORLD_AXIS_WIDTH, WORLD_COLORS[an])
        for bid in LINK_BODY_IDS:
            if bid < 0: continue
            lp, lr = data.xpos[bid], data.xmat[bid].reshape(3, 3)
            for an, av in AXES:
                draw_arrow(viewer.user_scn, lp, lr @ av, 0.06, 0.002, BODY_COLORS[an] * 0.6)

        # Analytical FK marker (yellow sphere at computed position)
        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.012, 0, 0]),
                                pos=fk_analytical_pos, mat=np.eye(3).flatten(),
                                rgba=np.array([1, 1, 0, 0.8], dtype=np.float32))
            viewer.user_scn.ngeom += 1

        # HUD
        q_deg = np.degrees(q_current)
        hud = ee_pos_sim + np.array([0, 0, 0.50])
        lines = [
            f"Sim EE: ({ee_pos_sim[0]:+.4f}, {ee_pos_sim[1]:+.4f}, {ee_pos_sim[2]:+.4f})",
            f"FK  EE: ({fk_analytical_pos[0]:+.4f}, {fk_analytical_pos[1]:+.4f}, {fk_analytical_pos[2]:+.4f})",
            f"Pos Err: {fk_error_pos_mm:.4f} mm | Rot Err: {fk_error_rot_deg:.4f} deg",
            f"RPY: ({np.degrees(roll):+.1f}, {np.degrees(pitch):+.1f}, {np.degrees(yaw):+.1f}) deg",
            f"Mode: {'DEMO' if demo_mode else 'SLIDER'}",
        ]
        for i, txt in enumerate(lines):
            draw_label(viewer.user_scn, hud + np.array([0, 0, 0.08 * (len(lines)-1-i)]), txt)

        if follow_camera:
            viewer.cam.lookat[:] = ee_pos_sim
        viewer.sync()

        # Terminal
        now = time.time()
        if now - last_print_time > PRINT_INTERVAL:
            print(
                f"t={t:5.2f} | "
                f"Sim=({ee_pos_sim[0]:+.4f},{ee_pos_sim[1]:+.4f},{ee_pos_sim[2]:+.4f}) | "
                f"FK=({fk_analytical_pos[0]:+.4f},{fk_analytical_pos[1]:+.4f},{fk_analytical_pos[2]:+.4f}) | "
                f"PosErr={fk_error_pos_mm:.4f}mm RotErr={fk_error_rot_deg:.4f}° | "
                f"q={np.round(q_deg,1)}"
            )
            last_print_time = now

        sl = model.opt.timestep - (time.time() - step_start)
        if sl > 0: time.sleep(sl)

gui_running = False
print("Simulation ended.")

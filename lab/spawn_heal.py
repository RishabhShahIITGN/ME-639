"""
spawn_heal.py — Addverb HEAL (6-DOF) Interactive MuJoCo Simulation
====================================================================
Features:
  • Tkinter SLIDER GUI — one slider per joint
  • PD controller with gravity compensation
  • ANALYTICAL Forward Kinematics (from body chain transforms)
  • Comparison: Analytical FK vs MuJoCo Simulation coordinates
  • Position error (mm) and orientation error (deg) displayed live
  • Jacobian computation + end-effector velocity
  • Body/world frame axes visualization
  • Telemetry HUD overlay
  • Predefined poses + demo trajectory mode

Usage:
  cd ME-639/lab && source ../venv/bin/activate && python3 spawn_heal.py
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
model = mujoco.MjModel.from_xml_path("single_arm_heal_effort_actuation_rs.xml")
data = mujoco.MjData(model)

# ═══════════════════════════════════════════════════════════════════
# 2. ROBOT STRUCTURE
# ═══════════════════════════════════════════════════════════════════
JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]
JOINT_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
EE_BODY_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
EE_SITE_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_center")
LINK_BODY_NAMES = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "end_effector"]
LINK_BODY_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in LINK_BODY_NAMES]
NUM_JOINTS = 6
NUM_ACTUATORS = model.nu

JOINT_LIMITS = []
for jid in JOINT_IDS:
    lo = model.jnt_range[jid, 0] if model.jnt_limited[jid] else -3.14
    hi = model.jnt_range[jid, 1] if model.jnt_limited[jid] else 3.14
    JOINT_LIMITS.append((lo, hi))

JOINT_LABELS = ["J1 (Turret)", "J2 (Shoulder)", "J3 (Elbow)",
                "J4 (Wrist 1)", "J5 (Wrist 2)", "J6 (Wrist 3)"]

# ═══════════════════════════════════════════════════════════════════
# 3. ANALYTICAL FK — Transformation Chain from XML Body Hierarchy
# ═══════════════════════════════════════════════════════════════════
#
# MuJoCo body chain: T_world_body = T_parent * Trans(pos) * Rot(orient) * Rot_joint(q)
#
# Chain extracted from single_arm_heal_effort_actuation_rs.xml:
#   base_link:     pos=[0,0,0]              euler=[0,0,0]                         (no joint)
#   link_1:        pos=[0,0,0.171]          (identity orientation)                joint_1 axis=[0,0,1]
#   link_2:        pos=[0,0.0875,0.1498]    quat=[0.707105,0.707108,0,0] → Rx(90°) joint_2 axis=[0,0,-1]
#   link_3:        pos=[0,0.3,0]            euler=[0,0,-1.57] → Rz(-90°)         joint_3 axis=[0,0,1]
#   link_4:        pos=[0,0.1593,0.0875]    euler=[-1.57,0,0] → Rx(-90°)         joint_4 axis=[0,0,1]
#   link_5:        pos=[0,0.03185,0.16105]  euler=[0.50951,0,1.57]               joint_5 axis=[0,0,1]
#   end_effector:  pos=[0,-0.1227,0.0654]   quat=[0.707105,0.707108,0,0] → Rx(90°) joint_6 axis=[0,0,-1]
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


def euler_xyz_to_rotmat(e):
    """MuJoCo intrinsic xyz euler → 3x3 rotation matrix: R = Rx(e0) @ Ry(e1) @ Rz(e2)."""
    cx, sx = np.cos(e[0]), np.sin(e[0])
    cy, sy = np.cos(e[1]), np.sin(e[1])
    cz, sz = np.cos(e[2]), np.sin(e[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def rot_axis(axis, angle):
    """Rodrigues' formula: rotation matrix about axis by angle (rad)."""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def make_transform(pos, rot3x3):
    T = np.eye(4)
    T[:3, :3] = rot3x3
    T[:3, 3] = pos
    return T


# Chain definition: (translation, rotation_3x3, joint_axis_or_None)
# Rotation is precomputed from quat or euler as specified in XML.
# joint_axis = None for fixed bodies.
HEAL_CHAIN = [
    # base_link (fixed, at origin)
    ([0, 0, 0],              np.eye(3),                                        None),
    # link_1 → joint_1, axis=[0,0,1]
    ([0, 0, 0.171],          np.eye(3),                                        [0, 0, 1]),
    # link_2 → joint_2, axis=[0,0,-1]
    ([0, 0.0875, 0.1498],    quat_to_rotmat([0.707105, 0.707108, 0, 0]),       [0, 0, -1]),
    # link_3 → joint_3, axis=[0,0,1]
    ([0, 0.3, 0],            euler_xyz_to_rotmat([0, 0, -1.57]),               [0, 0, 1]),
    # link_4 → joint_4, axis=[0,0,1]
    ([0, 0.1593, 0.0875],    euler_xyz_to_rotmat([-1.57, 0, 0]),               [0, 0, 1]),
    # link_5 → joint_5, axis=[0,0,1]
    ([0, 0.03185, 0.16105],  euler_xyz_to_rotmat([0.50951, 0, 1.57]),          [0, 0, 1]),
    # end_effector → joint_6, axis=[0,0,-1]
    ([0, -0.1227, 0.0654],   quat_to_rotmat([0.707105, 0.707108, 0, 0]),      [0, 0, -1]),
]


def analytical_fk_heal(q):
    """
    Compute 4x4 homogeneous transform of the HEAL end-effector
    using the kinematic chain extracted from the MJCF XML.

    Args:
        q: array of 6 joint angles [q1, ..., q6] in radians.

    Returns:
        T: 4x4 numpy array, homogeneous transform from world to end_effector.
    """
    T = np.eye(4)
    joint_idx = 0

    for (pos, R_body, axis) in HEAL_CHAIN:
        # Fixed body transform
        T_body = make_transform(pos, R_body)
        T = T @ T_body

        # Joint rotation
        if axis is not None:
            R_joint = rot_axis(axis, q[joint_idx])
            T = T @ make_transform([0, 0, 0], R_joint)
            joint_idx += 1

    return T


def compute_fk_error(T_analytical, sim_pos, sim_rotmat):
    """Position error (mm) and orientation error (deg)."""
    p_fk = T_analytical[:3, 3]
    pos_error_mm = np.linalg.norm(sim_pos - p_fk) * 1000.0

    R_fk = T_analytical[:3, :3]
    R_err = R_fk.T @ sim_rotmat
    trace_val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    rot_error_deg = np.degrees(np.arccos(trace_val))

    return p_fk, pos_error_mm, rot_error_deg


# ═══════════════════════════════════════════════════════════════════
# 4. POSES & CONTROLLER
# ═══════════════════════════════════════════════════════════════════
POSES = {
    "home":    np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0]),
    "zero":    np.zeros(6),
    "stretch": np.array([0.0, 0.0, 0.0, 0.0, -1.57, 0.0]),
}

KP = np.array([400.0, 400.0, 300.0, 150.0, 100.0, 50.0])
KD = np.array([40.0,  40.0,  30.0,  15.0,  10.0,  5.0])

# ═══════════════════════════════════════════════════════════════════
# 5. SHARED STATE
# ═══════════════════════════════════════════════════════════════════
q_target = POSES["home"].copy()
follow_camera = False
demo_mode = False
gui_running = True
last_print_time = 0.0
PRINT_INTERVAL = 0.5

fk_error_pos_mm = 0.0
fk_error_rot_deg = 0.0
fk_analytical_pos = np.zeros(3)


# ═══════════════════════════════════════════════════════════════════
# 6. VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════
BODY_AXIS_LEN, BODY_AXIS_WIDTH = 0.12, 0.004
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
    global q_target, follow_camera, demo_mode, gui_running

    root = tk.Tk()
    root.title("HEAL — Joint Control + FK Error Analysis")
    root.geometry("520x800")
    root.configure(bg="#2b2b2b")
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root))

    style = ttk.Style(); style.theme_use("clam")
    style.configure("TScale", background="#2b2b2b")
    style.configure("TLabel", background="#2b2b2b", foreground="#e0e0e0", font=("Consolas", 10))
    style.configure("TButton", font=("Consolas", 10))
    style.configure("Header.TLabel", font=("Consolas", 13, "bold"), foreground="#66bb6a")
    style.configure("Error.TLabel", background="#2b2b2b", foreground="#ff8a65", font=("Consolas", 10, "bold"))
    style.configure("Good.TLabel", background="#2b2b2b", foreground="#66bb6a", font=("Consolas", 10, "bold"))

    ttk.Label(root, text="HEAL Robot (6-DOF)", style="Header.TLabel").pack(pady=6)

    slider_vars, angle_labels = [], []
    jf = ttk.Frame(root); jf.pack(fill="x", padx=10)
    for i in range(NUM_JOINTS):
        lo_d, hi_d = np.degrees(JOINT_LIMITS[i][0]), np.degrees(JOINT_LIMITS[i][1])
        f = ttk.Frame(jf); f.pack(fill="x", pady=2)
        ttk.Label(f, text=f"{JOINT_LABELS[i]}:", width=16, anchor="w").pack(side="left")
        v = tk.DoubleVar(value=np.degrees(q_target[i])); slider_vars.append(v)
        ttk.Scale(f, from_=lo_d, to=hi_d, orient="horizontal", variable=v, length=220).pack(side="left", padx=5)
        l = ttk.Label(f, text=f"{np.degrees(q_target[i]):+7.1f}°", width=9); l.pack(side="left"); angle_labels.append(l)

    # --- FK & Error display ---
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=10, pady=6)
    ttk.Label(root, text="FK Comparison: Analytical vs Simulation", style="Header.TLabel").pack()

    sim_pos_var = tk.StringVar(value="Sim EE:  (---, ---, ---)")
    fk_pos_var = tk.StringVar(value="FK  EE:  (---, ---, ---)")
    fk_rpy_var = tk.StringVar(value="EE RPY:  (---, ---, ---)")
    pos_err_var = tk.StringVar(value="Pos Error: --- mm")
    rot_err_var = tk.StringVar(value="Rot Error: --- deg")
    vel_var = tk.StringVar(value="EE |v|:  --- m/s")
    tcp_var = tk.StringVar(value="TCP Pos: (---, ---, ---)")

    ttk.Label(root, textvariable=sim_pos_var).pack(anchor="w", padx=15)
    ttk.Label(root, textvariable=fk_pos_var).pack(anchor="w", padx=15)
    ttk.Label(root, textvariable=fk_rpy_var).pack(anchor="w", padx=15)
    ttk.Label(root, textvariable=vel_var).pack(anchor="w", padx=15)
    ttk.Label(root, textvariable=tcp_var).pack(anchor="w", padx=15)

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
    ttk.Button(bf, text="💪 Stretch", command=lambda: set_pose("stretch")).pack(side="left", padx=4, expand=True, fill="x")

    bf2 = ttk.Frame(root); bf2.pack(fill="x", padx=10, pady=5)
    demo_var, follow_var = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
    def toggle_demo():
        global demo_mode; demo_mode = demo_var.get()
    def toggle_follow():
        global follow_camera; follow_camera = follow_var.get()
    ttk.Checkbutton(bf2, text="Demo Trajectory", variable=demo_var, command=toggle_demo).pack(side="left", padx=10)
    ttk.Checkbutton(bf2, text="Follow Camera", variable=follow_var, command=toggle_follow).pack(side="left", padx=10)

    def update():
        global q_target
        if not gui_running: root.destroy(); return
        if not demo_mode:
            for i in range(NUM_JOINTS): q_target[i] = np.radians(slider_vars[i].get())
        else:
            for i in range(NUM_JOINTS): slider_vars[i].set(np.degrees(q_target[i]))
        for i in range(NUM_JOINTS): angle_labels[i].config(text=f"{slider_vars[i].get():+7.1f}°")
        try:
            ee_pos = data.xpos[EE_BODY_ID]
            ee_rot = data.xmat[EE_BODY_ID].reshape(3, 3)
            r, p, y = rotation_matrix_to_euler(ee_rot)
            sim_pos_var.set(f"Sim EE:  ({ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f})")
            fk_pos_var.set(f"FK  EE:  ({fk_analytical_pos[0]:+.4f}, {fk_analytical_pos[1]:+.4f}, {fk_analytical_pos[2]:+.4f})")
            fk_rpy_var.set(f"EE RPY:  ({np.degrees(r):+.1f}°, {np.degrees(p):+.1f}°, {np.degrees(y):+.1f}°)")
            if EE_SITE_ID >= 0:
                tcp = data.site_xpos[EE_SITE_ID]
                tcp_var.set(f"TCP Pos: ({tcp[0]:+.4f}, {tcp[1]:+.4f}, {tcp[2]:+.4f})")
            jacp = np.zeros((3, model.nv)); mujoco.mj_jacBody(model, data, jacp, None, EE_BODY_ID)
            vel_var.set(f"EE |v|:  {np.linalg.norm(jacp @ data.qvel):.4f} m/s")
            pos_err_var.set(f"Position Error:    {fk_error_pos_mm:.4f} mm")
            rot_err_var.set(f"Orientation Error:  {fk_error_rot_deg:.4f} deg")
            pos_err_label.configure(style="Good.TLabel" if fk_error_pos_mm < 0.1 else "Error.TLabel")
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
            q_target[1] = POSES["home"][1] + 0.4 * np.sin(0.4 * t + 1.0)
            q_target[2] = POSES["home"][2] + 0.3 * np.sin(0.6 * t + 0.5)
            q_target[3] = POSES["home"][3] + 0.5 * np.sin(0.7 * t + 2.0)
            q_target[4] = POSES["home"][4] + 0.3 * np.sin(0.8 * t + 1.5)
            q_target[5] = POSES["home"][5] + 0.6 * np.sin(0.3 * t + 3.0)

        # PD + gravity comp
        gc = data.qfrc_bias[:NUM_JOINTS].copy()
        for i in range(NUM_JOINTS):
            data.ctrl[i] = KP[i] * (q_target[i] - data.qpos[i]) + KD[i] * (-data.qvel[i]) + gc[i]

        mujoco.mj_step(model, data)

        # --- Simulation FK (ground truth) ---
        ee_pos_sim = data.xpos[EE_BODY_ID].copy()
        ee_rot_sim = data.xmat[EE_BODY_ID].reshape(3, 3).copy()
        roll, pitch, yaw = rotation_matrix_to_euler(ee_rot_sim)

        # --- Analytical FK ---
        q_current = data.qpos[:NUM_JOINTS].copy()
        T_fk = analytical_fk_heal(q_current)
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
                draw_arrow(viewer.user_scn, lp, lr @ av, 0.05, 0.002, BODY_COLORS[an] * 0.6)

        # Analytical FK marker (yellow sphere)
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

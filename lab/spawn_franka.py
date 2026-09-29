"""
spawn_franka.py — Franka Emika Panda (7-DOF) Interactive MuJoCo Simulation
===========================================================================
Features:
  • Tkinter SLIDER GUI — one slider per joint + gripper, updates in real time
  • PD joint-space controller with gravity compensation
  • Forward Kinematics: live end-effector position + orientation
  • Jacobian computation + end-effector velocity
  • Body/world frame axes visualization (arrows drawn in viewer)
  • Per-link coordinate frame visualization
  • Telemetry HUD overlay (position, orientation, joint angles)
  • Predefined poses (home, zero, ready) via GUI buttons
  • Follow-camera mode
  • Sinusoidal demo trajectory mode (toggle on/off)

Usage:
  cd ME-639/lab
  source ../venv/bin/activate
  python3 spawn_franka.py
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
    """Ensure we are in robot_descriptions/ so MuJoCo resolves relative mesh paths."""
    cwd = os.getcwd()
    if os.path.basename(cwd) == "robot_descriptions":
        return
    candidates = [
        "ITR_mujoco_fk_lab/robot_descriptions",
        "lab/ITR_mujoco_fk_lab/robot_descriptions",
    ]
    for c in candidates:
        if os.path.exists(c):
            os.chdir(c)
            return
    raise FileNotFoundError(
        "Could not find 'robot_descriptions'. Run from lab/ or ME-639/."
    )


setup_working_directory()

MODEL_PATH = "franka_clean.xml"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find '{MODEL_PATH}'.")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ═══════════════════════════════════════════════════════════════════
# 2. IDENTIFY ROBOT STRUCTURE
# ═══════════════════════════════════════════════════════════════════
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
JOINT_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]

EE_BODY_NAME = "hand"
EE_BODY_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
LINK_BODY_NAMES = [f"link{i}" for i in range(8)] + ["hand"]
LINK_BODY_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in LINK_BODY_NAMES]

NUM_JOINTS = 7
NUM_ACTUATORS = model.nu  # 7 motors + 1 gripper = 8

# Joint limits (from the model)
JOINT_LIMITS = []
for jid in JOINT_IDS:
    if model.jnt_limited[jid]:
        lo = model.jnt_range[jid, 0]
        hi = model.jnt_range[jid, 1]
    else:
        lo, hi = -3.14, 3.14
    JOINT_LIMITS.append((lo, hi))

JOINT_LABELS = [
    "J1 (Base)",
    "J2 (Shoulder)",
    "J3 (Elbow 1)",
    "J4 (Elbow 2)",
    "J5 (Wrist 1)",
    "J6 (Wrist 2)",
    "J7 (Wrist 3)",
]

print(f"Franka Panda loaded: {NUM_JOINTS} arm joints, {NUM_ACTUATORS} actuators")
print(f"End-effector body: '{EE_BODY_NAME}' (id={EE_BODY_ID})")

# ═══════════════════════════════════════════════════════════════════
# 3. PREDEFINED POSES
# ═══════════════════════════════════════════════════════════════════
POSES = {
    "home":  np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
    "zero":  np.zeros(7),
    "ready": np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.0]),
}

# ═══════════════════════════════════════════════════════════════════
# 4. CONTROLLER PARAMETERS
# ═══════════════════════════════════════════════════════════════════
KP = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
KD = np.array([50.0,  50.0,  50.0,  50.0,  20.0,  15.0,  5.0])
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.0

# ═══════════════════════════════════════════════════════════════════
# 5. SHARED STATE (thread-safe via simple reads/writes on floats)
# ═══════════════════════════════════════════════════════════════════
q_target = POSES["home"].copy()
gripper_target = GRIPPER_OPEN
follow_camera = False
demo_mode = False
gui_running = True
last_print_time = 0.0
PRINT_INTERVAL = 0.5


# ═══════════════════════════════════════════════════════════════════
# 6. TKINTER SLIDER GUI (runs in a separate thread)
# ═══════════════════════════════════════════════════════════════════
def run_slider_gui():
    global q_target, gripper_target, follow_camera, demo_mode, gui_running

    root = tk.Tk()
    root.title("Franka Panda — Joint Control Panel")
    root.geometry("480x750")
    root.configure(bg="#2b2b2b")
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root))

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TScale", background="#2b2b2b")
    style.configure("TLabel", background="#2b2b2b", foreground="#e0e0e0",
                    font=("Consolas", 10))
    style.configure("TButton", font=("Consolas", 10))
    style.configure("Header.TLabel", font=("Consolas", 14, "bold"),
                    foreground="#4fc3f7")

    ttk.Label(root, text="Franka Panda (7-DOF)", style="Header.TLabel").pack(pady=8)

    # --- Joint sliders ---
    slider_vars = []
    slider_widgets = []
    angle_labels = []

    joint_frame = ttk.Frame(root)
    joint_frame.pack(fill="x", padx=10)

    for i in range(NUM_JOINTS):
        lo_deg = np.degrees(JOINT_LIMITS[i][0])
        hi_deg = np.degrees(JOINT_LIMITS[i][1])
        cur_deg = np.degrees(q_target[i])

        frame = ttk.Frame(joint_frame)
        frame.pack(fill="x", pady=2)

        ttk.Label(frame, text=f"{JOINT_LABELS[i]}:", width=16, anchor="w").pack(side="left")

        var = tk.DoubleVar(value=cur_deg)
        slider_vars.append(var)

        s = ttk.Scale(frame, from_=lo_deg, to=hi_deg, orient="horizontal",
                      variable=var, length=220)
        s.pack(side="left", padx=5)
        slider_widgets.append(s)

        lbl = ttk.Label(frame, text=f"{cur_deg:+7.1f}°", width=9)
        lbl.pack(side="left")
        angle_labels.append(lbl)

    # --- Gripper slider ---
    grip_frame = ttk.Frame(joint_frame)
    grip_frame.pack(fill="x", pady=2)
    ttk.Label(grip_frame, text="Gripper:", width=16, anchor="w").pack(side="left")

    grip_var = tk.DoubleVar(value=GRIPPER_OPEN * 1000)
    grip_slider = ttk.Scale(grip_frame, from_=0, to=40, orient="horizontal",
                            variable=grip_var, length=220)
    grip_slider.pack(side="left", padx=5)
    grip_label = ttk.Label(grip_frame, text=f"{GRIPPER_OPEN*1000:.0f} mm", width=9)
    grip_label.pack(side="left")

    # --- FK display ---
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=10, pady=8)
    ttk.Label(root, text="Forward Kinematics", style="Header.TLabel").pack()

    fk_pos_var = tk.StringVar(value="EE Pos: (---, ---, ---)")
    fk_rpy_var = tk.StringVar(value="EE RPY: (---, ---, ---)")
    fk_vel_var = tk.StringVar(value="EE |v|: --- m/s")

    ttk.Label(root, textvariable=fk_pos_var).pack(anchor="w", padx=15)
    ttk.Label(root, textvariable=fk_rpy_var).pack(anchor="w", padx=15)
    ttk.Label(root, textvariable=fk_vel_var).pack(anchor="w", padx=15)

    # --- Buttons ---
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=10, pady=8)
    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", padx=10)

    def set_pose(name):
        nonlocal slider_vars, grip_var
        global demo_mode
        demo_mode = False
        pose = POSES[name]
        for i in range(NUM_JOINTS):
            slider_vars[i].set(np.degrees(pose[i]))

    ttk.Button(btn_frame, text="🏠 Home", command=lambda: set_pose("home")).pack(side="left", padx=4, expand=True, fill="x")
    ttk.Button(btn_frame, text="0️⃣ Zero", command=lambda: set_pose("zero")).pack(side="left", padx=4, expand=True, fill="x")
    ttk.Button(btn_frame, text="✋ Ready", command=lambda: set_pose("ready")).pack(side="left", padx=4, expand=True, fill="x")

    btn_frame2 = ttk.Frame(root)
    btn_frame2.pack(fill="x", padx=10, pady=5)

    demo_var = tk.BooleanVar(value=False)
    follow_var = tk.BooleanVar(value=False)

    def toggle_demo():
        global demo_mode
        demo_mode = demo_var.get()

    def toggle_follow():
        global follow_camera
        follow_camera = follow_var.get()

    ttk.Checkbutton(btn_frame2, text="Demo Trajectory", variable=demo_var,
                    command=toggle_demo).pack(side="left", padx=10)
    ttk.Checkbutton(btn_frame2, text="Follow Camera", variable=follow_var,
                    command=toggle_follow).pack(side="left", padx=10)

    # --- Periodic update loop ---
    def update():
        global q_target, gripper_target
        if not gui_running:
            root.destroy()
            return

        # Read slider values → update targets
        if not demo_mode:
            for i in range(NUM_JOINTS):
                q_target[i] = np.radians(slider_vars[i].get())
        else:
            # In demo mode, update sliders from q_target
            for i in range(NUM_JOINTS):
                slider_vars[i].set(np.degrees(q_target[i]))

        gripper_target = grip_var.get() / 1000.0

        # Update angle labels
        for i in range(NUM_JOINTS):
            angle_labels[i].config(text=f"{slider_vars[i].get():+7.1f}°")
        grip_label.config(text=f"{grip_var.get():.0f} mm")

        # Update FK display
        try:
            ee_pos = data.xpos[EE_BODY_ID]
            ee_rot = data.xmat[EE_BODY_ID].reshape(3, 3)
            roll, pitch, yaw = rotation_matrix_to_euler(ee_rot)
            fk_pos_var.set(f"EE Pos: ({ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f})")
            fk_rpy_var.set(f"EE RPY: ({np.degrees(roll):+.1f}°, {np.degrees(pitch):+.1f}°, {np.degrees(yaw):+.1f}°)")

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, EE_BODY_ID)
            ee_vel = jacp @ data.qvel
            fk_vel_var.set(f"EE |v|: {np.linalg.norm(ee_vel):.4f} m/s")
        except Exception:
            pass

        root.after(33, update)  # ~30 Hz GUI refresh

    def on_close(r):
        global gui_running
        gui_running = False

    root.after(100, update)
    root.mainloop()


# ═══════════════════════════════════════════════════════════════════
# 7. VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════
BODY_AXIS_LEN = 0.15
BODY_AXIS_WIDTH = 0.005
BODY_COLORS = {
    "x": np.array([1.0, 0.0, 0.0, 1.0]),
    "y": np.array([0.0, 1.0, 0.0, 1.0]),
    "z": np.array([0.0, 0.0, 1.0, 1.0]),
}
WORLD_AXIS_LEN = 0.4
WORLD_AXIS_WIDTH = 0.003
WORLD_COLORS = {
    "x": np.array([0.6, 0.2, 0.2, 0.5]),
    "y": np.array([0.2, 0.6, 0.2, 0.5]),
    "z": np.array([0.2, 0.2, 0.6, 0.5]),
}
AXES = [("x", np.array([1.0, 0.0, 0.0])),
        ("y", np.array([0.0, 1.0, 0.0])),
        ("z", np.array([0.0, 0.0, 1.0]))]


def draw_arrow(scn, origin, direction, length, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    end = origin + direction * length
    mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=np.zeros(3), pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=rgba.astype(np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, origin, end)
    scn.ngeom += 1


def draw_label(scn, pos, text):
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.001, 0.0, 0.0]),
                        pos=pos, mat=np.eye(3).flatten(),
                        rgba=np.array([1, 1, 1, 1], dtype=np.float32))
    geom.label = text
    scn.ngeom += 1


def rotation_matrix_to_euler(R):
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


# ═══════════════════════════════════════════════════════════════════
# 8. LAUNCH GUI IN SEPARATE THREAD
# ═══════════════════════════════════════════════════════════════════
gui_thread = threading.Thread(target=run_slider_gui, daemon=True)
gui_thread.start()

# ═══════════════════════════════════════════════════════════════════
# 9. MAIN SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════
print("\nSlider GUI launched in a separate window!")
print("Drag sliders to control joints. Use buttons for preset poses.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and gui_running:
        step_start = time.time()
        t = data.time

        # --- Demo trajectory (sinusoidal sweep) ---
        if demo_mode:
            q_target[0] = POSES["home"][0] + 0.8 * np.sin(0.5 * t)
            q_target[1] = POSES["home"][1] + 0.3 * np.sin(0.4 * t + 1.0)
            q_target[3] = POSES["home"][3] + 0.4 * np.sin(0.6 * t + 0.5)
            q_target[5] = POSES["home"][5] + 0.5 * np.sin(0.7 * t + 2.0)

        # --- PD control with gravity compensation ---
        grav_comp = data.qfrc_bias[:NUM_JOINTS].copy()
        for i in range(NUM_JOINTS):
            error = q_target[i] - data.qpos[i]
            error_dot = -data.qvel[i]
            data.ctrl[i] = KP[i] * error + KD[i] * error_dot + grav_comp[i]

        # Gripper
        if NUM_ACTUATORS > NUM_JOINTS:
            data.ctrl[NUM_JOINTS] = gripper_target

        # --- Step physics ---
        mujoco.mj_step(model, data)

        # --- FK ---
        ee_pos = data.xpos[EE_BODY_ID].copy()
        ee_rot = data.xmat[EE_BODY_ID].reshape(3, 3).copy()
        roll, pitch, yaw = rotation_matrix_to_euler(ee_rot)

        # --- Jacobian & EE velocity ---
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, EE_BODY_ID)
        ee_lin_vel = jacp @ data.qvel

        # --- Visualization ---
        viewer.user_scn.ngeom = 0

        # End-effector frame
        for axis_name, axis_vec in AXES:
            draw_arrow(viewer.user_scn, ee_pos, ee_rot @ axis_vec,
                       BODY_AXIS_LEN, BODY_AXIS_WIDTH, BODY_COLORS[axis_name])

        # World frame
        for axis_name, axis_vec in AXES:
            draw_arrow(viewer.user_scn, np.zeros(3), axis_vec,
                       WORLD_AXIS_LEN, WORLD_AXIS_WIDTH, WORLD_COLORS[axis_name])

        # Per-link frames
        for bid in LINK_BODY_IDS:
            if bid < 0:
                continue
            lpos = data.xpos[bid]
            lrot = data.xmat[bid].reshape(3, 3)
            for axis_name, axis_vec in AXES:
                draw_arrow(viewer.user_scn, lpos, lrot @ axis_vec,
                           0.06, 0.002, BODY_COLORS[axis_name] * 0.6)

        # --- HUD overlay ---
        q_deg = np.degrees(data.qpos[:NUM_JOINTS])
        hud_pos = ee_pos + np.array([0.0, 0.0, 0.45])
        lines = [
            f"EE pos: ({ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f})",
            f"EE rpy: ({np.degrees(roll):+.1f}, {np.degrees(pitch):+.1f}, {np.degrees(yaw):+.1f}) deg",
            f"EE vel: ({ee_lin_vel[0]:+.3f}, {ee_lin_vel[1]:+.3f}, {ee_lin_vel[2]:+.3f}) m/s",
            f"Joints: [{', '.join(f'{a:+.1f}' for a in q_deg)}] deg",
            f"Mode: {'DEMO' if demo_mode else 'SLIDER CONTROL'}",
        ]
        for i, text in enumerate(lines):
            draw_label(viewer.user_scn,
                       hud_pos + np.array([0, 0, 0.08 * (len(lines) - 1 - i)]),
                       text)

        # --- Follow camera ---
        if follow_camera:
            viewer.cam.lookat[:] = ee_pos
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        viewer.sync()

        # --- Terminal output ---
        now = time.time()
        if now - last_print_time > PRINT_INTERVAL:
            print(
                f"t={t:6.2f} | "
                f"EE=({ee_pos[0]:+.3f},{ee_pos[1]:+.3f},{ee_pos[2]:+.3f}) | "
                f"RPY=({np.degrees(roll):+.1f},{np.degrees(pitch):+.1f},{np.degrees(yaw):+.1f})° | "
                f"q={np.round(q_deg, 1)} | "
                f"|v|={np.linalg.norm(ee_lin_vel):.3f} m/s | "
                f"grip={'CLOSED' if gripper_target < 0.01 else 'OPEN'}"
            )
            last_print_time = now

        # --- Realtime sync ---
        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

gui_running = False
print("Simulation ended.")

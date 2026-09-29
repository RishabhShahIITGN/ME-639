"""
spawn_franka.py — Franka Emika Panda (7-DOF) Interactive MuJoCo Simulation
===========================================================================
Features:
  • PD joint-space controller with gravity compensation
  • Keyboard control: select joints, nudge ±, open/close gripper
  • Forward Kinematics: live end-effector position + orientation printed
  • Jacobian computation + end-effector velocity display
  • Body/world frame axes visualization (arrows drawn in viewer)
  • Telemetry HUD overlay (position, orientation, joint angles)
  • Predefined poses (home, zero, ready) via hotkeys
  • Follow-camera mode
  • Sinusoidal demo trajectory mode (toggle on/off)

Controls:
  Arrow UP / DOWN   →  select previous / next joint
  Arrow LEFT / RIGHT →  nudge selected joint −/+
  G / H              →  close / open gripper
  1                  →  go to HOME pose
  2                  →  go to ZERO pose
  3                  →  go to READY pose
  T                  →  toggle demo trajectory mode
  F                  →  toggle follow-camera
  R                  →  reset simulation
  SPACE              →  hold current joint positions

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

MODEL_PATH = "franka_scene.xml"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find '{MODEL_PATH}'.")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ═══════════════════════════════════════════════════════════════════
# 2. IDENTIFY ROBOT STRUCTURE
# ═══════════════════════════════════════════════════════════════════
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
JOINT_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
FINGER_JOINT = "finger_joint1"

# Body IDs for FK
EE_BODY_NAME = "hand"
EE_BODY_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
LINK_BODY_NAMES = [f"link{i}" for i in range(8)] + ["hand"]
LINK_BODY_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in LINK_BODY_NAMES]

NUM_JOINTS = 7  # arm joints (excludes gripper)
NUM_ACTUATORS = model.nu  # 7 motors + 1 gripper = 8

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
# PD gains (tuned per-joint: heavier proximal joints get higher gains)
KP = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
KD = np.array([50.0,  50.0,  50.0,  50.0,  20.0,  15.0,  5.0])

GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.0
JOINT_NUDGE = 0.05  # rad per keypress

# ═══════════════════════════════════════════════════════════════════
# 5. CONTROLLER STATE
# ═══════════════════════════════════════════════════════════════════
q_target = POSES["home"].copy()
gripper_target = GRIPPER_OPEN
selected_joint = 0  # 0..6
follow_camera = False
demo_mode = False
last_print_time = 0.0
PRINT_INTERVAL = 0.5

# ═══════════════════════════════════════════════════════════════════
# 6. KEYBOARD CALLBACK
# ═══════════════════════════════════════════════════════════════════
def keyboard_callback(keycode):
    global selected_joint, follow_camera, demo_mode, q_target, gripper_target

    if keycode == 265:  # UP arrow — previous joint
        selected_joint = max(0, selected_joint - 1)
    elif keycode == 264:  # DOWN arrow — next joint
        selected_joint = min(NUM_JOINTS - 1, selected_joint + 1)
    elif keycode == 263:  # LEFT arrow — nudge joint negative
        q_target[selected_joint] -= JOINT_NUDGE
    elif keycode == 262:  # RIGHT arrow — nudge joint positive
        q_target[selected_joint] += JOINT_NUDGE

    elif keycode in (71, 103):  # G — close gripper
        gripper_target = GRIPPER_CLOSE
    elif keycode in (72, 104):  # H — open gripper
        gripper_target = GRIPPER_OPEN

    elif keycode in (49,):  # 1 — home pose
        q_target = POSES["home"].copy()
        demo_mode = False
    elif keycode in (50,):  # 2 — zero pose
        q_target = POSES["zero"].copy()
        demo_mode = False
    elif keycode in (51,):  # 3 — ready pose
        q_target = POSES["ready"].copy()
        demo_mode = False

    elif keycode in (84, 116):  # T — toggle trajectory demo
        demo_mode = not demo_mode
    elif keycode in (70, 102):  # F — toggle follow camera
        follow_camera = not follow_camera
    elif keycode in (82, 114):  # R — reset
        mujoco.mj_resetData(model, data)
        q_target[:] = POSES["home"]
        gripper_target = GRIPPER_OPEN
        demo_mode = False
    elif keycode == 32:  # SPACE — hold current position
        q_target[:] = data.qpos[:NUM_JOINTS]
        demo_mode = False


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
    """Extract roll, pitch, yaw from a 3x3 rotation matrix."""
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def compute_jacobian(model, data, body_id):
    """Compute the full 6xN Jacobian for a body (position + orientation rows)."""
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    return jacp, jacr


# ═══════════════════════════════════════════════════════════════════
# 8. GRAVITY COMPENSATION
# ═══════════════════════════════════════════════════════════════════
def gravity_compensation(model, data):
    """Return the bias forces (gravity + Coriolis) for the arm joints."""
    return data.qfrc_bias[:NUM_JOINTS].copy()


# ═══════════════════════════════════════════════════════════════════
# 9. MAIN SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════
CONTROLS_TEXT = (
    "UP/DOWN=joint  LEFT/RIGHT=nudge  G/H=grip  "
    "1=home 2=zero 3=ready  T=demo  F=cam  R=reset  SPACE=hold"
)

print(f"\n{CONTROLS_TEXT}\n")

with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_callback) as viewer:
    while viewer.is_running():
        step_start = time.time()
        t = data.time

        # --- Demo trajectory (sinusoidal sweep) ---
        if demo_mode:
            q_target[0] = POSES["home"][0] + 0.8 * np.sin(0.5 * t)
            q_target[1] = POSES["home"][1] + 0.3 * np.sin(0.4 * t + 1.0)
            q_target[3] = POSES["home"][3] + 0.4 * np.sin(0.6 * t + 0.5)
            q_target[5] = POSES["home"][5] + 0.5 * np.sin(0.7 * t + 2.0)

        # --- PD control with gravity compensation ---
        grav_comp = gravity_compensation(model, data)
        for i in range(NUM_JOINTS):
            error = q_target[i] - data.qpos[i]
            error_dot = -data.qvel[i]
            data.ctrl[i] = KP[i] * error + KD[i] * error_dot + grav_comp[i]

        # Gripper control (actuator index 7)
        if NUM_ACTUATORS > NUM_JOINTS:
            data.ctrl[NUM_JOINTS] = gripper_target

        # --- Step physics ---
        mujoco.mj_step(model, data)

        # --- Forward Kinematics ---
        ee_pos = data.xpos[EE_BODY_ID].copy()
        ee_rot = data.xmat[EE_BODY_ID].reshape(3, 3).copy()
        roll, pitch, yaw = rotation_matrix_to_euler(ee_rot)

        # --- Jacobian & EE velocity ---
        jacp, jacr = compute_jacobian(model, data, EE_BODY_ID)
        ee_lin_vel = jacp @ data.qvel
        ee_ang_vel = jacr @ data.qvel

        # --- Visualization ---
        viewer.user_scn.ngeom = 0

        # Draw end-effector body frame
        for axis_name, axis_vec in AXES:
            draw_arrow(viewer.user_scn, ee_pos, ee_rot @ axis_vec,
                       BODY_AXIS_LEN, BODY_AXIS_WIDTH, BODY_COLORS[axis_name])

        # Draw world frame at origin
        for axis_name, axis_vec in AXES:
            draw_arrow(viewer.user_scn, np.zeros(3), axis_vec,
                       WORLD_AXIS_LEN, WORLD_AXIS_WIDTH, WORLD_COLORS[axis_name])

        # Draw axes on each link
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
            f"Selected: joint{selected_joint+1} | {'DEMO' if demo_mode else 'MANUAL'}",
            CONTROLS_TEXT,
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
                f"|v_ee|={np.linalg.norm(ee_lin_vel):.3f} m/s | "
                f"grip={'CLOSED' if gripper_target < 0.01 else 'OPEN'}"
            )
            last_print_time = now

        # --- Realtime sync ---
        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

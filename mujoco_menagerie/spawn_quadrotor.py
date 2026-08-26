import mujoco
import mujoco.viewer
import numpy as np
import os
import time

# ==================================================================
# 1. LOAD THE SKYDIO X2 MODEL
# ==================================================================
MODEL_PATH = os.path.join("skydio_x2", "scene.xml")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find '{MODEL_PATH}'. Run from inside 'mujoco_menagerie/'.")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

BODY_NAME = "x2"
ACTUATOR_NAMES = ["thrust1", "thrust2", "thrust3", "thrust4"]
KEYFRAME_NAME = "hover"

BODY_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
HOVER_KEY_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
assert BODY_ID != -1, f"Body '{BODY_NAME}' not found."
for a in ACTUATOR_NAMES:
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) != -1, \
        f"Actuator '{a}' not found."

mujoco.mj_resetDataKeyframe(model, data, HOVER_KEY_ID)
mujoco.mj_forward(model, data)

MASS = model.body_mass[BODY_ID]                   # 1.325 kg  (from model)
GRAVITY = abs(model.opt.gravity[2])               # 9.81      (from model)
IXX, IYY, IZZ = model.body_inertia[BODY_ID]        # real inertia (from model)
CTRL_LO = model.actuator_ctrlrange[:, 0].copy()    # real actuator limits
CTRL_HI = model.actuator_ctrlrange[:, 1].copy()

print(f"Body '{BODY_NAME}': mass={MASS:.3f} kg, "
      f"inertia=({IXX:.4f},{IYY:.4f},{IZZ:.4f}) kg*m^2")

# ==================================================================
# 2. MOTOR MIXER — built from the model's real geometry
# ==================================================================
rows = []
for name in ACTUATOR_NAMES:
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    site_id = model.actuator_trnid[act_id, 0]
    site_pos = model.site_pos[site_id]
    yaw_coeff = model.actuator_gear[act_id, 5]
    rows.append((site_pos[0], site_pos[1], yaw_coeff))
    print(f"  {name}: pos=({site_pos[0]:+.3f},{site_pos[1]:+.3f}), "
          f"yaw_coeff={yaw_coeff:+.4f} ({'CW' if yaw_coeff < 0 else 'CCW'})")

M = np.zeros((4, 4))
for i, (x_i, y_i, k_i) in enumerate(rows):
    M[0, i] = 1.0
    M[1, i] = y_i      # roll torque  (about body x)
    M[2, i] = -x_i      # pitch torque (about body y)
    M[3, i] = k_i        # yaw torque   (motor reaction torque)
M_inv = np.linalg.inv(M)
MAX_TORQUE_XY = 4.0     # N*m safety clamp on commanded roll/pitch torque before mixing
MAX_TORQUE_Z = 0.5      # N*m safety clamp on commanded yaw torque

# ==================================================================
# 3. TUNING PARAMETERS  (all here, as requested)
# ==================================================================
# ---- Simulation ----
SIM_DT = model.opt.timestep            # 0.01s — controller runs at physics rate, not render rate

# ---- Motion limits ----
MAX_HORIZONTAL_SPEED = 1.5             # m/s
MAX_VERTICAL_SPEED = 1.0               # m/s
MAX_HORIZONTAL_ACCEL = 3.0             # m/s^2 - also used as the velocity-target ramp rate
MAX_VERTICAL_ACCEL = 6.0               # m/s^2
MAX_YAW_RATE = 1.0                     # rad/s
MAX_YAW_ACCELERATION = 2.0             # rad/s^2 - ramp rate for yaw-rate target
MAX_ANGULAR_VELOCITY = 6.0             # rad/s - defensive clip on gyro reading used in D-terms

# ---- Attitude limits ----
MAX_ROLL = 0.35                        # rad (~20 deg)
MAX_PITCH = 0.35                       # rad (~20 deg)

# ---- Position/velocity controller ----
# KP_POSITION: gain from (target velocity - actual velocity) -> desired accel.
#   This is the PRIMARY mechanism that tracks your commanded velocity.
# KD_POSITION: gain on position error, used only while idle (keys released) to
#   gently anchor against long-term drift. NOT used while actively commanding.
KP_POSITION = 0.6
KD_POSITION = 0.3

# ---- Attitude controller ----
# Derived from the drone's REAL inertia via critical-damping design:
#   Kp = I * wn^2,  Kd = 2*zeta*I*wn
WN_ATT, ZETA_ATT = 6.0, 0.8            # roll & pitch bandwidth/damping
WN_YAW, ZETA_YAW = 2.0, 1.0            # yaw is slower: weak motor authority for yaw
KP_ROLL = IXX * WN_ATT ** 2
KD_ROLL = 2 * ZETA_ATT * IXX * WN_ATT
KP_PITCH = IYY * WN_ATT ** 2
KD_PITCH = 2 * ZETA_ATT * IYY * WN_ATT
KP_YAW = IZZ * WN_YAW ** 2
KD_YAW = 2 * ZETA_YAW * IZZ * WN_YAW

# ---- Aerodynamics (real external force/torque, not a fake stop) ----
LINEAR_DRAG = 0.5        # N per (m/s)      : F = -LINEAR_DRAG * v
QUADRATIC_DRAG = 0.15    # N per (m/s)^2    : F += -QUADRATIC_DRAG * |v| * v
ANGULAR_DRAG = 0.08      # N*m per (rad/s)  : tau = -ANGULAR_DRAG * omega_body
# Linear + quadratic combined: linear term dominates at low (typical indoor)
# speeds and gives gentle, physically-motivated settling; the quadratic term
# adds extra resistance if speed limits are pushed higher.

# ---- Motor dynamics ----
MOTOR_TIME_CONSTANT = 0.03   # s - first-order lag (raw actuators have none)

# ---- Keyboard shaping ----
CMD_DECAY = 0.98          # per-tick decay of raw keyboard command (see note below)
NUDGE_V = 0.35            # m/s added to raw command per key event
NUDGE_YAW_RATE = 0.6      # rad/s added to yaw-rate command per key event

# ---- Safety ----
GROUND_SAFE_HEIGHT = 0.15
MAX_ALTITUDE = 3.0

# ==================================================================
# 4. CONTROLLER STATE
# ==================================================================
motor_filt = np.array(model.key_ctrl[HOVER_KEY_ID], dtype=float)
cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
vel_target = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
yaw_rate_target = 0.0
yaw_target = 0.0
hold_active = False
hold_pos_xy = None
follow_camera = False
last_debug_time = 0.0
DEBUG_INTERVAL = 0.5

# ==================================================================
# 5. KEYBOARD CALLBACK
#    NOTE: MuJoCo's passive-viewer key_callback fires on key-DOWN
#    events only — there's no key-release event in this API. So
#    "hold to keep moving, release to stop" is approximated: each
#    event nudges a raw command that decays every tick (CMD_DECAY),
#    which is then rate-limited into a smooth velocity target. If
#    your system's key-repeat is on, holding a key gives a stream
#    of nudges that behaves like continuous input; a single tap
#    gives a gentle nudge that glides back to hover either way.
# ==================================================================
def keyboard_callback(keycode):
    global follow_camera
    if keycode == 265:          # UP - forward
        cmd["vx"] = np.clip(cmd["vx"] + NUDGE_V, -MAX_HORIZONTAL_SPEED, MAX_HORIZONTAL_SPEED)
    elif keycode == 264:        # DOWN - backward
        cmd["vx"] = np.clip(cmd["vx"] - NUDGE_V, -MAX_HORIZONTAL_SPEED, MAX_HORIZONTAL_SPEED)
    elif keycode == 263:        # LEFT
        cmd["vy"] = np.clip(cmd["vy"] + NUDGE_V, -MAX_HORIZONTAL_SPEED, MAX_HORIZONTAL_SPEED)
    elif keycode == 262:        # RIGHT
        cmd["vy"] = np.clip(cmd["vy"] - NUDGE_V, -MAX_HORIZONTAL_SPEED, MAX_HORIZONTAL_SPEED)
    elif keycode in (87, 119):  # W - ascend
        cmd["vz"] = np.clip(cmd["vz"] + NUDGE_V, -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED)
    elif keycode in (83, 115):  # S - descend
        cmd["vz"] = np.clip(cmd["vz"] - NUDGE_V, -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED)
    elif keycode in (65, 97):   # A - yaw left
        cmd["yaw_rate"] = np.clip(cmd["yaw_rate"] + NUDGE_YAW_RATE, -MAX_YAW_RATE, MAX_YAW_RATE)
    elif keycode in (68, 100):  # D - yaw right
        cmd["yaw_rate"] = np.clip(cmd["yaw_rate"] - NUDGE_YAW_RATE, -MAX_YAW_RATE, MAX_YAW_RATE)
    elif keycode == 32:         # SPACE - cancel commands, hold position
        cmd["vx"] = cmd["vy"] = cmd["vz"] = cmd["yaw_rate"] = 0.0
    elif keycode in (70, 102):  # F - toggle follow camera (view only, no physics change)
        follow_camera = not follow_camera
    elif keycode in (82, 114):  # R - full reset
        mujoco.mj_resetDataKeyframe(model, data, HOVER_KEY_ID)
        mujoco.mj_forward(model, data)
        cmd["vx"] = cmd["vy"] = cmd["vz"] = cmd["yaw_rate"] = 0.0
        vel_target["vx"] = vel_target["vy"] = vel_target["vz"] = 0.0
        globals()["yaw_rate_target"] = 0.0
        globals()["yaw_target"] = 0.0
        globals()["hold_active"] = False
        motor_filt[:] = model.key_ctrl[HOVER_KEY_ID]
        data.xfrc_applied[BODY_ID, :] = 0.0
    # Note on ESC: this viewer's window-close behavior is handled by the
    # compiled MuJoCo GUI backend, not by this Python key_callback, and its
    # exact ESC binding isn't something I can verify without running it on
    # your machine. If ESC doesn't close the window, use the window's close
    # button, or Ctrl+C in the terminal.


# ==================================================================
# 6. VISUALIZATION HELPERS
# ==================================================================
BODY_AXIS_LENGTH, WORLD_AXIS_LENGTH = 0.30, 0.6
BODY_AXIS_WIDTH, WORLD_AXIS_WIDTH = 0.010, 0.005
BODY_COLORS = {'x': np.array([1.,0.,0.,1.]), 'y': np.array([0.,1.,0.,1.]), 'z': np.array([0.,0.,1.,1.])}
WORLD_COLORS = {'x': np.array([.6,.2,.2,.5]), 'y': np.array([.2,.6,.2,.5]), 'z': np.array([.2,.2,.6,.5])}
AXES = [('x', np.array([1.,0.,0.])), ('y', np.array([0.,1.,0.])), ('z', np.array([0.,0.,1.]))]


def draw_arrow(scn, origin, direction, length, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    end = origin + direction * length
    mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_ARROW, size=np.zeros(3),
                         pos=np.zeros(3), mat=np.eye(3).flatten(), rgba=rgba.astype(np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, origin, end)
    scn.ngeom += 1


def draw_label(scn, pos, text):
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.001, 0., 0.]),
                         pos=pos, mat=np.eye(3).flatten(), rgba=np.array([1,1,1,1], dtype=np.float32))
    geom.label = text
    scn.ngeom += 1


def matrix_rows(R):
    return ["[ " + "  ".join(f"{R[i,j]: .3f}" for j in range(3)) + " ]" for i in range(3)]


# ==================================================================
# 7. MAIN LOOP
# ==================================================================
CONTROLS_TEXT = (
    "UP/DOWN=fwd/back  LEFT/RIGHT=strafe  W/S=up/down  "
    "A/D=yaw  SPACE=hold  R=reset  F=follow-cam"
)

with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_callback) as viewer:
    print("\n" + CONTROLS_TEXT + "\n")

    while viewer.is_running():
        step_start = time.time()

        cmd["vx"] *= CMD_DECAY
        cmd["vy"] *= CMD_DECAY
        cmd["vz"] *= CMD_DECAY
        cmd["yaw_rate"] *= CMD_DECAY

        R = data.xmat[BODY_ID].reshape(3, 3).copy()
        pos = data.xpos[BODY_ID].copy()
        vel_world = data.cvel[BODY_ID][3:6].copy()
        gyro = np.clip(data.sensor('body_gyro').data.copy(),
                        -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)

        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        yaw = np.arctan2(R[1, 0], R[0, 0])

        cy, sy = np.cos(yaw), np.sin(yaw)
        vx_body = cy*vel_world[0] + sy*vel_world[1]
        vy_body = -sy*vel_world[0] + cy*vel_world[1]

        vel_target["vx"] += np.clip(cmd["vx"]-vel_target["vx"], -MAX_HORIZONTAL_ACCEL*SIM_DT, MAX_HORIZONTAL_ACCEL*SIM_DT)
        vel_target["vy"] += np.clip(cmd["vy"]-vel_target["vy"], -MAX_HORIZONTAL_ACCEL*SIM_DT, MAX_HORIZONTAL_ACCEL*SIM_DT)
        vel_target["vz"] += np.clip(cmd["vz"]-vel_target["vz"], -MAX_VERTICAL_ACCEL*SIM_DT, MAX_VERTICAL_ACCEL*SIM_DT)
        yaw_rate_target += np.clip(cmd["yaw_rate"]-yaw_rate_target, -MAX_YAW_ACCELERATION*SIM_DT, MAX_YAW_ACCELERATION*SIM_DT)
        yaw_target += yaw_rate_target * SIM_DT

        idle = abs(cmd["vx"]) < 1e-4 and abs(cmd["vy"]) < 1e-4
        if idle and not hold_active:
            hold_pos_xy = pos[:2].copy()
            hold_active = True
        elif not idle:
            hold_active = False

        # position-hold correction, computed in WORLD frame then rotated
        # into BODY frame before use (this is the Problem-B fix)
        pos_corr_bx = pos_corr_by = 0.0
        if hold_active and hold_pos_xy is not None:
            err_wx, err_wy = pos[0]-hold_pos_xy[0], pos[1]-hold_pos_xy[1]
            err_bx = cy*err_wx + sy*err_wy
            err_by = -sy*err_wx + cy*err_wy
            pos_corr_bx = -KD_POSITION * err_bx
            pos_corr_by = -KD_POSITION * err_by

        ax_des = np.clip(KP_POSITION*(vel_target["vx"]-vx_body) + pos_corr_bx, -MAX_HORIZONTAL_ACCEL, MAX_HORIZONTAL_ACCEL)
        ay_des = np.clip(KP_POSITION*(vel_target["vy"]-vy_body) + pos_corr_by, -MAX_HORIZONTAL_ACCEL, MAX_HORIZONTAL_ACCEL)
        pitch_des = np.clip(ax_des/GRAVITY, -MAX_PITCH, MAX_PITCH)
        roll_des = np.clip(-ay_des/GRAVITY, -MAX_ROLL, MAX_ROLL)

        az_des = 6.0*(vel_target["vz"]-vel_world[2])
        if pos[2] < GROUND_SAFE_HEIGHT:
            az_des = max(az_des, 0.0)
        if pos[2] > MAX_ALTITUDE:
            az_des = min(az_des, 0.0)
        Fz = MASS*(GRAVITY+az_des)

        Tx = np.clip(KP_ROLL*(roll_des-roll) - KD_ROLL*gyro[0], -MAX_TORQUE_XY, MAX_TORQUE_XY)
        Ty = np.clip(KP_PITCH*(pitch_des-pitch) - KD_PITCH*gyro[1], -MAX_TORQUE_XY, MAX_TORQUE_XY)
        Tz = np.clip(KP_YAW*(yaw_target-yaw) - KD_YAW*gyro[2], -MAX_TORQUE_Z, MAX_TORQUE_Z)

        motor_cmd = np.clip(M_inv @ np.array([Fz, Tx, Ty, Tz]), CTRL_LO, CTRL_HI)
        motor_filt += (motor_cmd-motor_filt) * (SIM_DT/MOTOR_TIME_CONSTANT)
        data.ctrl[:] = motor_filt

        # real aerodynamic drag, applied as an external force/torque
        speed = np.linalg.norm(vel_world)
        F_drag = -LINEAR_DRAG*vel_world - QUADRATIC_DRAG*speed*vel_world
        tau_drag_world = R @ (-ANGULAR_DRAG * gyro)
        data.xfrc_applied[BODY_ID, 0:3] = F_drag
        data.xfrc_applied[BODY_ID, 3:6] = tau_drag_world

        mujoco.mj_step(model, data)

        # ---- draw frames ----
        viewer.user_scn.ngeom = 0
        for axis_name, axis_vec in AXES:
            draw_arrow(viewer.user_scn, pos, R @ axis_vec, BODY_AXIS_LENGTH, BODY_AXIS_WIDTH, BODY_COLORS[axis_name])
        for axis_name, axis_vec in AXES:
            draw_arrow(viewer.user_scn, np.zeros(3), axis_vec, WORLD_AXIS_LENGTH, WORLD_AXIS_WIDTH, WORLD_COLORS[axis_name])

        # ---- floating telemetry overlay (this viewer API has no fixed
        # 2D HUD, so it's drawn as 3D text anchored above the drone) ----
        lo = pos + np.array([0.0, 0.0, 0.55])
        lines = [
            f"R_world_body   roll={np.degrees(roll):6.1f} pitch={np.degrees(pitch):6.1f} yaw={np.degrees(yaw):6.1f} deg",
            *matrix_rows(R),
            f"pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) vel=({vel_world[0]:+.2f},{vel_world[1]:+.2f},{vel_world[2]:+.2f}) |v|={speed:.2f}",
            f"omega_body=({gyro[0]:+.2f},{gyro[1]:+.2f},{gyro[2]:+.2f}) motors={np.round(motor_filt,2)}",
            CONTROLS_TEXT,
        ]
        for i, text in enumerate(lines):
            draw_label(viewer.user_scn, lo + np.array([0, 0, 0.13*(len(lines)-1-i)]), text)

        # ---- camera follow (view only — never touches physics state) ----
        if follow_camera:
            viewer.cam.lookat[:] = pos
            viewer.cam.trackbodyid = BODY_ID
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        viewer.sync()

        now = time.time()
        if now - last_debug_time > DEBUG_INTERVAL:
            print(f"t={data.time:6.2f} pos={np.round(pos,3)} vel={np.round(vel_world,3)} |v|={speed:.3f} "
                  f"rpy_deg=({np.degrees(roll):+.1f},{np.degrees(pitch):+.1f},{np.degrees(yaw):+.1f}) "
                  f"omega={np.round(gyro,3)} Fz={Fz:.2f} motors={np.round(motor_filt,2)} "
                  f"drag_F={np.round(F_drag,3)} drag_tau={np.round(tau_drag_world,4)}")
            last_debug_time = now

        elapsed = time.time() - step_start
        sleep_time = SIM_DT - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
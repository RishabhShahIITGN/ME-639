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
        f"Could not find '{MODEL_PATH}'. Run this script from inside your "
        f"'mujoco_menagerie' folder (the one that contains 'skydio_x2/')."
    )

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

BODY_NAME = "x2"
ACTUATOR_NAMES = ["thrust1", "thrust2", "thrust3", "thrust4"]
KEYFRAME_NAME = "hover"

BODY_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
HOVER_KEY_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
assert BODY_ID != -1, f"Body '{BODY_NAME}' not found — inspect x2.xml."
for a in ACTUATOR_NAMES:
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) != -1, \
        f"Actuator '{a}' not found — inspect x2.xml <actuator>."

mujoco.mj_resetDataKeyframe(model, data, HOVER_KEY_ID)
mujoco.mj_forward(model, data)

MASS = model.body_mass[BODY_ID]                  # 1.325 kg, read from model
GRAVITY = abs(model.opt.gravity[2])              # 9.81, read from model
IXX, IYY, IZZ = model.body_inertia[BODY_ID]       # real inertia, read from model
CTRL_LO = model.actuator_ctrlrange[:, 0].copy()   # real actuator limits [0,13]
CTRL_HI = model.actuator_ctrlrange[:, 1].copy()
CONTROL_DT = model.opt.timestep                   # 0.01s — controller runs at physics rate

print(f"Body '{BODY_NAME}': mass={MASS:.3f} kg, "
      f"inertia=({IXX:.4f},{IYY:.4f},{IZZ:.4f}) kg*m^2")
print(f"Control timestep (= physics timestep): {CONTROL_DT}s "
      f"({1/CONTROL_DT:.0f} Hz)")

# ==================================================================
# 2. MOTOR MIXER — built from the model's real geometry, not assumed
# ==================================================================
rows = []
for name in ACTUATOR_NAMES:
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    site_id = model.actuator_trnid[act_id, 0]
    site_pos = model.site_pos[site_id]
    yaw_coeff = model.actuator_gear[act_id, 5]     # reaction-torque coeff, real value
    rows.append((site_pos[0], site_pos[1], yaw_coeff))
    print(f"  {name}: site_pos=({site_pos[0]:+.3f},{site_pos[1]:+.3f}), "
          f"yaw_coeff={yaw_coeff:+.4f} "
          f"({'CW' if yaw_coeff < 0 else 'CCW'} reaction)")

M = np.zeros((4, 4))
for i, (x_i, y_i, k_i) in enumerate(rows):
    M[0, i] = 1.0     # total thrust
    M[1, i] = y_i      # roll torque about body x-axis  (r x F)
    M[2, i] = -x_i      # pitch torque about body y-axis (r x F)
    M[3, i] = k_i       # yaw torque about body z-axis (motor reaction torque)
M_inv = np.linalg.inv(M)

# ==================================================================
# 3. GAINS AND LIMITS  (all in one place, as requested)
# ==================================================================
# --- Attitude PD gains: derived from REAL inertia via critical damping
#     Kp = I * wn^2,  Kd = 2*zeta*I*wn
#     wn = closed-loop natural frequency (rad/s) -> higher = snappier
#     zeta = damping ratio (1.0 = no overshoot, <1 = some overshoot)
WN_ATT, ZETA_ATT = 6.0, 0.8     # roll & pitch attitude loop
WN_YAW, ZETA_YAW = 2.0, 1.0     # yaw loop (weaker motor authority -> slower)

KP_ATTITUDE_ROLL = IXX * WN_ATT ** 2
KD_ATTITUDE_ROLL = 2 * ZETA_ATT * IXX * WN_ATT
KP_ATTITUDE_PITCH = IYY * WN_ATT ** 2
KD_ATTITUDE_PITCH = 2 * ZETA_ATT * IYY * WN_ATT
KP_YAW = IZZ * WN_YAW ** 2
KD_YAW = 2 * ZETA_YAW * IZZ * WN_YAW

# --- Velocity/position loop gains (keyboard command -> desired tilt/accel)
KP_VELOCITY = 0.6        # horizontal velocity-error -> desired accel (m/s^2 per m/s)
KP_VELOCITY_Z = 6.0      # vertical velocity-error -> desired vertical accel
KP_POSITION_HOLD = 0.3   # gentle pull back to the position captured when idle

# --- Limits (saturation / safety)
MAX_TILT_ANGLE = 0.35      # rad (~20 deg) - hard clip on commanded roll/pitch
MAX_ACCELERATION = 3.0     # m/s^2 - clip on desired horizontal accel
MAX_HORIZONTAL_SPEED = 1.5 # m/s   - clip on velocity target
MAX_VERTICAL_SPEED = 1.0   # m/s
MAX_YAW_RATE = 1.0         # rad/s - clip on commanded yaw rate
RAMP_ACCEL = 1.0           # m/s^2 - how fast the *velocity target* itself is allowed to ramp
CMD_DECAY = 0.98           # per-tick decay of the raw keyboard command (see note below)
NUDGE_V = 0.35             # m/s added to the raw command per key event
NUDGE_YAW_RATE = 0.6       # rad/s added to yaw-rate command per key event
MOTOR_TIME_CONSTANT = 0.03 # s - first-order motor response (model has no motor dynamics)
GROUND_SAFE_HEIGHT = 0.15  # m - below this, downward acceleration commands are blocked
MAX_ALTITUDE = 3.0         # m - above this, upward acceleration commands are blocked

# ==================================================================
# 4. CONTROLLER STATE
# ==================================================================
motor_filt = np.array(model.key_ctrl[HOVER_KEY_ID], dtype=float)  # start at real hover ctrl
cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
vel_target = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
yaw_target = 0.0
hold_active = False
hold_pos_xy = None
last_debug_time = 0.0
DEBUG_INTERVAL = 0.5  # seconds between debug prints

# ==================================================================
# 5. KEYBOARD CALLBACK
#    NOTE: MuJoCo's passive-viewer key_callback fires on key-DOWN
#    events only; there is no key-release event available. So "hold
#    to keep moving, release to stop" is approximated: each event
#    nudges a command that decays every tick (CMD_DECAY). If your
#    system's key-repeat is on, holding a key produces a stream of
#    nudges that behaves like continuous input; a single tap gives a
#    gentle nudge that glides back to hover. Both are stable.
# ==================================================================
def keyboard_callback(keycode):
    global yaw_target
    if keycode == 265:      # UP - forward
        cmd["vx"] = np.clip(cmd["vx"] + NUDGE_V, -MAX_HORIZONTAL_SPEED, MAX_HORIZONTAL_SPEED)
    elif keycode == 264:    # DOWN - backward
        cmd["vx"] = np.clip(cmd["vx"] - NUDGE_V, -MAX_HORIZONTAL_SPEED, MAX_HORIZONTAL_SPEED)
    elif keycode == 263:    # LEFT
        cmd["vy"] = np.clip(cmd["vy"] + NUDGE_V, -MAX_HORIZONTAL_SPEED, MAX_HORIZONTAL_SPEED)
    elif keycode == 262:    # RIGHT
        cmd["vy"] = np.clip(cmd["vy"] - NUDGE_V, -MAX_HORIZONTAL_SPEED, MAX_HORIZONTAL_SPEED)
    elif keycode in (87, 119):    # W - ascend
        cmd["vz"] = np.clip(cmd["vz"] + NUDGE_V, -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED)
    elif keycode in (83, 115):    # S - descend
        cmd["vz"] = np.clip(cmd["vz"] - NUDGE_V, -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED)
    elif keycode in (65, 97):     # A - yaw left
        cmd["yaw_rate"] = np.clip(cmd["yaw_rate"] + NUDGE_YAW_RATE, -MAX_YAW_RATE, MAX_YAW_RATE)
    elif keycode in (68, 100):    # D - yaw right
        cmd["yaw_rate"] = np.clip(cmd["yaw_rate"] - NUDGE_YAW_RATE, -MAX_YAW_RATE, MAX_YAW_RATE)
    elif keycode == 32:      # SPACE - cancel commands, hold current position
        cmd["vx"] = cmd["vy"] = cmd["vz"] = cmd["yaw_rate"] = 0.0
    elif keycode in (82, 114):    # R - full reset
        mujoco.mj_resetDataKeyframe(model, data, HOVER_KEY_ID)
        mujoco.mj_forward(model, data)
        cmd["vx"] = cmd["vy"] = cmd["vz"] = cmd["yaw_rate"] = 0.0
        vel_target["vx"] = vel_target["vy"] = vel_target["vz"] = 0.0
        yaw_target = 0.0
        motor_filt[:] = model.key_ctrl[HOVER_KEY_ID]
    # ESC is handled natively by the MuJoCo viewer (closes the window) —
    # no extra code needed here.


# ==================================================================
# 6. VISUALIZATION HELPERS (world frame, body frame, matrix label)
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
with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_callback) as viewer:
    print("\nUP/DOWN forward/back | LEFT/RIGHT strafe | W/S up/down | A/D yaw | SPACE hold | R reset\n")

    while viewer.is_running():
        step_start = time.time()

        # ---- decay raw commands (approximates "key released") ----
        cmd["vx"] *= CMD_DECAY
        cmd["vy"] *= CMD_DECAY
        cmd["vz"] *= CMD_DECAY
        cmd["yaw_rate"] *= CMD_DECAY

        # ---- read state ----
        R = data.xmat[BODY_ID].reshape(3, 3).copy()   # R_world_body
        pos = data.xpos[BODY_ID].copy()
        vel_world = data.cvel[BODY_ID][3:6].copy()
        gyro = data.sensor('body_gyro').data.copy()

        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        yaw = np.arctan2(R[1, 0], R[0, 0])

        cy, sy = np.cos(yaw), np.sin(yaw)
        vx_body = cy * vel_world[0] + sy * vel_world[1]
        vy_body = -sy * vel_world[0] + cy * vel_world[1]

        # ---- ramp velocity target toward commanded velocity (bounded accel) ----
        vel_target["vx"] += np.clip(cmd["vx"] - vel_target["vx"], -RAMP_ACCEL*CONTROL_DT, RAMP_ACCEL*CONTROL_DT)
        vel_target["vy"] += np.clip(cmd["vy"] - vel_target["vy"], -RAMP_ACCEL*CONTROL_DT, RAMP_ACCEL*CONTROL_DT)
        vel_target["vz"] += np.clip(cmd["vz"] - vel_target["vz"], -RAMP_ACCEL*CONTROL_DT, RAMP_ACCEL*CONTROL_DT)
        yaw_target += np.clip(cmd["yaw_rate"], -MAX_YAW_RATE, MAX_YAW_RATE) * CONTROL_DT

        # ---- position hold when idle (prevents long-term drift) ----
        idle = abs(cmd["vx"]) < 1e-4 and abs(cmd["vy"]) < 1e-4
        if idle and not hold_active:
            hold_pos_xy = pos[:2].copy()
            hold_active = True
        elif not idle:
            hold_active = False
        pos_corr_x = pos_corr_y = 0.0
        if hold_active and hold_pos_xy is not None:
            pos_corr_x = -KP_POSITION_HOLD * (pos[0] - hold_pos_xy[0])
            pos_corr_y = -KP_POSITION_HOLD * (pos[1] - hold_pos_xy[1])

        # ---- velocity -> desired acceleration -> desired tilt ----
        ax_des = np.clip(KP_VELOCITY * (vel_target["vx"] - vx_body) + pos_corr_x,
                          -MAX_ACCELERATION, MAX_ACCELERATION)
        ay_des = np.clip(KP_VELOCITY * (vel_target["vy"] - vy_body) + pos_corr_y,
                          -MAX_ACCELERATION, MAX_ACCELERATION)
        pitch_des = np.clip(ax_des / GRAVITY, -MAX_TILT_ANGLE, MAX_TILT_ANGLE)
        roll_des = np.clip(-ay_des / GRAVITY, -MAX_TILT_ANGLE, MAX_TILT_ANGLE)

        # ---- vertical velocity -> desired thrust (with ground/ceiling safety) ----
        az_des = KP_VELOCITY_Z * (vel_target["vz"] - vel_world[2])
        if pos[2] < GROUND_SAFE_HEIGHT:
            az_des = max(az_des, 0.0)     # forbid further descent near the ground
        if pos[2] > MAX_ALTITUDE:
            az_des = min(az_des, 0.0)     # forbid further climb above ceiling
        Fz = MASS * (GRAVITY + az_des)

        # ---- attitude PD (uses gyro for damping) ----
        Tx = KP_ATTITUDE_ROLL * (roll_des - roll) - KD_ATTITUDE_ROLL * gyro[0]
        Ty = KP_ATTITUDE_PITCH * (pitch_des - pitch) - KD_ATTITUDE_PITCH * gyro[1]
        Tz = KP_YAW * (yaw_target - yaw) - KD_YAW * gyro[2]

        # ---- mixer + saturation ----
        motor_cmd = np.clip(M_inv @ np.array([Fz, Tx, Ty, Tz]), CTRL_LO, CTRL_HI)

        # ---- simulated first-order motor response ----
        motor_filt += (motor_cmd - motor_filt) * (CONTROL_DT / MOTOR_TIME_CONSTANT)
        data.ctrl[:] = motor_filt

        # ---- step physics ----
        mujoco.mj_step(model, data)

        # ---- draw frames + rotation matrix ----
        viewer.user_scn.ngeom = 0
        for axis_name, axis_vec in AXES:
            draw_arrow(viewer.user_scn, pos, R @ axis_vec, BODY_AXIS_LENGTH, BODY_AXIS_WIDTH, BODY_COLORS[axis_name])
        for axis_name, axis_vec in AXES:
            draw_arrow(viewer.user_scn, np.zeros(3), axis_vec, WORLD_AXIS_LENGTH, WORLD_AXIS_WIDTH, WORLD_COLORS[axis_name])

        label_origin = pos + np.array([0.0, 0.0, 0.5])
        draw_label(viewer.user_scn, label_origin + np.array([0,0,0.36]),
                   f"R_world_body  roll={np.degrees(roll):5.1f} pitch={np.degrees(pitch):5.1f} yaw={np.degrees(yaw):5.1f}")
        for i, row_text in enumerate(matrix_rows(R)):
            draw_label(viewer.user_scn, label_origin + np.array([0,0,0.12*(2-i)]), row_text)

        viewer.sync()

        # ---- throttled debug printout ----
        now = time.time()
        if now - last_debug_time > DEBUG_INTERVAL:
            print(f"t={data.time:6.2f}  Fz={Fz:5.2f}  roll_err={np.degrees(roll_des-roll):+5.1f}deg "
                  f"pitch_err={np.degrees(pitch_des-pitch):+5.1f}deg yaw_err={np.degrees(yaw_target-yaw):+5.1f}deg  "
                  f"motors={np.round(motor_filt,2)}")
            last_debug_time = now

        elapsed = time.time() - step_start
        sleep_time = CONTROL_DT - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
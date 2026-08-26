import mujoco
import mujoco.viewer
import numpy as np
import time

# ------------------------------------------------------------------
# Load the robot model
# ------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path('robotis_tb3/scene_turtlebot3_waffle_pi.xml')
data = mujoco.MjData(model)

# ------------------------------------------------------------------
# Figure out which body is the robot's base
# ------------------------------------------------------------------
def get_robot_body_id(model):
    candidate = None
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name is None:
            continue
        if "base" in name.lower():
            candidate = i
            break
    if candidate is None:
        candidate = 1 if model.nbody > 1 else 0
    return candidate

ROBOT_BODY_ID = get_robot_body_id(model)
robot_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_BODY_ID)
print(f"Tracking body frame of: '{robot_body_name}' (id={ROBOT_BODY_ID})")

# ------------------------------------------------------------------
# Auto-detect wheel actuator indices by name
# ------------------------------------------------------------------
print("\nAvailable actuators in this model:")
for i in range(model.nu):
    aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    crange = model.actuator_ctrlrange[i]
    print(f"  [{i}] {aname}  ctrlrange={crange}")

def find_actuator(keywords, default_id):
    for i in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if aname and any(k in aname.lower() for k in keywords):
            return i
    return default_id

LEFT_ACT = find_actuator(["left"], default_id=0)
RIGHT_ACT = find_actuator(["right"], default_id=1 if model.nu > 1 else 0)
print(f"\nUsing actuator {LEFT_ACT} as LEFT wheel, actuator {RIGHT_ACT} as RIGHT wheel.\n")

CTRL_STEP = 1.0

def clamp_ctrl(act_id, value):
    lo, hi = model.actuator_ctrlrange[act_id]
    return float(np.clip(value, lo, hi))

# ------------------------------------------------------------------
# Visualization settings
# ------------------------------------------------------------------
BODY_AXIS_LENGTH = 0.35
WORLD_AXIS_LENGTH = 0.5
BODY_AXIS_WIDTH = 0.012
WORLD_AXIS_WIDTH = 0.006

BODY_COLORS = {
    'x': np.array([1.0, 0.0, 0.0, 1.0]),
    'y': np.array([0.0, 1.0, 0.0, 1.0]),
    'z': np.array([0.0, 0.0, 1.0, 1.0]),
}
WORLD_COLORS = {
    'x': np.array([0.6, 0.2, 0.2, 0.5]),
    'y': np.array([0.2, 0.6, 0.2, 0.5]),
    'z': np.array([0.2, 0.2, 0.6, 0.5]),
}
AXES = [
    ('x', np.array([1.0, 0.0, 0.0])),
    ('y', np.array([0.0, 1.0, 0.0])),
    ('z', np.array([0.0, 0.0, 1.0])),
]

# ------------------------------------------------------------------
# Keyboard callback
# ------------------------------------------------------------------
def keyboard_callback(keycode):
    if keycode == 265:          # UP - forward
        data.ctrl[LEFT_ACT] = clamp_ctrl(LEFT_ACT, data.ctrl[LEFT_ACT] + CTRL_STEP)
        data.ctrl[RIGHT_ACT] = clamp_ctrl(RIGHT_ACT, data.ctrl[RIGHT_ACT] + CTRL_STEP)
    elif keycode == 264:        # DOWN - backward
        data.ctrl[LEFT_ACT] = clamp_ctrl(LEFT_ACT, data.ctrl[LEFT_ACT] - CTRL_STEP)
        data.ctrl[RIGHT_ACT] = clamp_ctrl(RIGHT_ACT, data.ctrl[RIGHT_ACT] - CTRL_STEP)
    elif keycode == 263:        # LEFT - rotate left
        data.ctrl[LEFT_ACT] = clamp_ctrl(LEFT_ACT, data.ctrl[LEFT_ACT] - CTRL_STEP)
        data.ctrl[RIGHT_ACT] = clamp_ctrl(RIGHT_ACT, data.ctrl[RIGHT_ACT] + CTRL_STEP)
    elif keycode == 262:        # RIGHT - rotate right
        data.ctrl[LEFT_ACT] = clamp_ctrl(LEFT_ACT, data.ctrl[LEFT_ACT] + CTRL_STEP)
        data.ctrl[RIGHT_ACT] = clamp_ctrl(RIGHT_ACT, data.ctrl[RIGHT_ACT] - CTRL_STEP)
    elif keycode == 32:         # SPACE - stop
        data.ctrl[LEFT_ACT] = 0.0
        data.ctrl[RIGHT_ACT] = 0.0


def draw_arrow(scn, origin, direction, length, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    end = origin + direction * length
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=rgba.astype(np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, origin, end)
    scn.ngeom += 1


def draw_label(scn, pos, text, rgba=(1.0, 1.0, 1.0, 1.0)):
    """Draw a floating 3D text label at `pos` using an invisible
    marker geom with its `label` field set. Works in the passive
    viewer since it's standard scene rendering, not a 2D overlay."""
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.001, 0.0, 0.0]),   # essentially invisible
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    geom.label = text
    scn.ngeom += 1


def matrix_rows(R):
    """Return the 3 rows of R as separate formatted strings."""
    return ["[ " + "  ".join(f"{R[i, j]: .3f}" for j in range(3)) + " ]" for i in range(3)]


# ------------------------------------------------------------------
# Launch the viewer
# ------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_callback) as viewer:
    print("Click INSIDE the 3D window first, then use ARROW KEYS to drive the robot.")
    print("SPACE stops the motors. The rotation matrix floats above the robot in the 3D view.\n")

    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)

        R = data.xmat[ROBOT_BODY_ID].reshape(3, 3).copy()
        body_pos = data.xpos[ROBOT_BODY_ID].copy()

        # --------------------------------------------------------
        # Redraw custom geoms every frame: axes + text labels
        # --------------------------------------------------------
        viewer.user_scn.ngeom = 0

        # Robot body frame axes
        for axis_name, axis_vec in AXES:
            world_dir = R @ axis_vec
            draw_arrow(viewer.user_scn, body_pos, world_dir,
                       BODY_AXIS_LENGTH, BODY_AXIS_WIDTH, BODY_COLORS[axis_name])

        # Fixed world frame axes
        for axis_name, axis_vec in AXES:
            draw_arrow(viewer.user_scn, np.zeros(3), axis_vec,
                       WORLD_AXIS_LENGTH, WORLD_AXIS_WIDTH, WORLD_COLORS[axis_name])

        # Rotation matrix as floating text above the robot
        label_origin = body_pos + np.array([0.0, 0.0, 0.6])  # height above robot
        row_spacing = 0.12

        draw_label(viewer.user_scn, label_origin + np.array([0, 0, row_spacing * 3]),
                   f"R (body -> world)  t={data.time:5.2f}s")

        for i, row_text in enumerate(matrix_rows(R)):
            row_pos = label_origin + np.array([0, 0, row_spacing * (2 - i)])
            draw_label(viewer.user_scn, row_pos, row_text)

        viewer.sync()

        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
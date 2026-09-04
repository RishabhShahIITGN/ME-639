"""
01_rotation_sandbox.py -- HW1 Part 2, Task 1: does rotation order matter?

This script lets you queue up a sequence of elemental rotations (about x, y,
or z), where EACH STEP independently chooses to be about the current body
frame or the fixed space frame. It supports two modes:

  1. SCRIPTED MODE (RUN_INTERACTIVE_MODE = False): runs the three
     pre-defined test cases (A, B, C) exactly as before -- numeric preview,
     then animated in the viewer, one rotation at a time.

  2. INTERACTIVE MODE (RUN_INTERACTIVE_MODE = True): opens the viewer and
     lets you drive the rotation live from the keyboard:
         1 / 2 / 3  -> rotate about the FIXED  (space) x / y / z axis
         4 / 5 / 6  -> rotate about the CURRENT (body)  x / y / z axis
         R          -> reset orientation to identity
     Each keypress applies one fixed-size rotation step (see
     INTERACTIVE_ANGLE_STEP) and prints the updated rotation matrix, so you
     can build up any sequence -- e.g. "1 2 3" for three fixed-frame
     rotations about x, y, z, followed by "4 5 6" for three current-frame
     rotations about x, y, z -- and watch it live in the viewer. This is
     the fastest way to explore non-commutativity by hand before/while
     recording.

Recording tip for the required deliverable: record the fixed-frame
experiment and the current-frame experiment (Test Case B, or the
equivalent interactive keypresses) as two separate clips, then place them
side by side in your video editor (or screen-record two terminal/viewer
windows simultaneously).
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from utils import Rx, Ry, Rz, set_body_orientation

np.set_printoptions(precision=4, suppress=True)

MODEL_PATH = "../model/asymmetric_body.xml"
STEP_DELAY = 3.0
FINAL_PAUSE = 4.0
RUN_TEST_CASE_A = True
RUN_TEST_CASE_B = True
RUN_TEST_CASE_C = True

# Set this to True to skip the scripted test cases and instead drive
# rotations live from the keyboard (1-3 = fixed frame x/y/z,
# 4-6 = current/body frame x/y/z, R = reset).
RUN_INTERACTIVE_MODE = False

# Degrees applied per keypress in interactive mode.
INTERACTIVE_ANGLE_STEP = np.radians(15)


# ---------------------------------------------------------------------------
# Test-case sequences. Each rotation is a dict: axis in {"x","y","z"},
# angle in radians, frame in {"fixed","current"}. Edit freely -- this list
# editing IS the sandbox.
# ---------------------------------------------------------------------------

TEST_CASE_A_SEQUENCE = [
    {"axis": "x", "angle": np.radians(30), "frame": "fixed"},
    {"axis": "y", "angle": np.radians(135), "frame": "fixed"},
    {"axis": "z", "angle": np.radians(-120), "frame": "fixed"},
]

TEST_CASE_B_FIXED_SEQUENCE = [
    {"axis": "x", "angle": np.radians(90), "frame": "fixed"},
    {"axis": "y", "angle": np.radians(90), "frame": "fixed"},
]

TEST_CASE_B_CURRENT_SEQUENCE = [
    {"axis": "x", "angle": np.radians(90), "frame": "current"},
    {"axis": "y", "angle": np.radians(90), "frame": "current"},
]

# Mixed frame-choice-per-step example: first rotation about the fixed
# space x-axis, second rotation about whatever the body y-axis has
# *become* after step 1. This is the general case the assignment
# describes -- frame is chosen independently at each step.
TEST_CASE_C_MIXED_SEQUENCE = [
    {"axis": "x", "angle": np.radians(90), "frame": "fixed"},
    {"axis": "y", "angle": np.radians(90), "frame": "current"},
]

# Keycodes for interactive mode. GLFW digit keycodes match ASCII codes for
# '0'-'9', so comparing against ord(...) works directly against the
# keycode the viewer's key_callback receives.
INTERACTIVE_KEY_MAP = {
    ord("1"): {"axis": "x", "frame": "fixed"},
    ord("2"): {"axis": "y", "frame": "fixed"},
    ord("3"): {"axis": "z", "frame": "fixed"},
    ord("4"): {"axis": "x", "frame": "current"},
    ord("5"): {"axis": "y", "frame": "current"},
    ord("6"): {"axis": "z", "frame": "current"},
}
INTERACTIVE_RESET_KEY = ord("R")


def _format_rotation(axis, angle, frame):
    return f"{np.degrees(angle):.1f} degrees about {frame} {axis.upper()}"


def _apply_rotation(R, rotation):
    axis = rotation["axis"]
    angle = rotation["angle"]
    frame = rotation["frame"]
    rotation_functions = {"x": Rx, "y": Ry, "z": Rz}
    if axis not in rotation_functions or frame not in ("fixed", "current"):
        raise ValueError(f"Invalid rotation step: {rotation}")

    R_step = rotation_functions[axis](angle)
    if frame == "fixed":
        # Fixed-space axes are outside the current body orientation, so
        # the new rotation acts on the left.
        return R_step @ R

    # Current-body axes move with the body, so the new rotation acts
    # inside the current orientation and is multiplied on the right.
    return R @ R_step


def compose_sequence(sequence, verbose=True):
    """Numerically predict the final orientation, no viewer needed.

    Use this to sanity-check a sequence against your Problem 3 hand
    derivation before spending time animating it.
    """
    R = np.eye(3)
    for step_number, rotation in enumerate(sequence, start=1):
        R = _apply_rotation(R, rotation)
        if verbose:
            print(
                f"  Step {step_number}: "
                f"{_format_rotation(rotation['axis'], rotation['angle'], rotation['frame'])}"
            )
            print(R)
    return R


def apply_sequence(model, data, viewer, name, sequence):
    """Animate a sequence in the MuJoCo viewer, one rotation at a time."""
    print(f"\n{name}")
    print("=" * len(name))

    print("--- RESETTING TO IDENTITY ---")
    R = np.eye(3)
    print("Initial rotation matrix:")
    print(R)

    set_body_orientation(data, R)
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(3.0)

    for step_number, rotation in enumerate(sequence, start=1):
        R = _apply_rotation(R, rotation)
        print(
            f"  Step {step_number}: "
            f"{_format_rotation(rotation['axis'], rotation['angle'], rotation['frame'])}"
        )
        print(R)
        set_body_orientation(data, R)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(STEP_DELAY)

    print("Final rotation matrix:")
    print(R)
    time.sleep(FINAL_PAUSE)
    return R


def run_scripted_test_cases(model, data, viewer):
    """Run Test Cases A, B, C exactly as before."""

    # --- Fast numeric preview of Test Case B before animating anything ---
    # Confirms the "different frame => different final orientation" claim
    # mathematically, matching your Problem 3 derivation.
    print("Numeric preview (no viewer) -- Test Case B")
    print("-" * 42)
    print("Fixed-frame sequence:")
    R_fixed_preview = compose_sequence(TEST_CASE_B_FIXED_SEQUENCE)
    print("Current-frame sequence:")
    R_current_preview = compose_sequence(TEST_CASE_B_CURRENT_SEQUENCE)
    print(
        "Matrices diverge? "
        f"{not np.allclose(R_fixed_preview, R_current_preview)}"
    )

    if RUN_TEST_CASE_A:
        apply_sequence(
            model, data, viewer, "Test Case A: Problem 3 reference", TEST_CASE_A_SEQUENCE
        )

    if RUN_TEST_CASE_B and viewer.is_running():
        R_fixed = apply_sequence(
            model, data, viewer, "Test Case B, Experiment 1: fixed frame", TEST_CASE_B_FIXED_SEQUENCE
        )
        if viewer.is_running():
            R_current = apply_sequence(
                model, data, viewer,
                "Test Case B, Experiment 2: current frame",
                TEST_CASE_B_CURRENT_SEQUENCE,
            )
            print("\nTest Case B comparison")
            print("Fixed-frame final matrix:")
            print(R_fixed)
            print("Current-frame final matrix:")
            print(R_current)
            is_different = not np.allclose(R_fixed, R_current)
            print(f"CONCLUSION: Fixed and Current frame matrices diverge? {is_different}")

    if RUN_TEST_CASE_C and viewer.is_running():
        apply_sequence(
            model, data, viewer,
            "Test Case C: mixed per-step frame choice",
            TEST_CASE_C_MIXED_SEQUENCE,
        )

    while viewer.is_running():
        viewer.sync()
        time.sleep(1 / 60)


def run_interactive_mode(model, data):
    """
    Manual keyboard-driven rotation mode.

    Opens its own passive viewer (key_callback can only be registered at
    launch time) and lets you build up a rotation sequence live:

        1 / 2 / 3  -> rotate about the FIXED  (space) x / y / z axis
        4 / 5 / 6  -> rotate about the CURRENT (body)  x / y / z axis
        R          -> reset orientation to identity

    e.g. pressing "1 2 3" applies three fixed-frame rotations about
    x, y, z in that order; then pressing "4 5 6" applies three
    current-frame (body) rotations about x, y, z on top of that.
    Each keypress prints the step taken and the running rotation matrix,
    so you can read off the exact sequence you demonstrated for your
    recording.
    """
    print("\nInteractive mode")
    print("=================")
    print(f"  Rotation step per keypress: {np.degrees(INTERACTIVE_ANGLE_STEP):.1f} degrees")
    print("  1 / 2 / 3  -> rotate about FIXED   x / y / z")
    print("  4 / 5 / 6  -> rotate about CURRENT x / y / z")
    print("  R          -> reset to identity")
    print("  Close the viewer window to exit.\n")

    # Mutable holder so the key_callback closure can update the running
    # rotation matrix (key_callback itself can't return a value).
    state = {"R": np.eye(3)}

    def key_callback(keycode):
        if keycode in INTERACTIVE_KEY_MAP:
            step = INTERACTIVE_KEY_MAP[keycode]
            rotation = {
                "axis": step["axis"],
                "angle": INTERACTIVE_ANGLE_STEP,
                "frame": step["frame"],
            }
            state["R"] = _apply_rotation(state["R"], rotation)
            print(f"  {_format_rotation(rotation['axis'], rotation['angle'], rotation['frame'])}")
            print(state["R"])
            set_body_orientation(data, state["R"])
            mujoco.mj_forward(model, data)
        elif keycode == INTERACTIVE_RESET_KEY:
            state["R"] = np.eye(3)
            print("  Reset to identity")
            print(state["R"])
            set_body_orientation(data, state["R"])
            mujoco.mj_forward(model, data)

    set_body_orientation(data, state["R"])
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.sync()
        print("Viewer open. Press keys 1-6 (and R to reset). Close the window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    if RUN_INTERACTIVE_MODE:
        run_interactive_mode(model, data)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nViewer open. Close the window to exit.")
        run_scripted_test_cases(model, data, viewer)


if __name__ == "__main__":
    main()
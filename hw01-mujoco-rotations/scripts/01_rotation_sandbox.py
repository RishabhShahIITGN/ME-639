"""
01_rotation_sandbox.py -- HW1 Part 2, Task 1: does rotation order matter?

STARTER CODE. The model loading, viewer, and simulation loop below
are complete and working -- run this file as-is and you should see
the asymmetric dart sitting in the viewer. Your job is to fill in
the TODOs so that:

  1. The user can queue up a sequence of elemental rotations
     (about x, y, or z), each one EITHER about the current body
     frame OR about the fixed space frame (their choice).
  2. The dart's orientation updates to reflect that sequence.
  3. You can run the SAME sequence of angles twice -- once
     "current frame" and once "fixed frame" -- and see (and
     screen-record) that the final orientation is visibly
     different, exactly as you proved symbolically in HW1
     Problem 3 (Lynch & Park Ch.3 Ex.3.4-style reasoning) and
     the "Composition of Rotations" lecture derivation.

This is intentionally a plain script, not a GUI app -- editing the
test-case sequence lists below and re-running is a perfectly good
"sandbox." A slider UI is a nice-to-have, not a requirement. Use AI
freely here; document what you asked it for in your AI Use Note.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from utils import Rx, Ry, Rz, set_body_orientation

# Configure numpy formatting after all imports
np.set_printoptions(precision=4, suppress=True)

MODEL_PATH = "../model/asymmetric_body.xml"
STEP_DELAY = 3.0
FINAL_PAUSE = 4.0
RUN_TEST_CASE_A = True
RUN_TEST_CASE_B = True


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

    # Current-body axes move with the body, so the new rotation acts inside
    # the current orientation and is multiplied on the right.
    return R @ R_step


def compose_sequence(sequence):
    """Return the final orientation after applying a rotation sequence."""
    R = np.eye(3)
    for step_number, rotation in enumerate(sequence, start=1):
        R = _apply_rotation(R, rotation)
        print(
            f"  Step {step_number}: "
            f"{_format_rotation(rotation['axis'], rotation['angle'], rotation['frame'])}"
        )
        print(R)
    return R


def apply_sequence(model, data, viewer, name, sequence):
    print(f"\n{name}")
    print("=" * len(name))
    
    print("--- RESETTING TO IDENTITY ---")
    R = np.eye(3)
    print("Initial rotation matrix:")
    print(R)
    
    set_body_orientation(data, R)
# ... (keep the rest the same)
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


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer open. Close the window to exit.")

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

        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()

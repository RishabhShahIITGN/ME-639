"""
01_rotation_sandbox.py -- HW1 Part 2, Task 1: does rotation order matter?

Fill-in complete. This script lets you queue up a sequence of elemental
rotations (about x, y, or z), where EACH STEP independently chooses to be
about the current body frame or the fixed space frame. It then:

  1. Predicts the final orientation numerically (fast, no viewer) so you
     can sanity-check against your Problem 3 hand derivation before
     waiting through the animation.
  2. Animates the same sequence in the MuJoCo viewer, applying rotations
     one at a time so you can screen-record the motion.
  3. Runs three test cases:
       A: the exact Problem 3 sequence (fixed frame only) as a reference.
       B: the SAME two rotation angles applied once entirely in the fixed
          frame and once entirely in the current frame -- final
          orientations visibly diverge, proving composition order matters.
       C: a single sequence that MIXES fixed- and current-frame steps,
          demonstrating the per-step frame choice the assignment asks for.

This is a plain script, not a GUI app -- editing the test-case sequence
lists below and re-running is a perfectly good "sandbox." A slider UI is
a nice-to-have, not a requirement.

Recording tip: record Test Case B's two experiments as two separate clips,
then place them side by side in your video editor (or screen-record two
terminal/viewer windows simultaneously) for the "side by side" deliverable.
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


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nViewer open. Close the window to exit.")

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


if __name__ == "__main__":
    main()
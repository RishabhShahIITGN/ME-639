"""Load the workspace-sized HEAL pick-and-place scene in the MuJoCo viewer."""

from pathlib import Path

import mujoco
from mujoco.viewer import launch_passive

HERE = Path(__file__).resolve().parent
DESC = HERE.parent.parent / "ITR_mujoco_fk_lab" / "robot_descriptions"
XML = DESC / "heal_pick_place.xml"


def main():
    if not XML.exists():
        raise FileNotFoundError(
            f"{XML} missing — run python3 lab/lab-03/table_design.py first"
        )
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    # Hold the arm against gravity; the cube is a free joint and should settle on the table.
    arm_acts = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in ("turret", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3")
    ]
    mujoco.mj_forward(model, data)
    with launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            for aid in arm_acts:
                if aid < 0:
                    continue
                jnt_dof = model.actuator_trnid[aid][0]
                data.ctrl[aid] = data.qfrc_bias[jnt_dof]
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()

# Lab Overview

## Lab 01

- `spawn_waffle.py` — spawn the Waffle robot in MuJoCo.
- `spawn_quadrotor.py` — spawn the quadrotor in MuJoCo.

<b>Video demonstration:</b> <a href="https://youtu.be/5-cM8qot2qo?si=QACPg403qRGEix1p" target="_blank" rel="noopener noreferrer">Lab 01 — Waffle and quadrotor MuJoCo simulations</a>

### Run and operate the Waffle simulation

From the repository root, activate the project virtual environment and launch the script from the `lab` directory. Running it from `lab` is important because the model path in the script is relative to the current working directory.

```bash
cd lab
source ../venv/bin/activate
python3 spawn_waffle.py
```

The MuJoCo viewer opens with the Waffle robot. Click inside the viewer to give it keyboard focus, then use:

| Key | Action |
|---|---|
| Up arrow | Drive forward |
| Down arrow | Drive backward |
| Left arrow | Turn left |
| Right arrow | Turn right |
| Space | Stop both wheels |

The script detects the wheel actuators by name, limits commands to their supported control ranges, and prints the selected robot body and actuator information in the terminal. The 3D view shows the robot body axes (solid red X, green Y, blue Z), world axes (muted colors), and the robot body's rotation matrix and simulation time above the robot. Close the viewer window to stop the simulation.

### Run and operate the quadrotor simulation

From the repository root, activate the project virtual environment and launch the script from the `lab` directory. The model path is relative to that working directory.

```bash
cd lab
source ../venv/bin/activate
python3 spawn_quadrotor.py
```

The MuJoCo viewer starts the Skydio X2 from its `hover` keyframe. Click inside the viewer to give it keyboard focus. Keyboard presses nudge a velocity or yaw-rate command; the command decays smoothly, so one press gives a short glide and repeated key presses build or maintain motion.

| Key | Action |
|---|---|
| Up / Down arrows | Move forward / backward |
| Left / Right arrows | Strafe left / right |
| W / S | Ascend / descend |
| A / D | Yaw left / right |
| Space | Cancel movement commands and hold position |
| R | Reset to the hover keyframe and clear commands |
| F | Toggle the follow camera |

The viewer displays body-frame axes (solid red X, green Y, blue Z), world-frame axes (muted colors), and floating telemetry including attitude, rotation matrix, position, velocity, angular velocity, motor commands, and the controls reminder. The terminal prints model and motor information plus periodic flight telemetry. Close the viewer window to stop the simulation; if needed, use Ctrl+C in the terminal.

## Lab 02

- `spawn_heal.py` — spawn the HEAL robot in MuJoCo.
- `spawn_franka.py` — spawn the Franka robot in MuJoCo.

**Video demonstration:** [Lab 02 — HEAL and Franka MuJoCo simulations](https://youtu.be/GwFLrIUo620?si=YbysZpMFHA-QoY82)

### Run and operate the HEAL simulation

From the repository root, activate the project virtual environment, then run the script from the `lab` directory:

```bash
cd lab
source ../venv/bin/activate
python3 spawn_heal.py
```

The script opens the MuJoCo viewer and a separate Tkinter control window. It starts with the arm targeting the **Home** pose. Use the six joint sliders to change the target angles; the displayed limits are specific to the robot model.

| Control | Action |
|---|---|
| Joint sliders | Set the six joint target angles in degrees |
| **Home** | Move to the predefined home pose |
| **Zero** | Target zero angle for all joints |
| **Stretch** | Move to the predefined stretch pose |
| **Demo Trajectory** | Toggle the built-in time-varying joint motion |
| **Follow Camera** | Keep the viewer camera centered on the end-effector |

The arm uses a PD joint controller with gravity compensation. The Tkinter panel compares analytical forward kinematics with MuJoCo's simulated end-effector pose and reports position error (mm), orientation error (degrees), end-effector pose, TCP position, and end-effector speed from the Jacobian. In the 3D viewer, colored arrows show body and world coordinate axes, a yellow marker shows the analytical FK position, and a floating HUD displays the pose and FK errors. Similar telemetry is also printed periodically in the terminal.

Keep both windows open while operating the simulation. Close the Tkinter control window or the MuJoCo viewer to end the simulation. The script requires a graphical desktop session for both windows.

### Run and operate the Franka Panda simulation

From the repository root, activate the project virtual environment, then launch the script from the `lab` directory:

```bash
cd lab
source ../venv/bin/activate
python3 spawn_franka.py
```

The simulation opens a MuJoCo viewer and a separate Tkinter control window. The arm starts targeting its **Home** pose with the gripper open. Use the seven joint sliders to set joint targets in degrees and the gripper slider to adjust its opening (0–40 mm).

| Control | Action |
|---|---|
| Joint sliders | Set target angles for the seven arm joints |
| Gripper slider | Adjust the gripper opening from 0 to 40 mm |
| **Home** | Move to the predefined home pose |
| **Zero** | Target zero angle for all seven joints |
| **Ready** | Move to the predefined ready pose |
| **Demo Trajectory** | Toggle the built-in time-varying joint motion |
| **Follow Camera** | Keep the viewer camera centered on the end-effector |

The arm uses a PD joint controller with gravity compensation. The gripper is controlled independently by its slider. The Tkinter panel compares analytical forward kinematics with MuJoCo's simulated end-effector pose and displays position error (mm), orientation error (degrees), end-effector pose, and end-effector speed computed from the Jacobian. In the 3D viewer, colored arrows show body and world coordinate axes, a yellow marker indicates the analytical FK position, and a floating HUD shows pose and FK diagnostics. Periodic pose and error telemetry is also printed in the terminal.

Keep both windows open while operating the simulation. Close the Tkinter control window or the MuJoCo viewer to end the simulation. The script requires a graphical desktop session for both windows.

## Lab 03 — Autonomous Pick-and-Place Manipulation with HEAL & Franka Emika Panda

A physics-accurate pick-and-place manipulation pipeline in [MuJoCo](https://mujoco.org/), benchmarking Differential Inverse Kinematics (IK) solvers across both the 6-DOF Addverb HEAL and 7-DOF redundant Franka Emika Panda robotic arms.

📄 **Technical Report:** [`lab-03/Tutorial_4___itr.pdf`](lab-03/Tutorial_4___itr.pdf)  
📁 **Detailed Documentation & Code:** See the [`lab-03/`](lab-03/) directory and [`lab-03/README.md`](lab-03/README.md).

### Video Demonstrations

- **HEAL 6-DOF (Mink IK):** [Watch Video](https://youtu.be/h-dL-vL3UDM?si=1semsxsM4L094ZeF)
- **HEAL 6-DOF (Custom DLS):** [Watch Video](https://youtu.be/FMAL-rnbg08?si=yBDw17tpFTI-orKf)
- **HEAL 6-DOF (Custom QP):** [Watch Video](https://youtu.be/oaHxvoVMIUw?si=IdQEXkaOgRmZANhr)
- **Franka 7-DOF (DLS Failure Analysis):** [Watch Video](https://youtu.be/ERtZ837I5WE)

### Quick Run

From the repository root, activate the project virtual environment and run interactive pick-and-place episodes:

```bash
cd lab/lab-03
source ../../venv/bin/activate

# HEAL 6-DOF Pick-and-Place:
python3 scripts/ik_pick_place.py          # Mink
python3 scripts/ik_dls_pick_place.py      # Custom DLS
python3 scripts/ik_qp_pick_place.py       # Custom QP

# Franka 7-DOF Pick-and-Place:
python3 scripts_franka/ik_pick_place.py      # Mink
python3 scripts_franka/ik_dls_pick_place.py  # Custom DLS
python3 scripts_franka/ik_qp_pick_place.py   # Custom QP
```


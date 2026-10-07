# Lab Overview

## Lab 01

- `spawn_waffle.py` — spawn the Waffle robot in MuJoCo.
- `spawn_quadrotor.py` — spawn the quadrotor in MuJoCo.

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

## Lab 03

See the [`lab-03/`](lab-03/) folder for the Lab 03 materials, documentation, and implementation.

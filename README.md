<h1 align="center" style="font-size: 2.5em;">ME-639: Introduction to Robotics</h1>

<p align="center">
  <b>Department of Mechanical Engineering, Indian Institute of Technology Gandhinagar (IITGN)</b><br>
  <b>Author:</b> Rishabh Shah (<a href="https://github.com/RishabhShahIITGN">@RishabhShahIITGN</a>)<br>
  <i>A comprehensive academic portfolio of modern robotics algorithms, physics-accurate simulations, and physical manipulator implementations across MuJoCo and ROS 2 Humble.</i>
</p>

<p align="center">
  <a href="#-video-demonstrations-index"><b>Video Demonstrations</b></a> •
  <a href="#1-3d-kinematics--spatial-rotations"><b>3D Rotations</b></a> •
  <a href="#2-laboratory-simulations"><b>Labs 01–03</b></a> •
  <a href="#3-capstone-projects--the-tri-wizard-tournament"><b>Capstone Projects</b></a> •
  <a href="#4-environment-setup--quickstart"><b>Quickstart</b></a>
</p>

---

## 🎥 Video Demonstrations Index

All laboratory simulations, mathematical visualizers, and physical hardware experiments are documented with video demonstrations:

| Module | Topic / Platform | Demonstration Link | Focus Area |
|---|---|:---:|---|
| **Rigid Body Kinematics** | 3D WebGL Rotation Visualizer | [Watch Video](https://youtu.be/QE0hZ7LaZYM) | Real-time Three.js Euler angle matrix composition & coordinate transformations ([Live App](https://rishabhshahiitgn.github.io/rotation-visualizer/)) |
| **Lab 01** | TurtleBot3 & Skydio X2 (MuJoCo) | [Watch Video](https://youtu.be/5-cM8qot2qo?si=QACPg403qRGEix1p) | Differential drive mobile robot kinematics & 6-DOF aerial vehicle physics |
| **Lab 02** | Addverb HEAL & Franka Panda (MuJoCo) | [Watch Video](https://youtu.be/GwFLrIUo620?si=YbysZpMFHA-QoY82) | Analytical vs. simulated Forward Kinematics, Jacobians, and Tkinter GUI slider control |
| **Lab 03 (HEAL)** | Mink Differential IK | [Watch Video](https://youtu.be/h-dL-vL3UDM?si=1semsxsM4L094ZeF) | 7-phase pick-and-place with OSQP-based differential IK & collision limits |
| **Lab 03 (HEAL)** | Custom Damped Least Squares (DLS) | [Watch Video](https://youtu.be/FMAL-rnbg08?si=yBDw17tpFTI-orKf) | High-speed analytical Levenberg–Marquardt closed-loop IK (**2.5× faster** than Mink) |
| **Lab 03 (HEAL)** | Custom Quadratic Programming (QP) | [Watch Video](https://youtu.be/oaHxvoVMIUw?si=IdQEXkaOgRmZANhr) | Box-constrained QP strictly satisfying velocity and physical joint limits |
| **Lab 03 (Franka)** | Franka DLS Failure Case Analysis | [Watch Video](https://youtu.be/ERtZ837I5WE) | Diagnostic demonstration of 7-DOF redundant null-space elbow swivel during release |
| **Capstone Task 1** | Peg Sorting (OpenMANIPULATOR-X) | [Watch Video](https://youtu.be/XiwHk8Va8t8) | Autonomous RGB-D shape classification, collision planning, and physical insertion |
| **Capstone Task 2** | Wire Trace Forward Run (Addverb HEAL) | [Watch Video](https://youtu.be/hoY5CwrO2DI) | Autonomous 3D wire tracing with conductive loop via DLS-IK velocity control |
| **Capstone Task 2** | Wire Trace Backward Run (Addverb HEAL) | [Watch Video](https://youtu.be/m1hTtmguCew) | Reverse trajectory execution with zero touch events |

---

## Repository Architecture

```
ME-639/
├── rotation-visualizer/         # Interactive 3D WebGL Rigid Body Rotation Laboratory
├── hw01-mujoco-rotations/       # MuJoCo 3D Rotation Sandbox & Lie Algebra verification
├── lab/                         # Laboratory simulations (MuJoCo physics engine)
│   ├── spawn_waffle.py          # Lab 01: TurtleBot3 Waffle Pi simulation
│   ├── spawn_quadrotor.py       # Lab 01: Skydio X2 quadrotor 6-DOF flight control
│   ├── spawn_heal.py            # Lab 02: Addverb HEAL (6-DOF) FK & Jacobian GUI
│   ├── spawn_franka.py          # Lab 02: Franka Emika Panda (7-DOF) FK & Jacobian GUI
│   ├── lab-03/                  # Lab 03: Autonomous Pick-and-Place Manipulation
│   │   ├── README.md            # Lab 03 detailed report & IK documentation
│   │   ├── Tutorial_4___itr.pdf # Formal peer-reviewed research paper
│   │   ├── scripts/             # HEAL 6-DOF pick-and-place implementation
│   │   ├── scripts_franka/      # Franka Panda 7-DOF pick-and-place implementation
│   │   └── logs/                # Benchmark figures & episode JSON diagnostics
│   └── ITR_mujoco_fk_lab/       # Robot descriptions, MJCF XML models, and meshes
├── project/                     # Capstone Projects — The Tri-Wizard Tournament
│   ├── README.md                # Comprehensive documentation for both capstone trials
│   ├── RUN_STEPS.md             # Operator checklist for autonomous peg sorting
│   └── peg_ws/                  # Active ROS 2 Humble workspace
│       ├── src/open_manipulator # ROBOTIS OpenMANIPULATOR-X hardware driver (ros2_control)
│       └── src/tri-wizard-peg-trial # Peg sorting ROS 2 package (vision, planning, control)
├── mujoco_menagerie/            # Google DeepMind verified robot models (Skydio X2, Panda, etc.)
└── robotis_mujoco_menagerie/    # ROBOTIS verified robot models (TurtleBot3, OMX, OP3)
```

---

## 1. 3D Kinematics & Spatial Rotations

### Interactive WebGL Rotation Visualizer (`rotation-visualizer/`)
An interactive, browser-based 3D visualization laboratory built with Three.js (r128) designed to bridge abstract matrix algebra and physical spatial orientations.

* **Live Deployment:** [rishabhshahiitgn.github.io/rotation-visualizer](https://rishabhshahiitgn.github.io/rotation-visualizer/)
* **Demonstration Video:** [Watch on YouTube](https://youtu.be/QE0hZ7LaZYM)
* **Mathematical Features:**
  - Standard aerospace **$X$-$Y$-$Z$ Intrinsic Euler Angles** ($\text{Roll } \phi \to \text{Pitch } \theta \to \text{Yaw } \psi$) composed via post-multiplication:
    $$R = R_x(\phi) \cdot R_y(\theta) \cdot R_z(\psi)$$
  - Real-time point transformation engine: $P_{\text{world}} = R \cdot P_{\text{body}}$.
  - Continuous matrix validation: Orthogonality ($R^T R = I$) and proper rotation determinant ($\det(R) = +1$).
  - Conversion between Rotation Matrices, Unit Quaternions ($q_w, q_x, q_y, q_z$), and Axis-Angle representations ($\theta, \hat{\omega}$).
  - Step-by-step sequential analysis mode demonstrating the non-commutative nature of 3D rotations ($R_1 R_2 \neq R_2 R_1$).

### MuJoCo Rotation Sandbox (`hw01-mujoco-rotations/`)
A MuJoCo simulation environment (`model/asymmetric_body.xml`) evaluating 3D rotation mechanics:
* Numerical and visual demonstration of **current body-frame** vs. **fixed space-frame** rotation compositions.
* Lie algebraic verification of skew-symmetric properties for $so(3)$:
  $$\hat{\omega}^T = -\hat{\omega}, \quad \hat{\omega} v = \omega \times v, \quad R \hat{\omega} R^T = \widehat{R\omega}$$

---

## 2. Laboratory Simulations

### Lab 01 — Mobile Robotics & Aerial Flight
* **TurtleBot3 Waffle Pi (`spawn_waffle.py`):** Interactive differential-drive mobile robot simulation with wheel velocity control and real-time heading telemetry.
* **Skydio X2 Quadrotor (`spawn_quadrotor.py`):** 6-DOF autonomous aerial vehicle flight from keyframe hover, featuring attitude telemetry, thrust damping, and chase camera tracking.
* 🎥 **Demonstration:** [Watch Video (Lab 01)](https://youtu.be/5-cM8qot2qo?si=QACPg403qRGEix1p)

---

### Lab 02 — Forward Kinematics & Manipulator Jacobians
* **6-DOF Addverb HEAL (`spawn_heal.py`) & 7-DOF Franka Panda (`spawn_franka.py`):**
  - Live Tkinter joint-slider GUI controlling physical joint angles.
  - Analytical Forward Kinematics derived from DH / body transform chains compared against MuJoCo simulation positions:
    $$\Delta p = \|p_{\text{analytical}} - p_{\text{simulated}}\|_2 < 10^{-4}\text{ m}$$
  - Spatial Geometric Jacobian computation ($J \in \mathbb{R}^{6 \times n}$) and instantaneous end-effector Cartesian velocity tracking:
    $$\dot{x} = J(q) \dot{q}$$
  - Gravity compensation feedforward via `qfrc_bias`.
* 🎥 **Demonstration:** [Watch Video (Lab 02)](https://youtu.be/GwFLrIUo620?si=YbysZpMFHA-QoY82)

---

### Lab 03 — Autonomous Pick-and-Place with Differential IK
A comprehensive manipulation pipeline in MuJoCo benchmarking Differential Inverse Kinematics solvers across both the 6-DOF HEAL and 7-DOF redundant Franka Panda arms equipped with a Robotiq 2F-85 parallel-jaw gripper.

📄 **Full Technical Paper:** [`lab/lab-03/Tutorial_4___itr.pdf`](lab/lab-03/Tutorial_4___itr.pdf)  
📁 **Full Lab 03 Documentation:** [`lab/lab-03/README.md`](lab/lab-03/README.md)

#### Table & Workspace Optimization
A custom ergonomic table ($0.34\text{ m} \times 0.60\text{ m} \times 0.04\text{ m}$ at surface height $z = 0.20\text{ m}$) was derived using Monte Carlo workspace reachability and manipulability maps (`table_design.py`), placing top-down grasps directly inside the high-dexterity core ($z \approx 0.36\text{ m}$).

#### 7-Phase Finite State Machine
$$\text{[APPROACH]} \to \text{[DESCEND]} \to \text{[GRASP]} \to \text{[LIFT]} \to \text{[TRANSIT]} \to \text{[LOWER]} \to \text{[RELEASE]}$$

#### Inverse Kinematics Formulations
1. **Mink (Baseline Optimizer):** Differential IK formulated as an OSQP optimization problem with quadratic task tracking, velocity bounds, and linear obstacle distance inequality constraints ($C \Delta q \le d$).
2. **Custom Damped Least Squares (DLS):** Closed-loop Levenberg–Marquardt solver with task error clamping ($e_{\text{max}} = 0.02$) and null-space posture projection for 7-DOF redundancy resolution:
   $$\dot{q} = J_{DLS}^\dagger \dot{x}_d + \left(I_n - J_{DLS}^\dagger J\right) k_{\text{null}} (q_{\text{def}} - q)$$
3. **Custom Quadratic Programming (QP):** Box-constrained QP strictly satisfying hardware velocity saturation ($\pm 2.0\text{ rad/s}$) and joint limits via regularized positive-definite Hessian $P = J^T J + \lambda I_n$.

#### Quantitative Benchmark Results (Seed 19)

| Robot Platform | IK Solver | Success Rate | Mean Planning Time | Mean Wall Time | Min Clearance |
|---|---|:---:|:---:|:---:|:---:|
| **HEAL (6-DOF, 250 Episodes)** | **Mink** | **100.0%** (250/250) | $0.753\text{ s}$ | $9.15\text{ s}$ | $+1.39\text{ mm}$ |
| **HEAL (6-DOF, 250 Episodes)** | **Custom DLS** | **100.0%** (250/250) | **$0.299\text{ s}$ (⚡ 2.5× faster)** | **$8.51\text{ s}$** | **$+1.57\text{ mm}$** |
| **HEAL (6-DOF, 250 Episodes)** | **Custom QP** | **100.0%** (250/250) | $0.585\text{ s}$ (⚡ 1.3× faster) | $9.15\text{ s}$ | **$+1.57\text{ mm}$** |
| **Franka (7-DOF, 30 Episodes)** | **Mink** | **100.0%** (30/30) | $1.229\text{ s}$ | **$206.59\text{ s}$** | **$+1.58\text{ mm}$** |
| **Franka (7-DOF, 30 Episodes)** | **Custom DLS** | 93.3% (28/30) | **$1.166\text{ s}$** | $272.77\text{ s}$ | $-0.031\text{ mm}$ (grazing) |
| **Franka (7-DOF, 30 Episodes)** | **Custom QP** | 96.7% (29/30) | $1.275\text{ s}$ | $246.17\text{ s}$ | $-0.025\text{ mm}$ (grazing) |

---

## 3. Capstone Projects — The Tri-Wizard Tournament

📁 **Full Project Documentation:** [`project/README.md`](project/README.md)

### Tri-Wizard Task 1 — Autonomous Peg Sorting (OpenMANIPULATOR-X)
* **Standalone Refined Repo:** [`RishabhShahIITGN/peg-sorting-OMX`](https://github.com/RishabhShahIITGN/peg-sorting-OMX)
* 🎥 **Demonstration Video:** [Watch on YouTube](https://youtu.be/XiwHk8Va8t8)
* **Hardware:** ROBOTIS OpenMANIPULATOR-X (4-DOF + 1-DOF gripper) + 1× Intel RealSense D435 RGB-D camera.
* **Pipeline:**
  1. Live RealSense capture with ArUco extrinsic camera calibration ($T_{\text{base}}^{\text{cam}}$).
  2. HSV color segmentation & morphological filtering for 4 colored pegs (Circle, Triangle, Square, Hexagon) and board sockets.
  3. 3D point cloud back-projection, top-face footprint extraction, polygon approximation, and yaw estimation.
  4. Collision-aware 3D motion planning routing around previously placed pegs (modeled as dynamic obstacle cylinders).
  5. Analytic inverse kinematics execution via `ros2_control` and post-insertion camera verification.

### Tri-Wizard Task 2 — The Buzz Wire Trace Challenge (Addverb HEAL)
* **Standalone Project Repo:** [`RishabhShahIITGN/The-Wire-Trace-Trial-Heal`](https://github.com/RishabhShahIITGN/The-Wire-Trace-Trial-Heal)
* 🎥 **Demonstrations:** [Forward Run Video](https://youtu.be/hoY5CwrO2DI) • [Backward Run Video](https://youtu.be/m1hTtmguCew)
* **Hardware:** 6-DOF Addverb HEAL industrial cobot holding a conductive wire loop tool.
* **Pipeline:**
  1. Fine keyboard kinesthetic Cartesian path teaching along a complex 3D wire profile.
  2. Spline smoothing, trajectory cleaning, and time parameterization.
  3. High-precision closed-loop DLS-IK velocity control replay (`/velocity_controller/commands`).
  4. Real-time touch contact logging via electrical buzzer sensing (**0 touches recorded on final runs**).

---

## 4. Environment Setup & Quickstart

### Prerequisites
- Ubuntu 22.04 LTS
- Python 3.10+
- MuJoCo 3.x
- ROS 2 Humble Hawksbill (for `project/`)

### Setup Python Virtual Environment
```bash
git clone https://github.com/RishabhShahIITGN/ME-639.git
cd ME-639
python3 -m venv venv
source venv/bin/activate
pip install -r hw01-mujoco-rotations/requirements.txt
pip install mujoco mink numpy scipy matplotlib
```

### Running Simulations (Lab 01 & Lab 02)
```bash
cd lab
source ../venv/bin/activate

# Lab 01:
python3 spawn_waffle.py          # TurtleBot3 Waffle Pi
python3 spawn_quadrotor.py       # Skydio X2 Quadrotor

# Lab 02:
python3 spawn_heal.py            # Addverb HEAL FK & GUI
python3 spawn_franka.py          # Franka Panda FK & GUI
```

### Running Autonomous Pick-and-Place (Lab 03)
```bash
cd lab/lab-03
source ../../venv/bin/activate

# Interactive HEAL 6-DOF pick-and-place:
python3 scripts/ik_pick_place.py         # Mink solver
python3 scripts/ik_dls_pick_place.py     # Custom DLS solver
python3 scripts/ik_qp_pick_place.py      # Custom QP solver

# Interactive Franka Panda 7-DOF pick-and-place:
python3 scripts_franka/ik_pick_place.py     # Mink solver
python3 scripts_franka/ik_dls_pick_place.py # Custom DLS solver
python3 scripts_franka/ik_qp_pick_place.py  # Custom QP solver
```

### Building & Running the Peg Sorting Capstone Project
```bash
cd project/peg_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

# Run hardware bringup & cameras:
ros2 launch open_manipulator_x_bringup hardware.launch.py
ros2 launch peg_trial cameras.launch.py

# Launch autonomous peg sorting:
cd src/tri-wizard-peg-trial
./launch/run_autonomy.sh
```

---

## License

This repository is maintained for academic coursework under **ME-639: Introduction to Robotics** at IIT Gandhinagar. Code implementations and documentation are released under the [MIT License](LICENSE). Third-party robot models from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) and [ROBOTIS Menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie) are subject to their respective upstream licenses.

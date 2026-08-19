# Interactive 3D Rigid Body Rotation Laboratory

An interactive, browser-based visualization tool for rigid body kinematics. This application provides real-time rendering of 3D coordinate frame rotations, Euler angle sequences, and their corresponding rotation matrices.

Developed as a technical demonstration for **Introduction to Robotics**, this tool bridges the gap between abstract matrix algebra and physical 3D spatial orientation.

### 🔗 Resources
* **Live Application:** [(https://rishabhshahiitgn.github.io/rotation-visualizer/)]
* **Video Demonstration:** [Insert YouTube Link Here]

---

## Table of Contents
1. [Overview](#-overview)
2. [Core Features](#-core-features)
3. [Mathematical Formulation](#-mathematical-formulation)
4. [Technology Stack](#-technology-stack)
5. [Local Installation](#-local-installation)

---

## Overview

In robotics and aerospace engineering, determining the orientation of an end-effector or chassis in 3D space is critical for accurate kinematics, dynamics, and control. This application allows users to manipulate the Roll, Pitch, and Yaw of a rigid body and instantly observe the underlying mathematical transformations.

The application strictly models the progression:
**Euler Angles ➔ 3D Rotation ➔ Rotation Matrix ➔ Coordinate Transformation**

---

## Core Features

* **Real-Time WebGL Rendering:** Features a fixed global reference frame alongside a dynamic, rotating local body frame. Implements `OrbitControls` for unconstrained viewport manipulation.
* **Dynamic Matrix Computation:** Instantly calculates and displays the combined $3 \times 3$ rotation matrix as input parameters change.
* **Point Transformation Engine:** Tracks an arbitrary local coordinate point $P$ in the body frame and maps its transformed position in the global reference frame using matrix multiplication ($P_{world} = R \cdot P_{body}$).
* **Matrix Verification:** Continuously verifies and displays matrix properties, including determinant calculation ($\det(R) = 1$) and orthogonality checks ($R^T R = I$).
* **Sequential Analysis Mode:** Includes a step-by-step execution mode that isolates rotation about the X, Y, and Z axes to demonstrate the non-commutative nature of 3D rotations.
* **Quaternion & Axis-Angle Conversion:** Automatically converts the current rotation matrix into corresponding quaternions ($q_x, q_y, q_z, q_w$) and axis-angle representations.

---

## Mathematical Formulation

This application utilizes the standard aerospace **Z-Y-X Intrinsic Euler Angle** convention. 

Rotations are applied to the body axes in the following sequence:
1. **Roll ($\phi$):** Rotation about the body X-axis.
2. **Pitch ($\theta$):** Rotation about the intermediate body Y-axis.
3. **Yaw ($\psi$):** Rotation about the final body Z-axis.

The resulting composite rotation matrix $R$ is derived by multiplying the elemental rotation matrices:
$$R = R_z(\psi) \cdot R_y(\theta) \cdot R_x(\phi)$$

---

## Technology Stack

* **Frontend:** HTML5, CSS3 (Custom dashboard-style UI architecture)
* **Logic & Mathematics:** Vanilla JavaScript (ES6+)
* **3D Rendering Engine:** [Three.js (r128)](https://threejs.org/)

---

## Local Installation

This project is built using native web technologies and requires no package managers or local build servers.

1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/rishabhshahiitgn/rotation-visualizer.git](https://github.com/rishabhshahiitgn/rotation-visualizer.git)

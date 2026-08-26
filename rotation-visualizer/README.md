# Interactive 3D Rigid Body Rotation Laboratory

An interactive, browser-based visualization tool for rigid body kinematics. This application provides real-time rendering of 3D coordinate frame rotations, Euler angle sequences, and their corresponding rotation matrices.

Developed as a technical demonstration for **Introduction to Robotics**, this tool bridges the gap between abstract matrix algebra and physical 3D spatial orientation.

### 🔗 Resources

* **Live Application:** [rishabhshahiitgn.github.io/rotation-visualizer](https://rishabhshahiitgn.github.io/rotation-visualizer/)
* **Video Demonstration:** https://youtu.be/QE0hZ7LaZYM

---

## Table of Contents

1. [Overview](#overview)
2. [Core Features](#core-features)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Technology Stack](#technology-stack)
5. [Local Installation](#local-installation)

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

This application uses the standard aerospace **X-Y-Z Intrinsic Euler Angle** convention, also known as **Roll-Pitch-Yaw**.

Rotations are applied to the body axes in the following sequence:

1. **Roll ($\phi$):** Rotation about the body X-axis.
2. **Pitch ($\theta$):** Rotation about the intermediate (once-rotated) Y-axis.
3. **Yaw ($\psi$):** Rotation about the final (twice-rotated) Z-axis.

Because each rotation is intrinsic (applied about the body's own, already-rotated axes), the elemental matrices compose by **post-multiplication**, in the same order the rotations are applied:

$$R = R_x(\phi) \cdot R_y(\theta) \cdot R_z(\psi)$$

A point in the body frame is then mapped into the world frame as:

$$P_{world} = R \cdot P_{body}$$

> **Note:** This is equivalent to an *extrinsic* Z-Y-X sequence (rotating about the fixed world axes in Z, then Y, then X order) — the two descriptions produce the identical composite matrix $R$. The app consistently uses the intrinsic X-Y-Z / Roll-Pitch-Yaw framing above throughout its UI and derivation panel.

---

## Technology Stack

* **Frontend:** HTML5, CSS3 (Custom dashboard-style UI architecture)
* **Logic & Mathematics:** Vanilla JavaScript (ES6+)
* **3D Rendering Engine:** [Three.js (r128)](https://threejs.org/)

---

## Local Installation

This project is built using native web technologies and requires no package managers, bundlers, or local build servers.

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/rishabhshahiitgn/rotation-visualizer.git
   ```

2. Move into the project directory:
   ```bash
   cd rotation-visualizer
   ```

3. Open the app in your browser. Since everything is self-contained in a single HTML file loading Three.js from a CDN, you can simply double-click `index.html` (or open it via `File > Open` in your browser) — no local server is required.

   Optionally, if you'd prefer to serve it locally (e.g. for a consistent `http://` origin):
   ```bash
   # Python 3
   python3 -m http.server 8000

   # or, with Node.js installed
   npx serve .
   ```
   Then visit `http://localhost:8000` (or the port `serve` reports) in your browser.

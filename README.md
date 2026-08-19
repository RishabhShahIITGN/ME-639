# 3D Rigid Body Rotation & Euler Angle Visualizer 🚁

An interactive, web-based educational tool designed to visualize 3D coordinate frame rotations, Euler angles, and rotation matrices in real-time. This project was built as an assignment for the **Introduction to Robotics** course to bridge the gap between complex matrix mathematics and physical 3D spatial awareness.

### 🔗 Quick Links
* **Live Web Application:** [INSERT YOUR GITHUB PAGES .IO LINK HERE]
* **Video Demonstration:** [INSERT YOUR YOUTUBE VIDEO LINK HERE]

---

## 🎯 Project Overview

In robotics, understanding how an object is oriented in 3D space is critical for kinematics, dynamics, and control. This tool allows users to manipulate the Roll, Pitch, and Yaw of a 3D body (represented by an aircraft model) and instantly see the underlying mathematics update. 

The primary goal of this application is to demonstrate the sequence:
**Euler Angles ➔ 3D Rotation ➔ Rotation Matrix ➔ Coordinate Transformation**

## ✨ Core Features

* **Interactive 3D Stage:** Features a fixed World Frame (dim axes) and a rotating Body Frame (bright axes attached to a 3D aircraft model). Uses `OrbitControls` for free camera movement.
* **Live Mathematical Readouts:** As the user moves the sliders, the application instantly calculates and displays the combined $3 \times 3$ Rotation Matrix.
* **Point Transformation:** Tracks a local coordinate point $P$ in the body frame and maps its new position in the global world frame using matrix multiplication ($P_{world} = R \cdot P_{body}$).
* **Matrix Derivation & Properties:** Breaks down the final rotation matrix into its elemental components ($R_x$, $R_y$, $R_z$) and continuously verifies matrix orthogonality and determinant values.
* **Educational Modes:** * **Demo Mode:** An automated, guided animation that sequentially demonstrates Roll, Pitch, and Yaw.
  * **Step-by-Step Mode:** Allows the user to manually click through each axis rotation one at a time to understand order-of-operations.
* **Alternative Representations:** Live calculations for Axis-Angle representations and Quaternions ($q_x, q_y, q_z, q_w$).

## 🧮 Mathematical Convention

This tool preserves the classic aerospace **Z-Y-X Intrinsic Euler Angle** convention. 
* **Roll:** Rotation about the body's X-axis.
* **Pitch:** Rotation about the new body Y-axis.
* **Yaw:** Rotation about the final body Z-axis.

The resulting rotation matrix $R$ is calculated as:
$$R = R_z(\text{Yaw}) \cdot R_y(\text{Pitch}) \cdot R_x(\text{Roll})$$

## 💻 Technologies Used

* **HTML5 & CSS3:** For the structural layout and the "Avionics HUD" engineering aesthetic.
* **Vanilla JavaScript:** Handles all matrix mathematics, slider events, and UI synchronization without heavy frontend frameworks.
* **Three.js (r128):** The core WebGL 3D rendering engine used to draw the coordinate frames, grid, and aircraft mesh.

## 🚀 How to Run Locally

Because this project is built with native web technologies, no complex build steps, servers, or package managers are required.

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)

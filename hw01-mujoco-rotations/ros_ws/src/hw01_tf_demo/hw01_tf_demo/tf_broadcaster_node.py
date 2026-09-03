"""
tf_broadcaster_node.py -- HW1 Part 2, Task 3 (Problem 9).

Broadcasts a fixed `space_frame` and a rotating `body_frame` so you can
watch current-frame vs. fixed-frame rotation composition live in rviz2.

A ROS parameter `composition_mode` ("fixed" or "current") selects
pre-multiplication vs. post-multiplication and can be changed live:

    ros2 param set /hw01_tf_broadcaster composition_mode current
"""

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as SciRotation
from tf2_ros import TransformBroadcaster


def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


ELEMENTARY_ROTATIONS = {"x": Rx, "y": Ry, "z": Rz}

# Task 1 Test Case B: 90 deg about X, then 90 deg about Y.
STEP_SEQUENCE = [
    ("x", np.deg2rad(90)),
    ("y", np.deg2rad(90)),
]

PAUSE_TICKS = 2  # extra timer ticks at the end of a sequence before reset


def R_to_quat_xyzw(R):
    """3x3 rotation matrix -> ROS xyzw quaternion via scipy."""
    return SciRotation.from_matrix(R).as_quat()


class Hw01TfBroadcaster(Node):
    def __init__(self):
        super().__init__("hw01_tf_broadcaster")
        self.declare_parameter("composition_mode", "fixed")  # "fixed" or "current"
        self.declare_parameter("step_period", 1.0)  # seconds per elemental rotation

        self.tf_broadcaster = TransformBroadcaster(self)
        self.R_body = np.eye(3)
        self.step_index = 0
        self.pause_remaining = 0

        step_period = self.get_parameter("step_period").value
        self.timer = self.create_timer(step_period, self.on_timer)
        self.get_logger().info(
            "hw01_tf_broadcaster started. Toggle live with: "
            "ros2 param set /hw01_tf_broadcaster composition_mode current"
        )

    def on_timer(self):
        self.broadcast_frame("world", "space_frame", np.eye(3), z=0.0)
        self.broadcast_frame("space_frame", "body_frame", self.R_body, z=1.0)

        if self.pause_remaining > 0:
            self.pause_remaining -= 1
            if self.pause_remaining == 0:
                self.R_body = np.eye(3)
                self.step_index = 0
                self.get_logger().info("Reset to identity; restarting sequence")
            return

        if self.step_index >= len(STEP_SEQUENCE):
            self.get_logger().info("Sequence finished; pausing before reset")
            self.pause_remaining = PAUSE_TICKS
            return

        axis, angle = STEP_SEQUENCE[self.step_index]
        R_step = ELEMENTARY_ROTATIONS[axis](angle)
        mode = self.get_parameter("composition_mode").get_parameter_value().string_value

        if mode == "fixed":
            # Space-frame (fixed) composition: pre-multiply.
            self.R_body = R_step @ self.R_body
        elif mode == "current":
            # Body-frame (current) composition: post-multiply.
            self.R_body = self.R_body @ R_step
        else:
            self.get_logger().warn(
                f'Unknown composition_mode="{mode}"; expected "fixed" or "current"'
            )
            return

        self.get_logger().info(
            f"Step {self.step_index + 1}/{len(STEP_SEQUENCE)}: "
            f"{np.degrees(angle):.0f} deg about {axis.upper()} ({mode})"
        )
        self.step_index += 1

    def broadcast_frame(self, parent, child, R, z=0.0):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = float(z)
        qx, qy, qz, qw = R_to_quat_xyzw(R)
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = Hw01TfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

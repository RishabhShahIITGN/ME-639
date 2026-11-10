#!/usr/bin/env python3
"""Lightweight OpenCV-based image viewer for RealSense camera feed.
Bypasses Qt/GLib schema bugs in rqt.

Usage:
    python3 scripts/view_camera.py
    python3 scripts/view_camera.py --topic /cam1/camera/color/image_raw
"""
import argparse
import os
import sys

# Force X11 backend to bypass Wayland / GNOME schema antialiasing bug
os.environ["GDK_BACKEND"] = "x11"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageViewer(Node):
    def __init__(self, topic):
        super().__init__("simple_image_viewer")
        self.br = CvBridge()
        self.sub = self.create_subscription(Image, topic, self.on_image, 1)
        self.get_logger().info(f"Listening to {topic} ... Press 'q' or ESC in window to exit.")

    def on_image(self, msg):
        try:
            frame = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imshow("RealSense Live Feed", frame)
            key = cv2.waitKey(1)
            if key in (27, ord('q')):
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/cam1/camera/color/image_raw")
    args = parser.parse_args()

    rclpy.init()
    node = ImageViewer(args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

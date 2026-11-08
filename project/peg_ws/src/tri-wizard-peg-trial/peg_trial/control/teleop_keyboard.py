"""Keyboard teleoperation (the 30-point baseline + fallback during autonomy).

Run in its OWN terminal (it needs the keyboard):
    ros2 run peg_trial teleop_keyboard

The gripper always points straight down; you move its centre in x / y / z
(robot base frame) and IK computes the joints.
"""
from __future__ import annotations

import json
import math
import select
import sys
import termios
import threading
import time
import tty

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String

from peg_trial.control.arm_interface import ESTOP_TOPIC, ArmInterface
from peg_trial.planning.kinematics import IKError, Pose, fk, ik, interpolate_line
from peg_trial.utils.config import load

HELP = """
---------------- PEG TRIAL TELEOP ----------------
  w / s : forward / back   (+x / -x)
  a / d : left / right     (+y / -y)
  r / f : up / down        (+z / -z)
  [ / ] : smaller / bigger step   (now {step} mm)
  o / c : gripper OPEN / CLOSE
  g     : GRASP  = go down to grasp height of the peg below, settle, close
  l     : LIFT   = straight up 6 cm
  p     : PLACE  = lower slowly 3 cm, open, go up
  1-4   : hover over detected peg   (1 tri, 2 square, 3 hex, 4 circle)
  6-9   : hover over detected socket (6 tri, 7 square, 8 hex, 9 circle)
  h     : HOME pose
  SPACE : *** EMERGENCY STOP ***  (x = release e-stop)
  ?     : this help         ESC / Ctrl-C : quit
---------------------------------------------------
"""
ORDER = ["triangle", "square", "hexagon", "circle"]


class Teleop(Node):
    def __init__(self):
        super().__init__("peg_teleop")
        self.cfg = load("task.yaml")
        self.arm = ArmInterface(self, self.cfg["robot"])
        self.scene = {"pegs": [], "sockets": []}
        self.create_subscription(String, "/peg_trial/scene", self._on_scene, 10)
        self.estop_pub = self.create_publisher(Bool, ESTOP_TOPIC, 10)
        self.step = 0.01
        self.target = None

    def _on_scene(self, m):
        self.scene = json.loads(m.data)

    def current_pose(self):
        if self.target is None:
            q = self.arm.joints()
            if q is None:
                return Pose(0.16, 0.0, 0.08, math.pi / 2)
            x, y, z, pitch = fk(q)
            # If the arm is in the folded rest pose (high z, horizontal pitch):
            # initialize target to the reachable ready pose so subsequent moves succeed
            if z > 0.14 or abs(pitch - math.pi / 2) > 0.4:
                self.target = Pose(0.16, 0.0, 0.08, math.pi / 2)
            else:
                self.target = Pose(x, y, z, math.pi / 2)
        return self.target

    def go(self, pose: Pose, speed=0.04, line=True):
        start = self.current_pose()
        q_start = self.arm.joints()
        if q_start is None:
            print("\r  ! no joint states from arm")
            return False

        # 1. Protect against base singularity and workspace limits
        if pose.x < 0.08:
            print(f"\r  ! too close to base (x={pose.x:.3f} < 0.08m)")
            return False
        if pose.x > 0.24:
            print(f"\r  ! beyond maximum reach (x={pose.x:.3f} > 0.24m)")
            return False
        r = math.hypot(pose.x - 0.012, pose.y)
        if r < 0.10:
            print(f"\r  ! radial distance too close to base (r={r:.3f} < 0.10m)")
            return False
        if r > 0.24:
            print(f"\r  ! radial distance out of reach (r={r:.3f} > 0.24m)")
            return False
        if pose.z < -0.07:
            print(f"\r  ! table collision limit (z={pose.z:.3f} < -0.07m)")
            return False
        if pose.z > 0.22:
            print(f"\r  ! maximum height limit (z={pose.z:.3f} > 0.22m)")
            return False

        # 2. Compute IK for path
        try:
            if line:
                poses = interpolate_line(start, pose, 0.01)
                ik_points = [ik(p) for p in poses]
            else:
                poses = [pose]
                ik_points = [ik(pose)]
        except IKError as e:
            print(f"\r  ! unreachable: {e}")
            return False

        # 3. Detect sudden angle jumps (e.g. crossing branch cuts or singularities)
        MAX_SAFE_STEP = 0.40  # rad (~23 deg) per Cartesian step
        prev_q = q_start
        for q in ik_points:
            for j in range(4):
                if abs(q[j] - prev_q[j]) > MAX_SAFE_STEP:
                    print(f"\r  ! motion rejected: sudden joint{j+1} jump ({abs(q[j] - prev_q[j]):.2f} rad)")
                    return False
            prev_q = q

        # 4. Enforce maximum joint velocity so motors never spin wildly
        MAX_JOINT_VEL = 0.8  # rad/s max safe joint speed
        pts = []
        t_acc = 0.0
        prev_q = q_start
        prev_p = start

        for p, q in zip(poses, ik_points):
            cart_dist = math.dist((prev_p.x, prev_p.y, prev_p.z), (p.x, p.y, p.z))
            t_cart = cart_dist / speed
            t_joint = max(abs(q[j] - prev_q[j]) for j in range(4)) / MAX_JOINT_VEL
            dt = max(0.12, t_cart, t_joint)
            t_acc += dt
            pts.append((q, t_acc))
            prev_q = q
            prev_p = p

        self.target = pose
        print(f"\r  -> x={pose.x:.3f} y={pose.y:.3f} z={pose.z:.3f}")
        return self.arm.move(pts, settle=0.0)

    def handle_key(self, k):
        p = self.current_pose()
        s = self.step
        g = self.cfg["grasp"]
        moves = {"w": (s, 0, 0), "s": (-s, 0, 0), "a": (0, s, 0), "d": (0, -s, 0), "r": (0, 0, s), "f": (0, 0, -s)}
        if k == " ":
            self.estop_pub.publish(Bool(data=True))
            print("\r  *** E-STOP *** (press x to release)")
        elif k == "x":
            self.estop_pub.publish(Bool(data=False))
            self.target = None
        elif self.arm.estopped:
            print("\r  e-stop active — press x first")
        elif k in moves:
            dx, dy, dz = moves[k]
            self.go(Pose(p.x + dx, p.y + dy, p.z + dz), line=False if s >= 0.02 else True)
        elif k in "[]":
            self.step = max(0.002, self.step / 2) if k == "[" else min(0.05, self.step * 2)
            print(f"\r  step = {self.step * 1000:.0f} mm")
        elif k == "o":
            self.arm.gripper("open")
        elif k == "c":
            self.arm.gripper("close")
        elif k == "g":
            peg = min(self.scene["pegs"], key=lambda q: math.hypot(q["x"] - p.x, q["y"] - p.y), default=None)
            z = (peg["z_top"] if peg else p.z - g["pregrasp_offset"]) - g["grasp_depth"]
            if self.go(Pose(p.x, p.y, z), speed=0.03):
                time.sleep(g["settle_delay"])
                self.arm.gripper("close", wait=g["close_delay"])
        elif k == "l":
            self.go(Pose(p.x, p.y, p.z + 0.06), speed=0.03)
        elif k == "p":
            if self.go(Pose(p.x, p.y, p.z - 0.03), speed=0.015):
                self.arm.gripper("open")
                self.go(Pose(p.x, p.y, p.z + 0.03), speed=0.03)
        elif k in "1234" or k in "6789":
            shape = ORDER["1234".index(k)] if k in "1234" else ORDER["6789".index(k)]
            items = self.scene["pegs"] if k in "1234" else self.scene["sockets"]
            it = next((i for i in items if i["shape"] == shape), None)
            if it is None:
                print(f"\r  no {shape} detected")
                return
            z = it["z_top"] + (g["pregrasp_offset"] if k in "1234" else 0.10)
            self.go(Pose(it["x"], it["y"], z), line=False)
        elif k == "h":
            self.target = None
            q_now = self.arm.joints() or [0.0, 0.0, 0.0, 1.5708]
            ready_joints = [0.0, 0.0, 0.0, 1.5708]
            max_d = max(abs(a - b) for a, b in zip(q_now, ready_joints))
            t_home = max(3.0, max_d / 0.6)
            print(f"\r  moving to ready pose (gripper down) in {t_home:.1f}s ...")
            self.arm.move([(ready_joints, t_home)], settle=0.2)
            self.target = Pose(0.16, 0.0, 0.08, math.pi / 2)
        elif k == "?":
            print(HELP.format(step=int(self.step * 1000)))


def main():
    rclpy.init()
    node = Teleop()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not node.arm.wait_for_joints(10.0):
        print("no /joint_states — start the robot bringup first")
        return
    print(HELP.format(step=int(node.step * 1000)))
    old = termios.tcgetattr(sys.stdin)
    worker = None
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                k = sys.stdin.read(1)
                if k in ("\x1b", "\x03"):
                    break
                if k == " ":                       # E-STOP is handled HERE, never blocked
                    node.handle_key(k)
                    continue
                if worker and worker.is_alive():   # a motion is still running
                    print("\r  (busy — wait, or SPACE to stop)")
                    continue
                worker = threading.Thread(target=node.handle_key, args=(k,), daemon=True)
                worker.start()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.arm.hold()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

"""ROS 2 node: FULL AUTONOMY.

  1. Wait for the vision scene, cache the socket poses (board is empty at start).
  2. For each shape (order from task.yaml): find the peg, plan, execute.
  3. Verify: is the peg now sitting in the socket? If not -> retry (max_retries).
  4. Log every attempt to logs/trials.csv and the run to logs/runs.csv
     (scripts/evaluate_timing.py turns those into success-rate / timing tables).

E-STOP: publish `true` on /peg_trial/estop (teleop SPACE key, or
   ros2 topic pub --once /peg_trial/estop std_msgs/Bool "{data: true}")
"""
from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
import uuid

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from peg_trial.control.arm_interface import ArmInterface
from peg_trial.planning.pick_place_planner import PlanError, match_pegs_to_sockets, plan_pick_place
from peg_trial.utils.config import load
from peg_trial.utils.scene_buffer import SceneBuffer


class AutonomyNode(Node):
    def __init__(self):
        super().__init__("peg_autonomy")
        self.declare_parameter("task_file", "")
        self.declare_parameter("backend", "")          # override robot.backend
        self.declare_parameter("log_dir", os.environ.get("PEG_TRIAL_LOGS", "logs"))
        self.cfg = load("task.yaml", self.get_parameter("task_file").value or None)
        if self.get_parameter("backend").value:
            self.cfg["robot"]["backend"] = self.get_parameter("backend").value
        self.log_dir = self.get_parameter("log_dir").value
        os.makedirs(self.log_dir, exist_ok=True)

        self.arm = ArmInterface(self, self.cfg["robot"])
        self.scenes = SceneBuffer(20)
        self.create_subscription(String, "/peg_trial/scene", lambda m: self.scenes.add(json.loads(m.data)), 10)
        self.status_pub = self.create_publisher(String, "/peg_trial/status", 10)

    # ---------- helpers ----------
    def say(self, text):
        self.get_logger().info(text)
        self.status_pub.publish(String(data=text))

    def run_steps(self, steps) -> bool:
        for s in steps:
            if s["type"] == "move":
                ok = self.arm.move(s["points"])
            elif s["type"] == "gripper":
                ok = self.arm.gripper(s["cmd"], wait=0.3)
            else:
                ok = self.arm._sleep(s["sec"])
            if not ok:
                self.get_logger().error(f"step '{s['name']}' aborted")
                return False
        return True

    def fresh_snapshot(self, expected_sockets=0):
        self.scenes.clear()
        return self.scenes.snapshot(min_frames=6, timeout=6.0, expected_sockets=expected_sockets)

    def verify(self, color, sock, peg_h) -> tuple[bool, str]:
        ex = self.cfg["execution"]
        snap = self.fresh_snapshot()
        if snap is None:
            return False, "no vision"
        peg = next((p for p in snap["pegs"] if p["color"] == color), None)
        if peg is None:
            return False, "peg not visible"
        dxy = math.hypot(peg["x"] - sock["x"], peg["y"] - sock["y"])
        z_expect = sock["z_top"] + peg_h - self.cfg["place"]["insert_depth"]
        dz = peg["z_top"] - z_expect
        ok = dxy < ex["verify_xy_tol"] and abs(dz) < ex["verify_z_tol"]
        return ok, f"dxy={dxy * 1000:.0f}mm dz={dz * 1000:.0f}mm"

    def log_row(self, fname, row: dict):
        path = os.path.join(self.log_dir, fname)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    # ---------- main task ----------
    def run(self):
        if not self.arm.wait_for_joints(10.0):
            self.get_logger().error("no /joint_states — is the robot bringup running?")
            return
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        ex = self.cfg["execution"]
        self.say(f"RUN {run_id}: going home")
        self.arm.gripper("open")
        self.arm.move([(self.cfg["motion"]["home_joints"], 3.0)])

        snap = self.fresh_snapshot(expected_sockets=4)
        if snap is None or not snap["sockets"]:
            self.get_logger().error("no sockets detected — check board HSV / camera calibration")
            return
        sockets = snap["sockets"]
        self.say(f"sockets: {[s['shape'] for s in sockets]}  pegs: {[(p['color'], p['shape']) for p in snap['pegs']]}")

        t_run = time.time()
        done = 0
        order = ex["order"]
        pairs = match_pegs_to_sockets(snap["pegs"], sockets)
        pairs.sort(key=lambda ps: order.index(ps[0]["shape"]) if ps[0]["shape"] in order else 99)
        if len(pairs) < len(snap["pegs"]):
            self.get_logger().warn("some pegs have no matching socket shape!")

        for idx, (peg, sock) in enumerate(pairs):
            color = peg["color"]
            t0, attempts, ok, note, yaw_err = time.time(), 0, False, "", 0.0
            while attempts <= ex["max_retries"] and not ok and not self.arm.estopped:
                attempts += 1
                if attempts > 1:   # peg may have moved / fallen -> look again
                    s2 = self.fresh_snapshot()
                    peg = next((p for p in (s2 or {"pegs": []})["pegs"] if p["color"] == color), None)
                    if peg is None:
                        note = "peg lost"
                        break
                self.say(f"[{idx + 1}/{len(pairs)}] {color} {peg['shape']} -> socket at "
                         f"({sock['x']:.3f},{sock['y']:.3f})  attempt {attempts}")
                try:
                    other_pegs = [p for p in snap["pegs"] if p["color"] != color]
                    steps, info = plan_pick_place(peg, sock, self.cfg, self.arm.joints(), obstacles=other_pegs)
                except PlanError as e:
                    note = str(e)
                    self.get_logger().error(note)
                    break
                yaw_err = info["yaw_error_deg"]
                if abs(yaw_err) > 8:
                    self.get_logger().warn(
                        f"predicted yaw misalignment {yaw_err:.0f} deg (OMX has no wrist roll) — "
                        f"rotate this peg by {-yaw_err:+.0f} deg (+ = counter-clockwise seen from above) "
                        f"during setup, or use a chamfered socket")
                if not self.run_steps(steps):
                    note = "aborted (e-stop)"
                    break
                if ex.get("verify", True) and self.cfg["robot"]["backend"] != "dry_run":
                    ok, note = self.verify(color, sock, info["peg_height"])
                    self.say(f"verify {color}: {'OK' if ok else 'FAIL'} {note}")
                else:
                    ok = True
            done += ok
            self.log_row("trials.csv", dict(
                run_id=run_id, index=idx + 1, color=color, shape=peg["shape"] if peg else "", attempts=attempts,
                success=int(ok), duration_s=round(time.time() - t0, 2), yaw_error_deg=round(yaw_err, 1), note=note))
            if self.arm.estopped:
                break

        total = time.time() - t_run
        self.log_row("runs.csv", dict(run_id=run_id, backend=self.cfg["robot"]["backend"], pegs=len(pairs),
                                      success=done, total_time_s=round(total, 2), estop=int(self.arm.estopped)))
        self.say(f"RUN {run_id} finished: {done}/{len(pairs)} pegs in {total:.1f}s")


def main():
    rclpy.init()
    node = AutonomyNode()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)   # ROS callbacks in background
    spin.start()
    try:
        node.run()                                          # task logic in main thread
    except KeyboardInterrupt:
        node.arm.hold()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

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
from peg_trial.planning.pick_place_planner import PlanError, match_pegs_to_sockets, plan_order, plan_pick_place
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
                ok = self.arm.gripper(s["cmd"], wait=0.3, gap=s.get("gap"))
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

        if ex.get("mode", "one_by_one") == "one_by_one":
            return self.run_one_by_one(run_id, sockets)

        t_run = time.time()
        done = 0
        order = ex["order"]
        pairs = match_pegs_to_sockets(snap["pegs"], sockets)
        pairs.sort(key=lambda ps: order.index(ps[0]["shape"]) if ps[0]["shape"] in order else 99)
        if len(pairs) < len(snap["pegs"]):
            self.get_logger().warn("some pegs have no matching socket shape!")

        # find an order in which every peg can be placed without collisions (look-ahead search)
        best, errs = plan_order(pairs, self.cfg, sockets, self.arm.joints(), preferred=order)
        if len(best) < len(pairs):
            self.get_logger().warn(f"only {len(best)}/{len(pairs)} pegs have a collision-free plan:")
            for col, e in errs.items():
                self.get_logger().warn(f"  {col}: {e}")
        self.say(f"placement order: {[p['color'] for p, _ in best]}")
        remaining = list(best) + [ps for ps in pairs if ps not in best]   # (peg, socket) still to do
        placed = []                      # (peg, socket) already inserted -> obstacles for later pegs
        latest = snap
        idx = 0
        while remaining and not self.arm.estopped:
            idx += 1
            # ---- pick the first peg (in preferred order) that has a COLLISION-FREE plan
            chosen, plan_errors = None, []
            for peg, sock in remaining:
                cur = next((p for p in latest["pegs"] if p["color"] == peg["color"]), peg)
                done_colors = {p["color"] for p, _ in placed}
                others = [p for p in latest["pegs"] if p["color"] != cur["color"] and p["color"] not in done_colors]
                try:
                    steps, info = plan_pick_place(cur, sock, self.cfg, self.arm.joints(), obstacles=others,
                                                  placed=placed, sockets=sockets)
                    chosen = (peg, cur, sock, steps, info)
                    break
                except PlanError as e:
                    plan_errors.append(f"{peg['color']}: {e}")
            if chosen is None:
                for e in plan_errors:
                    self.get_logger().error(f"no safe plan — {e}")
                for peg, _ in remaining:
                    self.log_row("trials.csv", dict(run_id=run_id, index=idx, color=peg["color"], shape=peg["shape"],
                                                    attempts=0, success=0, duration_s=0, yaw_error_deg=0,
                                                    note="no collision-free plan"))
                break
            peg0, peg, sock, steps, info = chosen
            remaining.remove((peg0, sock))
            color = peg["color"]
            t0, attempts, ok, note = time.time(), 0, False, ""
            yaw_err = info["yaw_error_deg"]
            while attempts <= ex["max_retries"] and not ok and not self.arm.estopped:
                attempts += 1
                if attempts > 1:   # peg may have moved / fallen -> look again and re-plan
                    s2 = self.fresh_snapshot()
                    peg = next((p for p in (s2 or {"pegs": []})["pegs"] if p["color"] == color), None)
                    if peg is None:
                        note = "peg lost"
                        break
                    done_colors = {p["color"] for p, _ in placed}
                    others = [p for p in s2["pegs"] if p["color"] != color and p["color"] not in done_colors]
                    try:
                        steps, info = plan_pick_place(peg, sock, self.cfg, self.arm.joints(), obstacles=others,
                                                      placed=placed, sockets=sockets)
                    except PlanError as e:
                        note = str(e)
                        self.get_logger().error(note)
                        break
                self.say(f"[{idx}/{len(pairs)}] {color} {peg['shape']} -> socket ({sock['x']:.3f},{sock['y']:.3f}) "
                         f"attempt {attempts} | grip {info['grasp_depth'] * 1000:.0f} mm below top, "
                         f"carry z={info['carry_z']:.3f}, route={info['route']}")
                for w in info.get("warnings", []):
                    self.get_logger().warn(w)
                if abs(yaw_err) > self.cfg["execution"].get("yaw_warn_deg", 8):
                    self.get_logger().warn(
                        f"predicted socket misalignment {yaw_err:+.0f} deg after the squeeze grasp "
                        f"(OMX has no wrist roll, so this cannot be corrected by the arm; it depends only on "
                        f"how the socket is turned relative to the robot) — relying on socket clearance")
                if not self.run_steps(steps):
                    note = "aborted (e-stop)"
                    break
                if ex.get("verify", True) and self.cfg["robot"]["backend"] != "dry_run":
                    ok, note = self.verify(color, sock, info["peg_height"])
                    self.say(f"verify {color}: {'OK' if ok else 'FAIL'} {note}")
                else:
                    ok = True
            done += ok
            if ok:
                placed.append((peg, sock))
            self.log_row("trials.csv", dict(
                run_id=run_id, index=idx, color=color, shape=peg["shape"] if peg else "", attempts=attempts,
                success=int(ok), duration_s=round(time.time() - t0, 2), yaw_error_deg=round(yaw_err, 1), note=note))
            if remaining and self.cfg["robot"]["backend"] != "dry_run":
                latest = self.fresh_snapshot() or latest      # pegs may have been nudged: refresh positions

        total = time.time() - t_run
        self.log_row("runs.csv", dict(run_id=run_id, backend=self.cfg["robot"]["backend"], pegs=len(pairs),
                                      success=done, total_time_s=round(total, 2), estop=int(self.arm.estopped)))
        self.say(f"RUN {run_id} finished: {done}/{len(pairs)} pegs in {total:.1f}s")

    # ------------------------------------------------------------------ one peg at a time
    def wait_for_new_peg(self, placed_colors, sockets, bad_spots=(), timeout=600.0):
        """Wait (arm at home) until ONE new peg stands still in the workspace.

        'New' = colour not placed yet and not sitting in a socket. 'Still' = the same
        position (±5 mm) in two snapshots taken 1.5 s apart, so a hand putting the peg
        down is never chased.
        """
        t0, last = time.time(), None
        while time.time() - t0 < timeout and not self.arm.estopped:
            snap = self.fresh_snapshot()
            cands = []
            for p in (snap or {"pegs": []})["pegs"]:
                in_socket = any(math.hypot(p["x"] - s["x"], p["y"] - s["y"]) < 0.025 for s in sockets)
                tried = any(math.hypot(p["x"] - bx, p["y"] - by) < 0.01 for bx, by in bad_spots)
                if p["color"] not in placed_colors and not in_socket and not tried:
                    cands.append(p)
            if cands:
                p = cands[0]
                if last and last["color"] == p["color"] and math.hypot(p["x"] - last["x"], p["y"] - last["y"]) < 0.005:
                    return p, snap
                last = p
            else:
                last = None
            time.sleep(1.5)
        return None, None

    def run_one_by_one(self, run_id, sockets):
        """Task flow: a person puts ONE peg anywhere -> robot places it -> next peg ... until all sockets are full.
        Pegs already in their sockets stay there and are obstacles for every later move."""
        ex = self.cfg["execution"]
        placed = []                         # (peg, socket) already inserted
        t_run = time.time()
        n_total = len(sockets)
        bad_spots = []                      # positions where a peg could not be done safely
        while len(placed) < n_total and not self.arm.estopped:
            self.say(f"[{len(placed)}/{n_total} placed] waiting for the next peg in the workspace...")
            peg, snap = self.wait_for_new_peg({p["color"] for p, _ in placed}, sockets, bad_spots)
            if peg is None:
                self.get_logger().error("no new peg appeared — stopping")
                break
            used = {id(s) for _, s in placed}
            sock = next((s for s in sockets if s["shape"] == peg["shape"] and id(s) not in used), None)
            if sock is None:
                self.get_logger().error(f"{peg['color']} {peg['shape']}: no free socket of that shape")
                break
            t0, attempts, ok, note, yaw_err = time.time(), 0, False, "", 0.0
            while attempts <= ex["max_retries"] and not ok and not self.arm.estopped:
                attempts += 1
                if attempts > 1:
                    s2 = self.fresh_snapshot()
                    peg = next((p for p in (s2 or {"pegs": []})["pegs"] if p["color"] == peg["color"]), None)
                    if peg is None:
                        note = "peg lost"
                        break
                try:
                    # obstacles = only the pegs already in sockets (plus board/table); the new peg is the target
                    steps, info = plan_pick_place(peg, sock, self.cfg, self.arm.joints(), obstacles=[],
                                                  placed=placed, sockets=sockets)
                except PlanError as e:
                    note = str(e)
                    self.get_logger().error(f"{peg['color']}: no safe plan — {e}  (put the peg somewhere else)")
                    break
                yaw_err = info["yaw_error_deg"]
                self.say(f"{peg['color']} {peg['shape']} at ({peg['x']:.3f},{peg['y']:.3f}) -> socket "
                         f"({sock['x']:.3f},{sock['y']:.3f}) | grip {info['grasp_depth'] * 1000:.0f} mm below top, "
                         f"route={info['route']}, predicted yaw error {yaw_err:+.0f} deg")
                for w in info.get("warnings", []):
                    self.get_logger().warn(w)
                if not self.run_steps(steps):
                    note = "aborted (e-stop)"
                    break
                if ex.get("verify", True) and self.cfg["robot"]["backend"] != "dry_run":
                    ok, note = self.verify(peg["color"], sock, info["peg_height"])
                    self.say(f"verify {peg['color']}: {'OK' if ok else 'FAIL'} {note}")
                else:
                    ok = True
            if ok:
                placed.append((peg, sock))
            self.log_row("trials.csv", dict(
                run_id=run_id, index=len(placed) + (0 if ok else 1), color=peg["color"] if peg else "",
                shape=peg["shape"] if peg else "", attempts=attempts, success=int(ok),
                duration_s=round(time.time() - t0, 2), yaw_error_deg=round(yaw_err, 1), note=note))
            if self.arm.estopped:
                break
            if not ok and peg is not None:
                bad_spots.append((peg["x"], peg["y"]))      # wait until the person moves it
        total = time.time() - t_run
        self.log_row("runs.csv", dict(run_id=run_id, backend=self.cfg["robot"]["backend"], pegs=n_total,
                                      success=len(placed), total_time_s=round(total, 2),
                                      estop=int(self.arm.estopped)))
        self.say(f"RUN {run_id} finished: {len(placed)}/{n_total} pegs placed in {total:.1f}s")


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

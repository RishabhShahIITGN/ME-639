"""Pick-and-place planner: one peg + its socket -> list of robot steps.

Sequence (matches the task sheet's "Control and planning" rules):

  home
   -> PRE-GRASP  : hover `pregrasp_offset` above the peg top, gripper down, OPEN
   -> APPROACH   : straight down along the board normal to the grasp height
   -> SETTLE     : wait `settle_delay`, then CLOSE, wait `close_delay`
   -> LIFT       : straight up until the peg bottom clears the board
   -> TRANSIT    : move above the socket
   -> HOVER      : `hover_offset` above the socket
   -> INSERT     : lower SLOWLY in small steps to the insert depth
   -> RELEASE    : OPEN, wait, straight up, go home

Each step is a dict:
   {"type": "move", "name": ..., "points": [(q[4], t_from_start), ...]}
   {"type": "gripper", "name": ..., "cmd": "open" | "close"}
   {"type": "wait", "name": ..., "sec": float}
"""
from __future__ import annotations

import math

from .kinematics import IKError, Pose, fk, ik, ik_best, interpolate_line

DOWN = math.pi / 2
SHAPE_SYM = {"square": 4, "triangle": 3, "hexagon": 6, "circle": 0}


class PlanError(Exception):
    pass


def match_pegs_to_sockets(pegs, sockets):
    """Same shape = match. If two sockets have the same shape, take the closest."""
    pairs, used = [], set()
    for p in pegs:
        cands = [(i, s) for i, s in enumerate(sockets) if s["shape"] == p["shape"] and i not in used]
        if not cands:
            continue
        i, s = min(cands, key=lambda c: math.hypot(c[1]["x"] - p["x"], c[1]["y"] - p["y"]))
        used.add(i)
        pairs.append((p, s))
    return pairs


def yaw_error(peg, sock, sym):
    """Peg yaw after transport minus socket yaw (wrapped to the shape symmetry).

    The OMX has no wrist roll: the peg turns WITH joint1. So if joint1 turns
    by d = atan2(socket) - atan2(peg), the peg's yaw also changes by d.
    """
    if sym == 0:
        return 0.0
    d = math.atan2(sock["y"], sock["x"]) - math.atan2(peg["y"], peg["x"])
    period = 2 * math.pi / sym
    e = (peg["yaw"] + d - sock["yaw"]) % period
    return e - period if e > period / 2 else e


def _joint_time(q0, q1, max_vel=0.8, min_t=0.6):
    return max(min_t, max(abs(a - b) for a, b in zip(q0, q1)) / max_vel)


def _line(p0, p1, step, speed, min_t, t0, strict=True, pitch_tol=0.0):
    """IK every few mm along a straight line -> joint points with timestamps."""
    pts, t = [], t0
    poses = interpolate_line(p0, p1, step)
    seg_t = max(min_t, math.dist((p0.x, p0.y, p0.z), (p1.x, p1.y, p1.z)) / speed) / len(poses)
    for p in poses:
        q = ik(p) if strict else ik_best(p, pitch_tol)
        t += seg_t
        pts.append((q, t))
    return pts


def plan_pick_place(peg, sock, cfg, start_q=None):
    g, pl, mo = cfg["grasp"], cfg["place"], cfg["motion"]
    step, speed, min_t = mo["cartesian_step"], mo["move_speed"], mo["min_segment_time"]
    home = mo["home_joints"]
    start_q = start_q or home

    peg_h = max(0.03, peg["z_top"] - cfg["table_z"])          # measured peg height
    below_tcp = peg_h - g["grasp_depth"]                       # gripper centre -> peg bottom

    grasp = Pose(peg["x"], peg["y"], peg["z_top"] - g["grasp_depth"], DOWN)
    pre = Pose(grasp.x, grasp.y, peg["z_top"] + g["pregrasp_offset"], DOWN)
    lift_z = max(pre.z, sock["z_top"] + g["lift_clearance"] + below_tcp)
    lift = Pose(grasp.x, grasp.y, lift_z, DOWN)
    hover = Pose(sock["x"], sock["y"], sock["z_top"] + pl["hover_offset"] + below_tcp, DOWN)
    above_sock = Pose(sock["x"], sock["y"], max(lift_z, hover.z), DOWN)
    final = Pose(sock["x"], sock["y"], sock["z_top"] - pl["insert_depth"] + below_tcp, DOWN)

    # quick reachability check of every key pose (gives a clear error message)
    for name, p in (("pre-grasp", pre), ("grasp", grasp), ("hover", hover), ("insert", final)):
        try:
            ik(p)
        except IKError as e:
            raise PlanError(f"{name} pose ({p.x:.3f},{p.y:.3f},{p.z:.3f}) unreachable: {e}") from e

    steps = []
    q_pre = ik(pre)
    steps.append({"type": "gripper", "name": "open", "cmd": "open"})
    steps.append({"type": "move", "name": "to_pregrasp", "points": [(q_pre, _joint_time(start_q, q_pre))]})
    steps.append({"type": "move", "name": "approach", "points": _line(pre, grasp, step, speed / 2, min_t, 0.0)})
    steps.append({"type": "wait", "name": "settle", "sec": g["settle_delay"]})
    steps.append({"type": "gripper", "name": "close", "cmd": "close"})
    steps.append({"type": "wait", "name": "grip", "sec": g["close_delay"]})
    steps.append({"type": "move", "name": "lift", "points": _line(grasp, lift, step, speed / 2, min_t, 0.0, False, mo["pitch_tolerance"])})

    # transit: go up/over in joint space (fast), ending exactly above the socket
    q_lift = steps[-1]["points"][-1][0]
    q_above = ik_best(above_sock, mo["pitch_tolerance"])
    steps.append({"type": "move", "name": "transit", "points": [(q_above, _joint_time(q_lift, q_above))]})
    a = Pose(*fk(q_above))
    steps.append({"type": "move", "name": "to_hover", "points": _line(a, hover, step, speed / 2, min_t, 0.0)})

    # insert: tiny steps, slow
    ins, t = [], 0.0
    for p in interpolate_line(hover, final, pl["descend_step"]):
        t += pl["descend_time_per_step"]
        ins.append((ik(p), t))
    steps.append({"type": "move", "name": "insert", "points": ins})
    steps.append({"type": "gripper", "name": "release", "cmd": "open"})
    steps.append({"type": "wait", "name": "release_wait", "sec": pl["release_delay"]})
    steps.append({"type": "move", "name": "retreat", "points": _line(final, hover, step, speed / 2, min_t, 0.0)})
    steps.append({"type": "move", "name": "home", "points": [(home, _joint_time(ik(hover), home))]})

    info = {"yaw_error_deg": math.degrees(yaw_error(peg, sock, SHAPE_SYM.get(peg["shape"], 0))),
            "peg_height": peg_h}
    return steps, info

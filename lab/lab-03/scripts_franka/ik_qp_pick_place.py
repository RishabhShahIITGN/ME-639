#!/usr/bin/env python3
"""
ik_qp_pick_place.py — Custom IK method 2: Quadratic Program with joint limits
=============================================================================
Same 7-phase pick-and-place as Task 9, but each IK step is a QP:

    min  1/2 ||J dq - e||^2 + 1/2 λ^2 ||dq||^2
    s.t. q_min <= q + IK_DT * dq <= q_max   (limited hinge/slide joints)

Usage:
    cd lab/lab-03
    source ../../venv/bin/activate
    python3 scripts_franka/ik_qp_pick_place.py
    python3 scripts_franka/ik_qp_pick_place.py --episodes 25 --seed 42 --no-gui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import qpsolvers

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ik_custom_core import (  # noqa: E402
    arm_dof_indices,
    compute_error,
    get_ee_jacobian,
    make_run_episode,
    pack_arm_velocity,
)
from ik_pick_place import EE_SITE, IK_DT, run_batch  # noqa: E402


def solve_ik_qp(model, data, target_pos, target_quat, site_name=EE_SITE, damping=1e-2):
    """One QP IK step with box joint-limit constraints on the 7 Franka arm DoFs."""
    dofs = arm_dof_indices(model)
    J = get_ee_jacobian(model, data, site_name)[:, dofs]
    
    # 1. CLAMP the task-space error before scaling by 1/IK_DT
    raw_err = compute_error(model, data, target_pos, target_quat, site_name)
    max_err = 0.02
    err_norm = np.linalg.norm(raw_err)
    if err_norm > max_err:
        raw_err = raw_err * (max_err / err_norm)
        
    err = raw_err / IK_DT
    n = len(dofs)

    # Ensure strictly positive definite Hessian for 7-DoF redundant arm:
    # P = J.T @ J + damping^2 * I_n
    P = J.T @ J + (damping ** 2) * np.eye(n)
    q = -J.T @ err

    # 2. STRICTLY CLAMP final commanded joint velocities to safe physical limits
    # Start with global velocity bounds (-2.0 to 2.0 rad/s)
    lb = np.full(n, -2.0)
    ub = np.full(n, 2.0)
    
    dof_to_local = {int(d): i for i, d in enumerate(dofs)}
    for i in range(model.njnt):
        if not model.jnt_limited[i]:
            continue
        joint_type = int(model.jnt_type[i])
        if joint_type not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            continue
        dof_adr = int(model.jnt_dofadr[i])
        if dof_adr not in dof_to_local:
            continue
            
        q_current = data.qpos[model.jnt_qposadr[i]]
        q_range = model.jnt_range[i]
        k = dof_to_local[dof_adr]
        
        # Intersect the physical joint position limits with the velocity limits
        joint_lb = (q_range[0] - q_current) / IK_DT
        joint_ub = (q_range[1] - q_current) / IK_DT
        
        lb[k] = max(-2.0, joint_lb)
        ub[k] = min(2.0, joint_ub)
        
        # Failsafe in case the arm is spawned slightly outside a limit
        if lb[k] > ub[k]:
            lb[k] = ub[k]

    dq_arm = qpsolvers.solve_qp(P, q, lb=lb, ub=ub, solver="daqp")
    if dq_arm is None:
        return None
    return pack_arm_velocity(model, dq_arm)


run_episode = make_run_episode(solve_ik_qp, solver_name="qp")


def main():
    parser = argparse.ArgumentParser(
        description="Custom IK pick-and-place: Quadratic Program with joint limits"
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()
    run_batch(
        run_episode,
        episodes=args.episodes,
        seed=args.seed,
        no_gui=args.no_gui,
        batch_prefix="ik_qp_batch",
        aggregate_name="ik_qp_pick_place_log.json",
        solver_name="qp",
    )


if __name__ == "__main__":
    main()
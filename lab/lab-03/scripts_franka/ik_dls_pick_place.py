#!/usr/bin/env python3
"""
ik_dls_pick_place.py — Custom IK method 1: Damped Least Squares
================================================================
Same 7-phase pick-and-place as Task 9, but each IK step is solved with
Damped Least Squares instead of Mink.

Usage:
    cd lab/lab-03
    source ../../venv/bin/activate
    python3 scripts_franka/ik_dls_pick_place.py
    python3 scripts_franka/ik_dls_pick_place.py --episodes 25 --seed 42 --no-gui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ik_custom_core import (  # noqa: E402
    arm_dof_indices,
    arm_qpos_indices,
    compute_error,
    get_ee_jacobian,
    make_run_episode,
    pack_arm_velocity,
)
from ik_pick_place import EE_SITE, IK_DT, run_batch  # noqa: E402


Q_DEFAULT_FRANKA = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def solve_ik_dls(
    model,
    data,
    target_pos,
    target_quat,
    site_name=EE_SITE,
    damping=1e-2,
    k_null: float = 0.5,  # REDUCED from 3.0 so the elbow stops fighting the hand
    q_default: np.ndarray | None = None,
):
    """One DLS IK step on the 7 Franka arm DoFs with null-space posture projection."""
    dofs = arm_dof_indices(model)
    n = len(dofs)
    J = get_ee_jacobian(model, data, site_name)[:, dofs]
    
    # 1. Compute raw error
    raw_err = compute_error(model, data, target_pos, target_quat, site_name)
    
    # 2. CLAMP the task-space error before scaling by 1/IK_DT
    # This stops the solver from panicking if the physical arm lags behind the target
    max_err = 0.02  
    err_norm = np.linalg.norm(raw_err)
    if err_norm > max_err:
        raw_err = raw_err * (max_err / err_norm)
        
    err = raw_err / IK_DT
    lambda_sq = damping ** 2

    # DLS pseudo-inverse J_pinv = (J^T J + lambda^2 I)^-1 J^T
    A = J.T @ J + lambda_sq * np.eye(n)
    J_pinv = np.linalg.solve(A, J.T)
    dq_task = J_pinv @ err

    # Null-space projection: N = I - J_pinv @ J
    if q_default is None:
        q_default = Q_DEFAULT_FRANKA if n == 7 else np.zeros(n)
    q_default = np.asarray(q_default, dtype=float)

    q_arm = data.qpos[arm_qpos_indices(model)]
    dq_null = k_null * (q_default[:n] - q_arm[:n])
    N = np.eye(n) - J_pinv @ J
    dq_arm = dq_task + N @ dq_null

    # 3. STRICTLY CLAMP final commanded joint velocities to safe physical limits
    dq_arm = np.clip(dq_arm, -2.0, 2.0)

    return pack_arm_velocity(model, dq_arm)

run_episode = make_run_episode(solve_ik_dls, solver_name="dls")


def main():
    parser = argparse.ArgumentParser(description="Custom IK pick-and-place: Damped Least Squares")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()
    run_batch(
        run_episode,
        episodes=args.episodes,
        seed=args.seed,
        no_gui=args.no_gui,
        batch_prefix="ik_dls_batch",
        aggregate_name="ik_dls_pick_place_log.json",
        solver_name="dls",
    )


if __name__ == "__main__":
    main()

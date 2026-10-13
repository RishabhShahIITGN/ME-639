#!/usr/bin/env python3
"""
ik_dls_pick_place.py — Custom IK method 1: Damped Least Squares
================================================================
Same 7-phase pick-and-place as Task 9, but each IK step is solved with
Damped Least Squares instead of Mink.

Usage:
    cd lab/lab-03
    source ../../venv/bin/activate
    python3 scripts/ik_dls_pick_place.py
    python3 scripts/ik_dls_pick_place.py --episodes 25 --seed 42 --no-gui
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
    compute_error,
    get_ee_jacobian,
    make_run_episode,
    pack_arm_velocity,
)
from ik_pick_place import IK_DT, run_batch  # noqa: E402


def solve_ik_dls(model, data, target_pos, target_quat, site_name="right_center", damping=1e-2):
    """One DLS IK step on the 6 arm DoFs. Returns a full-nv joint velocity."""
    dofs = arm_dof_indices(model)
    J = get_ee_jacobian(model, data, site_name)[:, dofs]
    # Scale pose error by 1/dt so integrating dq * dt closes the error this step.
    err = compute_error(model, data, target_pos, target_quat, site_name) / IK_DT
    lambda_sq = damping ** 2
    A = J.T @ J + lambda_sq * np.eye(len(dofs))
    dq_arm = np.linalg.solve(A, J.T @ err)
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

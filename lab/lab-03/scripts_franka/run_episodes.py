#!/usr/bin/env python3
"""
run_episodes.py — Task 8: Episode generation and randomization
==============================================================
Runs N episodes using the pick-and-place environment from Task 7.

For each episode:
- Resets the simulation.
- Samples a randomized cube pose (x, y, yaw) on the +Y half of the table.
- Logs cube pose, end-effector pose, and time at each step.
- Saves the log data to a NumPy file.
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
import mujoco

from env_pick_place import build_model, gravity_comp, ARM_ACTUATORS

def euler_z_to_quat(yaw):
    """Convert yaw (rotation around Z) to a quaternion [w, x, y, z]."""
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

def main():
    parser = argparse.ArgumentParser(description="Task 8: Episode generation")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Building environment for {args.episodes} episodes...")
    model, data = build_model()
    rng = np.random.default_rng(args.seed)

    # Output directory for logs
    out_dir = Path(__file__).resolve().parent.parent / "logs"
    out_dir.mkdir(exist_ok=True)

    # Retrieve table limits
    with open(Path(__file__).resolve().parent.parent / "table_design.json") as f:
        design = json.load(f)
    
    table_h = design["table_height_m"]
    xmin = design["table_x_min_m"] + 0.04
    xmax = design["table_x_max_m"] - 0.04
    # Spawning on the positive Y side (avoiding the tray on the -Y side)
    ymin = 0.04
    ymax = design["table_y_max_m"] - 0.04

    # Pre-find IDs
    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_qpos_adr = model.jnt_qposadr[cube_jnt_id]
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")

    print(f"Running {args.episodes} episodes (Cube spawn limits: X=[{xmin:.2f}, {xmax:.2f}], Y=[{ymin:.2f}, {ymax:.2f}])...")

    for ep in range(args.episodes):
        mujoco.mj_resetData(model, data)

        # 1. Sample new pose
        spawn_x = rng.uniform(xmin, xmax)
        spawn_y = rng.uniform(ymin, ymax)
        spawn_yaw = rng.uniform(-np.pi, np.pi)
        spawn_z = table_h + 0.025 + 0.005 # Table height + cube half + small drop

        # 2. Update qpos
        data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = [spawn_x, spawn_y, spawn_z]
        data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7] = euler_z_to_quat(spawn_yaw)

        # 3. Initialize logs for this episode
        log_time = np.zeros(args.steps)
        log_cube_pos = np.zeros((args.steps, 3))
        log_cube_quat = np.zeros((args.steps, 4))
        log_ee_pos = np.zeros((args.steps, 3))
        
        # 4. Step loop
        for step in range(args.steps):
            gravity_comp(model, data, ARM_ACTUATORS)
            mujoco.mj_step(model, data)
            
            # Record state
            log_time[step] = data.time
            log_cube_pos[step] = data.xpos[cube_body_id]
            log_cube_quat[step] = data.xquat[cube_body_id]
            log_ee_pos[step] = data.xpos[ee_body_id]

        # 5. Save logs
        log_file = out_dir / f"episode_{ep:03d}.npz"
        np.savez(
            log_file,
            time=log_time,
            cube_pos=log_cube_pos,
            cube_quat=log_cube_quat,
            ee_pos=log_ee_pos,
            spawn_x=spawn_x,
            spawn_y=spawn_y,
            spawn_yaw=spawn_yaw
        )
        print(f"  Episode {ep:03d} | Spawn: ({spawn_x:.3f}, {spawn_y:.3f}, {spawn_yaw:+.2f} rad) -> Saved {log_file.name}")

    print("All episodes completed.")

if __name__ == "__main__":
    main()

import mujoco
import mujoco.viewer
import time
import os
import numpy as np

def main():
    original_dir = os.getcwd()
    target_dir = "ITR_mujoco_fk_lab/robot_descriptions"
    
    if not os.path.basename(original_dir) == "robot_descriptions":
        if os.path.exists(target_dir):
            os.chdir(target_dir)
        else:
            raise FileNotFoundError(f"Could not find '{target_dir}'. Please run from 'lab/'.")
            
    MODEL_PATH = "franka_scene.xml"
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # We will compute Forward Kinematics (FK) for the end effector
    # In Franka, the end effector body is typically 'hand' or 'link7'
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if ee_body_id == -1:
        ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    
    # Gains for a simple PD controller since it uses effort actuators (<motor>)
    # These are rough gains to keep the arm from collapsing
    Kp = np.array([50.0, 50.0, 50.0, 50.0, 20.0, 20.0, 10.0, 10.0])
    Kd = np.array([5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0])
    
    # Target joint positions (home position roughly)
    q0 = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.0])
    
    print("Launching Franka simulation (7-DOF)...")
    last_print_time = time.time()
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Create a gentle sine wave trajectory for the joints
            t = data.time
            q_target = q0.copy()
            q_target[0] += np.sin(t) * 0.5  # Sweep base joint
            q_target[3] += np.cos(t) * 0.3  # Sweep elbow
            
            # Simple PD control
            # Ensure we only apply control to the available actuators
            nu = model.nu
            for i in range(min(nu, len(q0))):
                error = q_target[i] - data.qpos[i]
                error_dot = 0.0 - data.qvel[i]
                data.ctrl[i] = Kp[i]*error + Kd[i]*error_dot
                
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Print Forward Kinematics (FK) periodically
            if time.time() - last_print_time > 0.5:
                ee_pos = data.xpos[ee_body_id]
                print(f"[Franka FK] End-Effector Position: X={ee_pos[0]:.3f}, Y={ee_pos[1]:.3f}, Z={ee_pos[2]:.3f}")
                last_print_time = time.time()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()

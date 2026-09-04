"""
02_verify_skew_properties.py -- HW1 Part 2, Task 2: verify the
skew-symmetric identities from Problem 5 in simulation.

Logs R(t) from a MuJoCo body under time-varying angular velocity and
numerically checks, for random v, w, omega in R^3:
    R (v x w) == (R v) x (R w)                 [Problem 5a]
    R [omega] R^T == [R omega]                 [Problem 5b]
using utils.hat() for the skew-symmetric (hat) operator.

This is a numerical sanity check, not a substitute for the hand proof.
"""
import time # Add this at the top of your script

# ... [rest of your code] ...

def main():
    np.set_printoptions(precision=4, suppress=True)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed=0)

    mujoco.mj_forward(model, data)

    logged_R = []
    global_max_cross = 0.0
    global_max_skew = 0.0

    print(f"{'step':>5} {'t (s)':>8} {'max resid: R(vxw)=(Rv)x(Rw)':>28} {'max resid: R[w]R^T=[Rw]':>24}")
    
    # LAUNCH THE VIEWER HERE
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for log_i in range(N_LOGGED_STEPS):
            for _ in range(STEPS_BETWEEN_LOGS):
                data.qvel[3:6] = time_varying_angular_velocity(data.time)
                mujoco.mj_step(model, data)
                
                # Sync the viewer to see the movement
                viewer.sync()
                time.sleep(model.opt.timestep) # Slow it down to real-time

            R = get_body_orientation(data)
            logged_R.append(R.copy())

            print(f"\nR(t) at t = {data.time:.4f} s:")
            print(R)

            assert is_close_to_identity(R @ R.T, tol=1e-6), "R is not orthonormal!"

            resid_cross, resid_skew = check_identities(R, rng)
            global_max_cross = max(global_max_cross, resid_cross)
            global_max_skew = max(global_max_skew, resid_skew)
            print(f"{log_i:5d} {data.time:8.3f} {resid_cross:28.3e} {resid_skew:24.3e}")
            
            # Pause briefly so you can read the log before it spins again
            time.sleep(1.0) 
            
# ... [rest of the function remains the same] ...
import numpy as np
import mujoco

from utils import hat, get_body_orientation, is_close_to_identity

MODEL_PATH = "../model/asymmetric_body.xml"

N_CHECKS_PER_STEP = 5     # how many random (v, w, omega) triples per logged step
N_LOGGED_STEPS = 5        # how many simulated time points to check
STEPS_BETWEEN_LOGS = 200  # sim steps to advance between each logged check


def skew(omega):
    """3-vector -> 3x3 skew-symmetric matrix. Same as utils.hat()."""
    return hat(omega)


def time_varying_angular_velocity(t):
    """Sine-wave angular velocity (rad/s) so omega is not constant."""
    return np.array([
        np.sin(t),
        0.5 * np.sin(2.0 * t),
        np.cos(0.7 * t),
    ])


def check_identities(R, rng):
    """Numerically check Problem 5 identities at a fixed R.

    For N_CHECKS_PER_STEP random vectors v, w, omega, compute residuals of:
        R @ (v x w)  vs.  (R v) x (R w)
        R @ [omega] @ R.T  vs.  [R omega]
    Return the worst-case (max) residual for each identity.
    """
    max_residual_cross = 0.0
    max_residual_skew = 0.0

    for _ in range(N_CHECKS_PER_STEP):
        v = rng.normal(size=3)
        w = rng.normal(size=3)
        omega = rng.normal(size=3)

        lhs_cross = R @ np.cross(v, w)
        rhs_cross = np.cross(R @ v, R @ w)
        resid_cross = np.linalg.norm(lhs_cross - rhs_cross)
        max_residual_cross = max(max_residual_cross, resid_cross)

        lhs_skew = R @ hat(omega) @ R.T
        rhs_skew = hat(R @ omega)
        resid_skew = np.linalg.norm(lhs_skew - rhs_skew)
        max_residual_skew = max(max_residual_skew, resid_skew)

    return max_residual_cross, max_residual_skew


def main():
    np.set_printoptions(precision=4, suppress=True)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed=0)

    mujoco.mj_forward(model, data)

    logged_R = []
    global_max_cross = 0.0
    global_max_skew = 0.0

    print(f"{'step':>5} {'t (s)':>8} {'max resid: R(vxw)=(Rv)x(Rw)':>28} {'max resid: R[w]R^T=[Rw]':>24}")
    for log_i in range(N_LOGGED_STEPS):
        for _ in range(STEPS_BETWEEN_LOGS):
            data.qvel[3:6] = time_varying_angular_velocity(data.time)
            mujoco.mj_step(model, data)

        R = get_body_orientation(data)
        logged_R.append(R.copy())

        print(f"\nR(t) at t = {data.time:.4f} s:")
        print(R)

        assert is_close_to_identity(R @ R.T, tol=1e-6), "R is not orthonormal!"

        resid_cross, resid_skew = check_identities(R, rng)
        global_max_cross = max(global_max_cross, resid_cross)
        global_max_skew = max(global_max_skew, resid_skew)
        print(f"{log_i:5d} {data.time:8.3f} {resid_cross:28.3e} {resid_skew:24.3e}")

    print("\nMaximum residual across all time steps:")
    print(f"  Identity 1  R(v x w) = (Rv) x (Rw) : {global_max_cross:.3e}")
    print(f"  Identity 2  R[ω]R^T = [Rω]         : {global_max_skew:.3e}")

    print(
        """
Why this simulation does not prove the identities:

A residual on the order of 1e-16 is consistent with float64 rounding
error, not with an exact algebraic identity holding in the computer.
Each multiply/add in forming R, the cross products, and the hat maps
truncates a real number to a finite mantissa, so the two sides of an
identity that are equal in SO(3) typically differ by a few units in
the last place. Checking a handful of random (v, w, omega) at a
handful of simulated R(t) only samples a finite set of cases; it
cannot cover every vector in R^3 or every rotation in SO(3). A
general proof (the paper derivation of Problem 5) is what establishes
the identities for all R and all vectors. Simulation is a sanity
check that our implementation matches that algebra within machine
precision.
"""
    )


if __name__ == "__main__":
    main()

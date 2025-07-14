import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from _utils.load_driving_cycle import load_driving_cycle
from _utils.utils import save_trajectory_plot, calc_req_input_from_acc, load_yaml_file
    

def gen_cacc_fv(t, y, fn_x_prec, fn_v_prec, sim_params, phy_params, noise:bool = False):
    h = sim_params["h"]
    d_min = sim_params["d_min"]
    kp = sim_params["kp"]
    kd = sim_params["kd"]
    ki = sim_params["ki"]
    a_max = phy_params["a_max"]
    a_min = phy_params["a_min"]

    # noise is now a tuple of (noise_x, noise_v, noise_a, t_sim) or False
    if noise is False:
        noise_x = 0.0
        noise_v = 0.0
        noise_a = 0.0
    else:
        noise_x, noise_v, noise_a, t_sim = noise
        # Find the index for the current time
        idx = np.searchsorted(t_sim, t)
        noise_x = noise_x[idx]
        noise_v = noise_v[idx]
        noise_a = noise_a[idx]

    x = y[0] + noise_x
    v = y[1] + noise_v
    e_int = y[2]
    x_prec = fn_x_prec(t) + noise_x
    v_prec = fn_v_prec(t) + noise_v

    d_target = d_min + h*v
    d = x_prec - x
    e = d - d_target
    a_target = kp*e + kd*(v_prec-v) * ki*e_int + noise_a
    a_clipped = np.clip(a_target, a_min, a_max)

    return np.array([v, a_clipped, e]) #dxdt, dvdt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a follower vehicle using CACC.")
    parser.add_argument(
        "driving_cycle_name",
        type=str,
        help="Name of driving cycle file (e.g. 'udds.txt') to use as leader behavior",
    )
    args = parser.parse_args() 
    traj_id: str = args.driving_cycle_name.split('.')[0]
    
    #load config file
    config = load_yaml_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yml'))
    print(config)
    PHYSICS_PARAMS = config["PHYSICS_PARAMS"]
    SIM_PARAMS = config["SIM_PARAMS"]

    dt = SIM_PARAMS["dt"]
    #load a driving cycle to serve as the leader
    t_sim, fn_leader_x, fn_leader_v = load_driving_cycle("../data/driving_cycles/"+args.driving_cycle_name, 
                                                         dt, b_plot=False)
    fv_X0 = np.array([fn_leader_x(0)-SIM_PARAMS["d_ini"], fn_leader_v(0), 0]) #same leader velocity but with an initial gap
    print("[INITIAL CONDITIONS]:\n")
    print(f"LV: {fn_leader_x(0)} m, {fn_leader_v(0)/3.6} km/h")
    print(f"FV0: {fv_X0[0]} m, {fv_X0[1]/3.6} km/h")


    # Pre-generate noise arrays for reproducibility
    np.random.seed(42)
    noise_x = np.random.normal(0, SIM_PARAMS["noise_std"]["x"], size=len(t_sim))
    noise_v = np.random.normal(0, SIM_PARAMS["noise_std"]["v"], size=len(t_sim))
    noise_a = np.random.normal(0, SIM_PARAMS["noise_std"]["a"], size=len(t_sim))

    # Simulate with fixed noise sequence
    ret = solve_ivp(
        gen_cacc_fv,
        t_span=(t_sim[0], t_sim[-1]),
        y0=fv_X0,
        t_eval=t_sim,
        args=(fn_leader_x, fn_leader_v, SIM_PARAMS, PHYSICS_PARAMS, (noise_x, noise_v, noise_a, t_sim))
    )
    print(f"[Follower Trajectory {traj_id}] {ret.message}")

    # Recompute acceleration (a_true) to store it for labels, but also v_true
    a_true = []
    for i in range(len(ret.t)):
        t = ret.t[i]
        y = ret.y[:, i]
        # Use zero noise for true acceleration
        _, a, _ = gen_cacc_fv(
            t, y, fn_leader_x, fn_leader_v, SIM_PARAMS, PHYSICS_PARAMS, False
        )
        a_true.append(a)
    a_true = np.array(a_true)

    fvo_x_sim = ret.y[0,:]
    fv0_v_sim = ret.y[1,:]
    lv_x_sim = fn_leader_x(ret.t) + noise_x
    lv_v_sim =  fn_leader_v(ret.t) + noise_v
    d_sim = lv_x_sim - fvo_x_sim
    d_target_sim = SIM_PARAMS["d_min"] + SIM_PARAMS["h"]*fv0_v_sim
    a_ref = calc_req_input_from_acc(ret.y[1,:], a_true, PHYSICS_PARAMS)

    data = {
        't': ret.t,
        'fv0_x': fvo_x_sim,
        'fv0_v': fv0_v_sim,
        'fv0_u': a_ref, #required input (acc reference)
        'fv0_a': a_true, #output acceleration (true)
        'lv_x': lv_x_sim,
        'lv_v': lv_v_sim,
        'd_fv0_lv': d_sim,
        'd*_fv0_lv': d_target_sim,
    }

    save_trajectory_plot(data, f"../data/driving_cycles/CACC_{traj_id}.csv", traj_id)


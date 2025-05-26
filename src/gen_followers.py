import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from config import PHYSICS_PARAMS, SIM_PARAMS
from _utils.load_driving_cycle import load_driving_cycle
from utils import save_trajectory_plot, calc_req_input_from_acc #TODO: refactor this

def gen_cacc_fv(t, y, fn_x_prec, fn_v_prec, sim_params, phy_params):
    h = sim_params["h"]
    d_min = sim_params["d_min"]
    kp = sim_params["kp"]
    kd = sim_params["kd"]
    ki = sim_params["ki"]

    a_max = phy_params["a_max"]
    a_min = phy_params["a_min"]

    x = y[0] 
    v = y[1]
    e_int = y[2]
    x_prec=fn_x_prec(t)
    v_prec=fn_v_prec(t)

    d_target = d_min + h*v
    d = x_prec-x
    e = d-d_target
    a_target = kp*e + kd*(v_prec-v) * ki*e_int
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
    dt = SIM_PARAMS["dt"]
    #load a driving cycle to serve as the leader
    t_sim, fn_leader_x, fn_leader_v = load_driving_cycle("../data/driving_cycles/"+args.driving_cycle_name, 
                                                         dt, b_plot=False)
    fv_X0 = np.array([fn_leader_x(0)-SIM_PARAMS["d_ini"], fn_leader_v(0), 0]) #same leader velocity but with an initial gap
    print("[INITIAL CONDITIONS]:\n")
    print(f"LV: {fn_leader_x(0)} m, {fn_leader_v(0)/3.6} km/h")
    print(f"FV0: {fv_X0[0]} m, {fv_X0[1]/3.6} km/h")

    ret = solve_ivp(gen_cacc_fv, t_span=(t_sim[0], t_sim[-1]), y0=fv_X0,
          t_eval=t_sim, args=(fn_leader_x, fn_leader_v, SIM_PARAMS, PHYSICS_PARAMS))
    print(f"[Follower Trajectory {traj_id}] {ret.message}")

    # Recompute acceleration (a_eff) to store 
    a_eff = []
    for i in range(len(ret.t)):
        t = ret.t[i]
        y = ret.y[:, i]
        _, a, _ = gen_cacc_fv(t, y, fn_leader_x, fn_leader_v, SIM_PARAMS, PHYSICS_PARAMS)
        a_eff.append(a)
    a_eff = np.array(a_eff)

    fvo_x_sim = ret.y[0,:]
    fv0_v_sim = ret.y[1,:]
    lv_x_sim = fn_leader_x(ret.t)
    lv_v_sim =  fn_leader_v(ret.t)
    d_sim = lv_x_sim - fvo_x_sim
    d_target_sim = SIM_PARAMS["d_min"] + SIM_PARAMS["h"]*fv0_v_sim
    u = calc_req_input_from_acc(ret.y[1,:], a_eff, PHYSICS_PARAMS)
    data = {
        't': ret.t,
        'fv0_x': fvo_x_sim,
        'fv0_v': fv0_v_sim,
        'fv0_v_kmh': fv0_v_sim*3.6,
        'fv0_u': u, #required input (FORCE)
        'fv0_a': a_eff, #output acceleration
        'lv_x': lv_x_sim,
        'lv_v': lv_v_sim,
        'd': d_sim,
        'd_target': d_target_sim
    }

    save_trajectory_plot(data, f"../data/CACC_{traj_id}.csv", traj_id)


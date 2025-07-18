import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from _utils.load_driving_cycle import load_driving_cycle
from _utils.utils import save_trajectory_plot, second_order_model, load_yaml_file

#TODO: NOISE IMPLEMENTATION NOT WORKING

def apply_control(kp: float, kd: float, ki: float, e: float,
          e_int:float, v_prec: float, v: float, phy_params: dict, noise: float):
    m = phy_params['m']
    u = (kp*e + kd*(v_prec-v) + ki*e_int + noise)*m #input force + actuator noise
    #being conservative here since this is output acc constraint
    u_clipped = np.clip(u, phy_params['a_min']*m, phy_params['a_max']*m) 
    a_output = second_order_model(v, u_clipped, phy_params)
    return u_clipped, a_output

def gen_cacc_fv(t, y, fn_x_prec, fn_v_prec, sim_params, phy_params, noise:bool|tuple):
    h = sim_params["h"]
    d_min = sim_params["d_min"]
    kp = sim_params["kp"]
    kd = sim_params["kd"]
    ki = sim_params["ki"]
    
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

    #policy parameters depend on noisy measurements
    x_noise = y[0] + noise_x
    v_noise = y[1] + noise_v
    e_int = y[2]
    x_prec = fn_x_prec(t) + noise_x
    v_prec = fn_v_prec(t) + noise_v

    d_target = d_min + h*v_noise
    d = x_prec - x_noise
    e = d - d_target
    
    #but here the plant needs a ground truth speed to apply physics 
    u, a_output = apply_control(kp=kp, kd=kd, ki=ki, e=e, e_int=e_int, v_prec=v_prec,
                                v=y[1], phy_params=phy_params, noise=0)
    ret = {'u': u, 'a_output': a_output, 'e': e, 'e_int': e_int, 
           'x': y[0], 'v': y[1], 'x_prec': x_prec, 'v_prec': v_prec,
           'd': d, 'd_ref': d_target}
    return ret

def integrate(t, y, fn_x_prec, fn_v_prec, sim_params, phy_params, noise:bool|tuple):
    '''
    Serves as an interface for solve_ivp(). gen_cacc_fv() retuns all necessary variables
    '''
    ret = gen_cacc_fv(t, y, fn_x_prec, fn_v_prec, sim_params, phy_params, noise)
    return np.array([ret['v'], ret['a_output'], ret['e']]) #dxdt, dvdt, integral action

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
        integrate,
        t_span=(t_sim[0], t_sim[-1]),
        y0=fv_X0,
        t_eval=t_sim,
        args=(fn_leader_x, fn_leader_v, SIM_PARAMS, PHYSICS_PARAMS, False) #TODO: NOISE 
    )
    print(f"[Follower Trajectory {traj_id}] {ret.message}")

    reconstructed_steps = [
        gen_cacc_fv(
            t_step, 
            ret.y[:, i], 
            fn_leader_x, 
            fn_leader_v, 
            SIM_PARAMS, 
            PHYSICS_PARAMS, 
            False # No noise for reconstruction 
        )
        for i, t_step in enumerate(ret.t)
    ]

    df_results = pd.DataFrame(reconstructed_steps)

    data = {
        't': ret.t,
        'fv0_x_noise': ret.y[0, :] + noise_x, #!DEBUG:adding a posteriori for now
        'fv0_v_noise': ret.y[1, :] + noise_v,
        'fv0_x': df_results['x'].to_numpy(),
        'fv0_v': df_results['v'].to_numpy(),
        'fv0_u': df_results['u'].to_numpy(),
        'fv0_a': df_results['a_output'].to_numpy(),
        'fv0_a_noise': df_results['a_output'].to_numpy() + noise_a,
        'lv_x': df_results['x_prec'].to_numpy(),
        'lv_v': df_results['v_prec'].to_numpy(),
        'd_fv0_lv': df_results['d'].to_numpy(),
        'd*_fv0_lv': df_results['d_ref'].to_numpy(),
    }
    
    # Also compute the equivalent input acceleration from u
    data['fv0_a_ref'] = data['fv0_u'] / PHYSICS_PARAMS['m']
    figpath = f'../data/driving_cycles/figs/CACC_{traj_id}_kp{SIM_PARAMS["kp"]}_ki{SIM_PARAMS["ki"]}_kd{SIM_PARAMS["kd"]}'
    csvpath = f'../data/driving_cycles/CACC_{traj_id}_kp{SIM_PARAMS["kp"]}_ki{SIM_PARAMS["ki"]}_kd{SIM_PARAMS["kd"]}'
    save_trajectory_plot(data, csvpath, figpath, traj_id)


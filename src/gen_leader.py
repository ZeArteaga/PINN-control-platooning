import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from config import PHYSICS_PARAMS, SIM_PARAMS
from utils import save_trajectory_plot, calc_req_input_from_acc

def vehicle_int(t, y, a_target_fn, params: dict):
    a_max = params["a_max"]
    a_min = params["a_min"]
    v=y[1]
    a_target=a_target_fn(t)
    a_eff = np.clip(a_target, a_min, a_max)

    dxdt = v
    dvdt = a_eff

    return np.array([dxdt, dvdt])

if __name__ == "__main__":
    dt = SIM_PARAMS["dt"]

    t_samp = np.linspace(0, t_end, 10, dtype=np.float32)

    #DEFINE U PROFILES
    target_profiles = []
    target_profiles.append(np.full_like(t_samp, 0, dtype=np.float32)) #crusing
    target_profiles.append(np.array([5, 5, 5, 5, 5, 5, -2, -2, 1, 1], dtype=np.float32)) #emerg braking
    target_profiles.append(np.array([2, 8, 0, 0, 1, 1, 0.7, 0.7, 0.7, 0.7], dtype=np.float32)) #accelerating
    target_profiles.append(np.array([0.5, 0.1]*5, dtype=np.float32)) #stop&go/oscillating
    np.random.seed(31)
    target_profiles.append(np.random.uniform(low=-0.1, high=0.6, size=t_samp.shape).astype(np.float32)) #random

    input_interp = []
    for profile in target_profiles:
        input_interp.append(interp1d(t_samp, profile, kind='quadratic'))

    for id, a_target in enumerate(input_interp):
        ret = solve_ivp(vehicle_int, t_span=t_span, y0=SIM_PARAMS["X0"],
            t_eval=np.arange(0, t_end+dt, dt), args=(a_target,PHYSICS_PARAMS)) # evalute and store at regular dt intervals, RK45 is default
        print(f"[Trajectory {id}] {ret.message}")
        a_eff = np.clip(a_target(ret.t), PHYSICS_PARAMS["a_min"], PHYSICS_PARAMS["a_max"])
        u = calc_req_input_from_acc(ret.y[1,:], a_eff, PHYSICS_PARAMS)

        data = {
            'traj_id': id,
            't': ret.t,
            'x': ret.y[0,:],
            'v': ret.y[1,:],
            'v_kmh': ret.y[1,:]*3.6,
            'u': u, #required input (FORCE)
            'a': a_eff #output acceleration
        }

        save_trajectory_plot(data, path=f"../data/leader_profile_{id}.csv")

        

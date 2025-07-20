import numpy as np
import os
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from _utils.utils import save_trajectory_plot, load_yaml_file, second_order_model

def integrate(t, y, u_fn, params):
    v = y[1]
    a_min = params['a_min']
    a_max = params['a_max']
    m = params['m']
    u = np.clip(u_fn(t), a_min*m, a_max*m)
    a = second_order_model(v, u, params)
    #dynamics
    dxdt = v
    dvdt = a
    return np.array([dxdt, dvdt])

if __name__ == "__main__":
    t_end = 30 # seconds
    config = load_yaml_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yml'))
    print(config)
    PHYSICS_PARAMS = config["PHYSICS_PARAMS"]
    SIM_PARAMS = config["SIM_PARAMS"]
    dt = SIM_PARAMS['dt']
    X0 = (0, 0)

    t_samp = np.linspace(0, t_end, 10, dtype=np.float32)
    # DEFINE ACCELERATION PROFILES
    a_profiles = []
    a_profiles.append(np.array([3, 1, 1.5, 0, 0, 0, 2, 1, 0, 0.1], dtype=np.float32)) # cruisng
    a_profiles.append(np.array([1]*10)) # accelerating
    a_profiles.append(np.array([0.8, -0.1]*5, dtype=np.float32)) # oscillating
    a_profiles.append(np.array([2, 2, 2, 0.5, 0.1, 0.2, 0.2, 0, 0, -10], dtype=np.float32)) #em. brake

    # Convert acceleration profiles to force profiles
    m = PHYSICS_PARAMS["m"]
    u_profiles = [profile * m for profile in a_profiles]
    input_interp = []
    for profile in u_profiles:
        input_interp.append(interp1d(t_samp, profile, kind='quadratic'))

    for id, u_fn in enumerate(input_interp):
        ret = solve_ivp(integrate, t_span=(0, t_end), y0=X0,
            t_eval=np.arange(0, t_end+dt, dt), args=(u_fn, PHYSICS_PARAMS))
        print(f"[Trajectory {id}] {ret.message}")
        v = ret.y[1,:]
        a = []
        for k in range(len(ret.t)):
            a.append(second_order_model(v[k], u_fn(k*dt), PHYSICS_PARAMS))
        data = {
            'traj_id': id,
            't': ret.t,
            'fv0_x': ret.y[0,:],
            'fv0_v': v,
            'fv0_v_noise': v + np.random.normal(0, scale=SIM_PARAMS['noise_std']['v'], size=len(ret.t)),
            'fv0_u': u_fn(ret.t),
            'fv0_a_ref': u_fn(ret.t)/m,
            'fv0_a': a #output acceleration (after actuator lag and clipping)
        }

        save_trajectory_plot(data,
            csvpath=f"../data/driving_cycles/car_profile_{id}",
            figpath=None,
            traj_id=str(id))

        

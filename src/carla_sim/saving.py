import os 
import pickle
import numpy as np
from mpc.utils import from_mpc_data_to_dict
from .Core import Vehicle, Platoon

def save_follower_data(sim_dt: float, control_dt: float, platoon: Platoon):
    results_dir = os.path.join(os.path.dirname(__file__), 'results/')
    vehicle: Vehicle
    for i, vehicle in enumerate(platoon):
        path = os.path.join(results_dir, f"vehicle_{i}.pkl")

        n_points_sim = len(vehicle.history["acc"])
        time_sim = (np.arange(n_points_sim) * sim_dt).reshape(-1, 1)
        sim_dict = vehicle.history.copy()
        sim_dict["time"] = time_sim

        output= {
            'sim': sim_dict,
            'sim_dt': sim_dt
        }
        #mpc_dict = follower.controller.data.export() #! not working correctly
        if i > 0: #followers
            sim_dict["index"] = platoon.gap_hist[str(vehicle.id)]["index"]
            sim_dict["d"] = platoon.gap_hist[str(vehicle.id)]["gap"]
            n_points_mpc = len(vehicle.long_mpc.data['_x', 'v'])
            time_mpc = (np.arange(n_points_mpc) * control_dt).reshape(-1, 1)
            mpc_dict = {}
            mpc_dict = from_mpc_data_to_dict(mpc_dict, vehicle.long_mpc, ['aux', 'tvp', 'x', 'u'])
            mpc_dict['time'] = time_mpc #add time entry

            output['mpc'] = mpc_dict
            output['control_dt'] = control_dt
            
        with open(path, 'wb') as f:
            pickle.dump(output, f)
        print(f"Saved data for vehicle {i} to {results_dir}")
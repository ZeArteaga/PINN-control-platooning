import argparse
import random
import pickle
import time 
import numpy as np
import os
import carla
from .agents.navigation.controller import PIDLongitudinalController

from do_mpc.model import Model
from do_mpc.data import save_results, Data

from .Core import *
from mpc.controller import setupDMPC
from mpc.modelling import SecondOrderPINNmodel
from mpc.utils import from_mpc_data_to_dict

#*This script assumes an already active server with a picked town: 
#* ./config.py --map Town05
#* then in carla root run ./CarlaUE4.sh (optionally: --quality -low-quality)

def main(n_followers: int, mpc_model: Model, opt_params, mpc_config,
         fn_get_prec_state, acc_cons: list, sim_dt:float=0.01, t_end:float=np.inf, 
          host='localhost', port=2000, render:bool=True):
    """
    Args:
            n_followers: number of platoon following vehicles to add (not counting the leader)
			sim_dt: length of a simulation time step
            control_rate: defines the rate in steps of the controller (e.g 2 -> every 2 steps of simulation, each one sim_dt time
			t_end: Defaults to np.inf to run indefinetly
			host: Carla server host
			port: Carla server port
			render: Turns rendering on (True) or off (False)
		"""
    
    actor_list = []
    SEED = 31

    try:
        #*GET CLIENT, WORLD, TRAFFIC MANAGER
        sim = Simulation(host, port,
                          large_map=False, dt=sim_dt, synchronous=True, render=render)
        print(f"Successfully connected to Carla. Current map: {sim.get_map().name}")

        spect = sim.get_spectator()
        tm = sim.get_trafficmanager(port=8000)
        tm_port = tm.get_port()
        tm.set_random_device_seed(SEED) #for simulation determinism
        print("Got Traffic Manager.")

        #*PICK VEHICLE
        vehicle_bp_lib = sim.get_vehicle_blueprints()
        lv_bp = vehicle_bp_lib.filter('vehicle.mini.cooper_s_2021')[0] #returns a list so we pick the only element

        #*SPAWN LEAD VEHICLE AND ADD TO ACTOR LIST AND PLATOON
        spawn_points = sim.get_map().get_spawn_points()
        if not spawn_points:
            print("Could not retrieve spawn points from map!")
            return
        
        lv_sp = spawn_points[1]
        platoon = Platoon(sim)
        lv: Vehicle = platoon.add_lead_vehicle(lv_bp, lv_sp)
        #DONT APPEND TO ACTORS LIST, THIS ONLY FOR NPCs
        print(f"Spawned LV: {lv.type_id} (id: {lv.id}) at {lv_sp.location}")
        sim.tick()

        #* SPAWN FOLLOWERS
        for i in range(0, n_followers):
            followers = platoon.get_follower_list()
            fv: Vehicle = platoon.add_follower_vehicle(lv_bp, (lv.transform_ahead(-10, force_straight=True) if i == 0
                                                else followers[-1].transform_ahead(-10, force_straight=True)))
            #*Setup controllers
            fv_mass = fv.get_physics_control().mass
            opt_params["u_max"] = acc_cons[1]*fv_mass
            opt_params["u_min"] = acc_cons[0]*fv_mass
            mpc = setupDMPC(mpc_model, mpc_config, opt_params, fn_get_prec_state, platoon, fv)
            print(f"\n FV{i} controller settings:", mpc.settings)
            mpc.set_initial_guess()
            pid = PIDLongitudinalController(fv, dt=sim_dt,
                                             K_P=8, K_I=0.6, K_D=0.5)
            fv.attach_controller(mpc, pid)
            print(f"Spawned FV: {fv.type_id} (id: {fv.id})")
            sim.tick()
            sim.tick()

        #*SIMULATING
        print("Running simulation loop...")
        for _ in range(0, int(5/sim_dt)):
                sim.tick() #tick 5 seconds until the spawned vehicles stabilize
        
        lv.set_autopilot(True, tm_port) #TODO: Modify default autopilot behavior
        i:int = 0
        if t_end != np.inf: 
            step_end = int(t_end/sim_dt)
        else:
            step_end = np.inf

        control_dt = mpc.settings.t_step
        control_rate = int(control_dt/sim_dt)
        while i<=step_end:
            print(f"[t={sim_dt*i}]\n")
            if i % control_rate == 0:
                sim.run_step(platoon, "control", control_dt)
            else:
                sim.run_step(platoon)
            i += 1

        sim.release_synchronous()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Simulation finished.")
        
        #*Save data
        if 'platoon' in locals() and len(platoon) > 0:
            results_dir = os.path.join(os.path.dirname(__file__), 'results/')
            follower: Vehicle
            for i, follower in enumerate(platoon.get_follower_list()):
                path = os.path.join(results_dir, f"follower_{i}.pkl")
                #add time manually so that do_mpc plotting works
                n_points = len(follower.controller.data['_x', 'v'])
                time_arr = (np.arange(n_points) * control_dt).reshape(-1, 1)
                #mpc_dict = follower.controller.data.export() #! not working correctly
                mpc_dict = {}
                mpc_dict['_time'] = time_arr #add time entry
                mpc_dict = from_mpc_data_to_dict(mpc_dict, follower.controller, ['aux', 'tvp', 'x', 'u'])
                
                with open(path, 'wb') as f:
                    pickle.dump(mpc_dict, f)
                print(f"Saved data for follower {i} to {results_dir}")

        print("Cleaning up...")
        if 'sim' in locals():
            world = sim.get_world()
            if world:
            # Restore original settings (disable sync mode)
                print("Restoring original world settings.")
                world.apply_settings(sim.get_original_settings())
        if 'platoon' in locals() and len(platoon) > 0:
            sim.apply_batch([carla.command.DestroyActor(v.id) for v in platoon])
            print(f"Destroying platoon with {len(platoon)} vehicles.")
        if actor_list:
            print(f"Destroying {len(actor_list)} actors.")
            sim.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            # A small delay to ensure actors are destroyed before script exits
            time.sleep(0.5) 
        print("Cleanup finished.")
        return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("n_followers", type=int, help="Number of followers to add behind the platoon leader")
    parser.add_argument("--sim-dt", type=float, default=0.01, help="Simulation time step")
    parser.add_argument("--initial-gap")
    parser.add_argument("--t-end", type=float, default=np.inf, help="Simulation end time")
    parser.add_argument("--host", type=str, default='localhost', help="Carla server host")
    parser.add_argument("--port", type=int, default=2000, help="Carla server port")
    parser.add_argument("--no-render", action="store_false", help="Enable rendering")
    
    parser.add_argument("--control-rate", type=int, default=10, help="Control rate (steps)")
    parser.add_argument("--n-horizon", type=int, default=15, help="MPC: Prediction horizon length (steps)")
    parser.add_argument("--Q", type=float, nargs=2, default=[1e4, 2], help="MPC: Q matrix diagonal. Usage: Q[0,0] -> spacing error, " \
    "Q[1,1] -> relative velocity error")
    parser.add_argument("--Qu", type=float, nargs=1, default=0, help="MPC: Qu value. Penalizes input acceleration magnitude. " \
    "Q[1,1] -> relative velocity error")
    parser.add_argument("--P", type=float, default=0, help="MPC: P weight (meyer term). Terminal error.")
    parser.add_argument("--R", type=float, default=1e-6, help="MPC: R weight (r-term). Penalizes input acceleration differences.")
    parser.add_argument("--a-limit", type=float, nargs=2, default=[-5, 8], help="MPC constraint (ref. acc): [a_min, a_max]")
    parser.add_argument("--L_prec", type=float, default=3.876, help="Model params: Preeceding vehicle length.")
    parser.add_argument("--d_min", type=float, default=2, help="Model params: Distance to preeceding vehicle when stopped (min).")
    parser.add_argument("--h", type=float, default=1, help="Model params: Time gap policy (seconds).")

    
    args = parser.parse_args()

    model_params = {'h': args.h,
                    'd_min': args.d_min,
                    'L_prec': args.L_prec}
    
    mpc_config = {
            'n_horizon': args.n_horizon,
            't_step': args.control_rate * args.sim_dt,
            'n_robust': 0, #not using: for scenario based mpc -> see mpc.set_uncertainty_values()
            'store_full_solution': True,
            'collocation_deg': 2, #default 2nd-degree polynomial 
            #to approximate the state trajectories
            'collocation_ni': 1, #default
            'nlpsol_opts': {'ipopt.linear_solver': 'MA27',
                            'ipopt.print_level':0, 'print_time':0}
            }
    
    opt_params = {
        'Q': [args.Q[0], args.Q[1]],
        'Qu': [args.Qu],
        'P': args.P,
        'R': args.R,
        #*input (acc) constraints added inside main 
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))

    pinn_model_path = os.path.join(script_dir, "../../models/onnx/pinn_FC_noWindow_udds_hwycol_nycccol_70%_alpha0.5_features3.onnx")
    results_path = os.path.join(script_dir, "/results/")

    # Building platoon...
    #same model for every vehicle (homogeneous platoon)
    mpc_model = SecondOrderPINNmodel(pinn_model_path, model_params,
                                     scalerX_path=None,
                                     scalerY_path=None)

    main(
        n_followers=args.n_followers,
        mpc_config=mpc_config,
        opt_params=opt_params,
        mpc_model=mpc_model,
        sim_dt=args.sim_dt,
        t_end=args.t_end,
        host=args.host,
        port=args.port,
        render=args.no_render,
        fn_get_prec_state=fn_get_prec_state,
        acc_cons = args.a_limit
    )
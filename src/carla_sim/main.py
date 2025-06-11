import argparse
import random
import time 
import numpy as np
import os
import carla
from do_mpc.model import Model

from .Core import Simulation, Platoon
from mpc.controller import setupDMPC
from mpc.modelling import SecondOrderPINNmodel

#*This script assumes an already active server with a picked town: 
#* ./config.py --map Town05
#* then in carla root run ./CarlaUE4.sh  

def fn_get_prec_state(platoon, vehicle):
    # Get index of follower in platoon
    idx = vehicle.index
    if idx == 0:
        prec = platoon.lead_vehicle
    else:
        prec = platoon.follower_vehicles[idx - 1]

    #*V2V: Get vel of preceding vehicle 
    v_prec = prec.speed

    return v_prec

def main(n_followers: int, mpc_model: Model, opt_params, mpc_config,
         fn_get_prec_state, sim_dt:float=0.01, t_end:float=np.inf, 
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
    sim = None
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
        lv_bp = vehicle_bp_lib.filter('vehicle.audi.tt')[0] #returns a list so we pick the only element

        #*SPAWN LEAD VEHICLE AND ADD TO ACTOR LIST AND PLATOON
        spawn_points = sim.get_map().get_spawn_points()
        if not spawn_points:
            print("Could not retrieve spawn points from map!")
            return
        
        lv_sp = random.choice(spawn_points)
        platoon = Platoon(sim)
        lv = platoon.add_lead_vehicle(lv_bp, lv_sp)
        #DONT APPEND TO ACTORS LIST, THIS ONLY FOR NPCs
        print(f"Spawned LV: {lv.type_id} (id: {lv.id}) at {lv_sp.location}")
        #TODO: Modify default autopilot behavior
        sim.tick()
        lv.set_autopilot(True, tm_port)

        #* SPAWN FOLLOWERS
        for i in range(0, n_followers):
            followers = platoon.get_follower_list()
            fv = platoon.add_follower_vehicle(lv_bp, (lv.transform_ahead(-10, force_straight=True) if i == 0
                                                else followers[-1].transform_ahead(-10, force_straight=True)))
            
            mpc = setupDMPC(mpc_model, mpc_config, opt_params, fn_get_prec_state, platoon, fv)
            mpc.settings.set_linear_solver(solver_name='MA27') #boosts speed supposedly
            print(f"\n FV{i} controller settings:", mpc.settings)
            mpc.set_initial_guess()
            fv.attach_controller(mpc)

            sim.tick()
            sim.tick()
            
            print(f"Spawned FV: {fv.type_id} (id: {fv.id})")
        
        #*SIMULATING
        print("Running simulation loop...")
        i:int = 0
        if t_end != np.inf: 
            step_end = int(t_end/sim_dt)
        else:
            step_end = np.inf

        control_rate = int(mpc.settings.t_step/sim_dt)
        while i<=step_end:
            if i % control_rate == 0:
                #*RUN PLATOON CONTROL STEP
                #!DEBUG
                a = platoon[1].acceleration
                print(f"Next step a={a}")
                platoon.control_step()

            #*PLACING SPECTATOR TO FRAME SPAWNED VEHICLES
            spect_transf = platoon[-1].transform_ahead(-5, force_straight=True) #platoon[0] is leader
            spect_transf.location.z += 3
            spect_transf.rotation.pitch = -15
            spect.set_transform(spect_transf)

                #print(f"Step {i}: LV at {lv.get_transform()}")
            sim.tick() #advance the simulation by one step (fixed_delta_seconds)            
            i += 1

        sim.release_synchronous()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Simulation finished.")
        print("Cleaning up...")
        world = sim.get_world()
        if sim and world:
            # Restore original settings (important to disable sync mode)
            print("Restoring original world settings.")
            world.apply_settings(sim.get_original_settings())
        if platoon:
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
    parser.add_argument("--n-horizon", type=int, default=10, help="MPC: Prediction horizon length (steps)")
    parser.add_argument("--Q", type=float, nargs=2, default=[100, 10], help="MPC: Q matrix diagonal (lagrange term). Usage: Q[0,0] Q[1,1]")
    parser.add_argument("--P", type=float, default=100, help="MPC: P weight (meyer term)")
    parser.add_argument("--R", type=float, default=1e-3, help="MPC: R weight (r-term)")
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
            'collocation_deg': 2, #default 2nd-degree polynomial to approximate the state trajectories
            'collocation_ni': 1, #default
            }
    
    opt_params = {
        'Q': np.diag([args.Q[0], args.Q[1]]),
        'P': args.P,
        'R': args.R,
        #TODO: Realistic Constraints
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))

    pinn_model_path = os.path.join(script_dir, "../../models/onnx/pinn_model_udds_10%.onnx")
    scalerX_path = os.path.join(script_dir, "../../models/scalers/scalerX_model_udds_10%.save")
    scalerY_path = os.path.join(script_dir, "../../models/scalers/scalerY_model_udds_10%.save")
    results_path = os.path.join(script_dir, "/results/")

    # Building platoon...
    #same model for every vehicle (homogeneous platoon)
    mpc_model = SecondOrderPINNmodel(pinn_model_path, model_params,
                                     scalerX_path=scalerX_path,
                                     scalerY_path=scalerY_path)

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
        fn_get_prec_state=fn_get_prec_state
    )
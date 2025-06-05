import argparse
import random
import time 
import numpy as np
import carla
from Core import Simulation, Platoon
#*assuming an already active server with a picked town: 
        #* ./config.py --map Town05
        #* then in carla root run ./CarlaUE4.sh  

def main(n_followers: int, sim_dt:float=0.01, control_rate=10, t_end:float=np.inf,
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
        lv_bp = vehicle_bp_lib.filter('vehicle.mini.cooper_s_2021')[0] #returns a list so we pick the only element

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
        #TODO: Modify deafault autopilot behavior
        sim.tick()
        lv.set_autopilot(True, tm_port)

        #* SPAWN FOLLOWERS
        for i in range(0, n_followers):
            followers = platoon.get_follower_list()
            fv = platoon.add_follower_vehicle(lv_bp, (lv.transform_ahead(-10, force_straight=True) if i == 0
                                                else followers[-1].transform_ahead(-10, force_straight=True)))
            fv.set_autopilot(True, tm_port) #! REMOVE WHEN FOLLOWER CONTROLLER IS INTEGRATED
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

        while i<=step_end:
            if i % control_rate == 0:
                #TODO: CONTROL PLATOON HERE

                #*PLACING SPECTATOR TO FRAME SPAWNED VEHICLES
                spect_transf = platoon[-1].transform_ahead(-5, force_straight=True) #platoon[0] is leader
                spect_transf.location.z += 3
                spect_transf.rotation.pitch = -15
                spect.set_transform(spect_transf)

                print(f"Step {i}: LV at {lv.get_transform()}")
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("n_followers", type=int, help="Number of followers to add behind the platoon leader")
    parser.add_argument("--sim-dt", type=float, default=0.01, help="Simulation time step")
    parser.add_argument("--control-rate", type=int, default=10, help="Control rate (steps)")
    parser.add_argument("--t-end", type=float, default=np.inf, help="Simulation end time")
    parser.add_argument("--host", type=str, default='localhost', help="Carla server host")
    parser.add_argument("--port", type=int, default=2000, help="Carla server port")
    parser.add_argument("--no-render", action="store_false", help="Enable rendering")
    args = parser.parse_args()
    main(
        n_followers=args.n_followers,
        sim_dt=args.sim_dt,
        control_rate=args.control_rate,
        t_end=args.t_end,
        host=args.host,
        port=args.port,
        render=args.no_render
    )
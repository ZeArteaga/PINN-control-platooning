import carla
import random
import time 
import numpy as np

#*assuming an already active server with a picked town: 
        #* ./config.py --map Town05
        #* then in carla root run ./CarlaUE4.sh  

def main(host='localhost', port=2000, render:bool=True, dt:float=0.1, t_end:float=np.inf):
    """
    Args:
			host: Carla server host
			port: Carla server port
			render: Turns rendering on (True) or off (False)
			dt: length of a simulation time step
			t_end: Defaults to np.inf to run indefinetly
		"""
    
    actor_list = []
    client = None

    try:
        #*GET CLIENT AND WORLD
        client = carla.Client(host, port)
        client.set_timeout(10.0)

        world = client.get_world()
        print(f"Successfully connected to Carla. Current map: {world.get_map().name}")
        
        #*APPLY SETTINGS
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True #enable sync mode for control purposes
        settings.fixed_delta_seconds = dt # matching MPC dts
        print("Synchronous mode enabled with fixed_delta_seconds =", settings.fixed_delta_seconds)
        settings.no_rendering_mode = not render
        if render:
            print("Rendering is ON!")
        else: print("Rendering is OFF!")
        world.apply_settings(settings)

        #*PICK VEHICLE
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.mini.cooper_s_2021')[0] #returns a list so we pick the only element

        #*SPAWN VEHICLE AND ADD TO ACTOR LIST
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("Could not retrieve spawn points from map!")
            return
        
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print(f"Spawned actor: {vehicle.type_id} (id: {vehicle.id}) at {spawn_point.location}")
        
        vehicle.set_autopilot(True)

        #*PLACING SPECTATOR TO FRAME SPAWNED VEHICLES
        spectator = world.get_spectator()
        vehicle_ini_tf = vehicle.get_transform()
        spectator_loc = vehicle_ini_tf.location + carla.Location(x=-10, z=5)
        spectator_rot = carla.Rotation(pitch=vehicle_ini_tf.rotation.pitch-20.0, 
                                           yaw=vehicle_ini_tf.rotation.yaw, 
                                           roll=vehicle_ini_tf.rotation.roll) #looking down at the car
        spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))

        #*SIMULATING
        print("Running simulation loop...")
        i:int = 0
        if t_end != np.inf: 
            step_end = int(t_end/dt)
        else:
            step_end = np.inf
        while i<=step_end:
            world.tick() #advance the simulation by one step (fixed_delta_seconds)
            i += 1
            print(f"Step {i}: Vehicle at {vehicle.get_transform()}")

    except KeyboardInterrupt:
                print("\nSimulation interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        print("Simulation finished.")
        print("Cleaning up...")
        if client and world:
            # Restore original settings (important to disable sync mode)
            if 'original_settings' in locals(): # Check if original_settings was defined
                 print("Restoring original world settings.")
                 world.apply_settings(original_settings)
            elif settings.synchronous_mode: # only if we actually changed it
                    print("Disabling synchronous mode (fallback).")
                    current_settings = world.get_settings()
                    current_settings.synchronous_mode = False
                    current_settings.fixed_delta_seconds = None
                    world.apply_settings(current_settings)
            if 'spectator' in locals():
                print("Restoring original spectator position.")
                spectator.set_transform(carla.Transform())
        if actor_list:
            print(f"Destroying {len(actor_list)} actors.")
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            # A small delay to ensure actors are destroyed before script exits, can sometimes help
            time.sleep(0.5) 
        print("Cleanup finished.")

if __name__ == '__main__':
    main()
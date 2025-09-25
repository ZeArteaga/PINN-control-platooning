import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("eval", lambda expr: eval(expr)) #allows eval in config file
import time 
import numpy as np
import os
import carla
from .agents.navigation.controller import PIDLongitudinalController, PIDLateralController

from .Core import Simulation, fn_get_prec_state
from .vehicle import Vehicle
from .platoon import Platoon
from .leader_agent import create_leader_agent
from mpc.controller import setupDMPC
from mpc.modelling import SecondOrderPINNmodel
from .saving import save_follower_data

#*This script assumes an already active server: 
#* then in carla root run ./CarlaUE4.sh (optionally: ./CarlaUE4.sh -quality-level=Low)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    sim_cfg = cfg.scenario.SIM
    mpc_cfg = cfg.controller.MPC
    pid_long_cfg = cfg.controller.PID_long
    pid_lat_cfg = cfg.controller.PID_lat
    lv_cfg = cfg.scenario.LEAD_AGENT
    sim_dt = sim_cfg.sim_dt
    t_end = sim_cfg.t_end
    control_dt = sim_cfg.control_dt
    control_rate = cfg.control_rate
    
    SEED = 50
    actor_list = []

    try:
        #*GET CLIENT, WORLD, TRAFFIC MANAGER
        sim = Simulation(cfg.host, cfg.port, world=sim_cfg.map,
                          large_map=False, dt=sim_cfg.sim_dt, synchronous=True, render=sim_cfg.render)
        print(f"Successfully connected to Carla. Current map: {sim.get_map().name}")

        world = sim.get_world()
        map = sim.get_map()

        tm = sim.get_trafficmanager(port=8000)
        tm_port = tm.get_port()
        tm.set_random_device_seed(SEED) #for simulation determinism
        print("Got Traffic Manager.")

        #*PICK VEHICLE
        vehicle_bp_lib = sim.get_vehicle_blueprints()
        #imu_bp = sim.get_sensor_blueprints().find('sensor.other.imu')
        lv_bp = vehicle_bp_lib.find(sim_cfg.vehicle_bp) 
        
        #*SPAWN LEAD VEHICLE
        spawn_points = sim.get_map().get_spawn_points()
        if not spawn_points:
            print("Could not retrieve spawn points from map!")
            return

        lv_sp = spawn_points[lv_cfg.spawn_point]
        if lv_cfg.path:
            locs = [carla.Location(*coords) for coords in lv_cfg.path]
        else:
            locs = None
        if lv_cfg.destination:
            destination = carla.Location(*lv_cfg.destination)
        else:
            destination = None

        platoon = Platoon(sim, cfg.controller.Path_settings)
        lv: Vehicle = platoon.add_lead_vehicle(blueprint=lv_bp, spawn_point=lv_sp)
        sim.tick() #!without this line, agent doesnt work
        lv, lv_agent = create_leader_agent(lv, locations=locs, agent_type=lv_cfg.type, 
                                           destination=destination,
                                           target_speed=lv_cfg.target_speed,
                                           hazard_settings=lv_cfg.hazards)
        if lv is None:
            raise RuntimeError("Failed to spawn lead vehicle")
        print("LV spawned and Agent created.")
        sim.tick()

        #* SPAWN FOLLOWERS
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, mpc_cfg.pinn_path)
        pinn_model = SecondOrderPINNmodel(model_path, mpc_cfg.policy, scalerX_path=mpc_cfg.scalerX_path,
                                          scalerY_path=mpc_cfg.scalerY_path)
        for i in range(0, sim_cfg.n_followers):
            followers = platoon.get_follower_list()
            fv: Vehicle = platoon.add_follower_vehicle(lv_bp, (lv.transform_ahead(-sim_cfg.ini_gap, force_straight=True) if i == 0
                                                else followers[-1].transform_ahead(-sim_cfg.ini_gap, force_straight=True)))
            """ *add follower sensors
            imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=fv._carla_vehicle)
            fv.attach_sensor('imu', imu)
            actor_list.append(imu)  """
            
            #*Setup controllers
            fv_mass = fv.get_physics_control().mass
            mpc_opt_params: dict = OmegaConf.to_container(mpc_cfg.opt_params, resolve=True)
            mpc_settings: dict = OmegaConf.to_container(mpc_cfg.settings, resolve=True)
            mpc_opt_params["u_min"] = mpc_cfg.opt_params.cons_acc[0]*fv_mass
            mpc_opt_params["u_max"] = mpc_cfg.opt_params.cons_acc[1]*fv_mass
            mpc = setupDMPC(pinn_model, mpc_settings, mpc_opt_params,
                             fn_get_prec_state, platoon, fv)
            
            print(f"\n FV{i} controller settings:", mpc.settings)
            mpc.set_initial_guess()
            pid_long = PIDLongitudinalController(fv, dt=sim_dt,
                                             K_P=pid_long_cfg.Kp, K_I=pid_long_cfg.Ki, K_D=pid_long_cfg.Kd)
            pid_lat = PIDLateralController(fv, dt=sim_dt,
                                             K_P=pid_lat_cfg.Kp, K_I=pid_lat_cfg.Ki, K_D=pid_lat_cfg.Kd)
            fv.attach_controller(mpc, pid_long, pid_lat)
            
            print(f"Spawned FV: {fv.type_id} (id: {fv.id})")
            sim.tick()
            sim.tick()
        #*SIMULATING
        print("Running simulation loop...")
        for _ in range(0, int(5/sim_dt)):
                sim.tick() #tick 5 seconds until the spawned vehicles stabilize
        
        i:int = 0
        if t_end is None:
            step_end = np.inf
        else:
            step_end = int(t_end/sim_dt)

        while i<=step_end:
            print(f"[t={sim_dt*i}]\n")

            platoon.update_kinematics_all(sim_dt) #update sensor measurements

            if not lv_agent.done():
                lv.apply_control(lv_agent.run_step()) #leader also controlled every tick
            else:
                print("LV has reached destination!")
                break
            
            if i % control_rate == 0: 
                platoon.compute_high_control() #MPC runs every control rate

            platoon.apply_low_control(sim_dt, debug=True) #long ppid (from mpc ref) and lat pid run with higher freq
            platoon.log_data_all()

            sim.update_spectator(platoon)
            sim.tick()
            i += 1

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Simulation finished.")
        
        #*Save data
        if 'platoon' in locals() and len(platoon) > 0:
            save_follower_data(sim_dt, control_dt, platoon)

        print("Cleaning up...")
        if 'sim' in locals():
            sim.release_synchronous()
            world = sim.get_world()
            if world:
                print("Restoring original world settings.")
                world.apply_settings(sim.get_original_settings())
        if 'platoon' in locals() and len(platoon) > 0:
            sim.apply_batch([carla.command.DestroyActor(v.id) for v in platoon])
            print(f"Destroying platoon with {len(platoon)} vehicles.")
        if actor_list:
            print(f"Destroying {len(actor_list)} actors.")
            sim.apply_batch([carla.command.DestroyActor(x.id) for x in actor_list])
            time.sleep(0.5) 
        print("Cleanup finished.")
        return 0

if __name__ == "__main__":
    main()
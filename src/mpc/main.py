import os
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from do_mpc.simulator import Simulator
from do_mpc.controller import MPC
from modelling import SecondOrderPINNmodel, SecondOrderIdeal, ThirdOrderModel, DiscreteWindowedPINN
from controller import setupDMPC, setupSim
from plotting import setup_graphics, plot
from do_mpc.data import save_results
class FV:
    def __init__(self, mpc: MPC, sim: Simulator, mass: float, window_size: int):
        '''Window size including current state. e.g: history of 2 steps -> window_size = 3'''
        self.mpc = mpc
        self.sim = sim
        self.mass = mass
        self.window_size = window_size
        self.sliding_window = deque(maxlen=window_size-1)

    def set_initial_conditions(self, x0: dict):
        """Set initial conditions for both MPC and simulator"""
        # The control input u is a scalar. Initialize it.
        self.u = x0['u0'] 
        
        # Initialize sliding window with repeated initial values
        # The window contains [t, v, u] for past steps. u is a scalar.
        initial_entry = [x0['t0'], x0['v0'], self.u] 
        for _ in range(self.window_size - 1):
            self.sliding_window.append(initial_entry)
        
        # mpc_state now holds the core states [x, v, Ie] that will be updated each step
        self.mpc_state = np.array([x0['x0'], x0['v0'], x0['Ie0']])
        
        # Set initial conditions for do_mpc objects
        self.sim.x0 = np.array([x0["x0"], x0["v0"], x0['Ie0'], x0["a0"]])
        self.mpc.x0 = self.extend_mpc_state(self.mpc_state)
        
        self.mpc.set_initial_guess()
        
    def extend_mpc_state(self, mpc_state) -> np.array:
        """
        Combines the current core state [x, v, Ie] with the historical window
        to form the full state vector required by the do_mpc model.
        Order: [x, v, Ie, t_km1, v_km1, u_km1, t_km2, v_km2, u_km2, ...]
        """
        extended = list(mpc_state)
        for entry in self.sliding_window:
            extended.extend(entry)
            
        return np.array(extended)

    def _update_mpc_window(self, t: float):
        """Updates the sliding window with the latest measurement."""
        v = self.mpc_state[1]
        
        new_entry = [t, v, float(self.u)]
        self.sliding_window.appendleft(new_entry)

    def mpc_step(self):
        """Runs one step of the MPC controller."""
        full_state = self.extend_mpc_state(self.mpc_state)
        self.u = self.mpc.make_step(full_state)

    def sim_step(self, t: float):
        self.sim_state = self.sim.make_step(self.u).reshape((-1,))
        self.mpc_state = self.sim_state[:-1] #ignore acceleration
        self._update_mpc_window(t)

    def set_graphics(self, mpc_graphics, sim_graphics):
        self.mpc_graphics = mpc_graphics
        self.sim_graphics = sim_graphics

    def print_latest_mpc(self):
        mpc_x = self.mpc.data['_x', 'x'][-1]
        mpc_x_prec = self.mpc.data['_tvp', 'x_prec'][-1]
        mpc_v = self.mpc.data['_x', 'v'][-1]
        mpc_v_prec = self.mpc.data['_tvp', 'v_prec'][-1]
        mpc_gap = self.mpc.data['_aux', 'd'][-1]
        mpc_desired_gap = self.mpc.data['_aux', 'd_ref'][-1]
        mpc_u = self.mpc.data['_u', 'u'][-1]

        print(f"  Ego position (x): {mpc_x}")
        print(f"  Ego velocity (v): {mpc_v}")
        print(f"  Preceeding position (x_prec): {mpc_x_prec}")
        print(f"  Preceeding velocity (v_prec): {mpc_v_prec}")
        print(f"  Desired gap (d_ref): {mpc_desired_gap}")
        print(f"  Actual gap (d): {mpc_gap}")
        print(f"  Computed input acceleration (u/m): {mpc_u/self.mass}")

    def print_latest_sim(self):
        sim_x = self.sim.data['_x', 'x'][-1]
        sim_x_prec = self.sim.data['_tvp', 'x_prec'][-1]
        sim_v = self.sim.data['_x', 'v'][-1]
        sim_v_prec = self.sim.data['_tvp', 'v_prec'][-1]
        sim_gap = self.sim.data['_aux', 'd'][-1]
        
        print(f"  Ego position (x): {sim_x}")
        print(f"  Ego velocity (v): {sim_v}")
        print(f"  Preceeding position (x_prec): {sim_x_prec}")
        print(f"  Preceeding position (v_prec): {sim_v_prec}")
        print(f"  Actual gap (d): {sim_gap}\n")
        
if __name__ == "__main__":
    #*CONFIG---
    VEHICLE_MASS = 1500
    dt = 0.1
    L_prec = 4.5
    t_samp = np.array([0, 5, 10, 15, 20]) #time check points
    t_end = t_samp[-1]
    noise_std = 0 #TODO
    t = np.arange(start=t_samp[0], stop=t_end+dt, step=dt) #end at t_end seconds
    #TODO: add driving cycle for leader
    lv_samp = np.array([50, 50, 50, 50, 50]) / 3.6 #leader speed check points 
    lv_speed = interp1d(t_samp, lv_samp, kind='quadratic') #quadratic interpolation -> no drivetrain limitation for now
    lv_x0 = 30 #start at arbitrary position
    lv_curr_state = { #initial state - Leader Vehicle
        'x': lv_x0, 
        'v': lv_speed(0)
    }

    def fn_get_prec_state(platoon: list, vehicle_idx: int):
        if vehicle_idx == 0:
            # For the first follower, the preceding vehicle is the leader.
            x_prec = lv_curr_state['x']
            v_prec = lv_curr_state['v']
        else:
            # For other followers, the preceding vehicle is in the platoon list.
            prec = platoon[vehicle_idx - 1]
            x_prec = prec.state[0]
            v_prec = prec.state[1]

        return x_prec, v_prec

    #Followers
    ini_gap = L_prec + 20 
    fv0_ini = {
        "x0": lv_x0 - ini_gap,
        "v0": 40/3.6,
        'Ie0': 0, #integral action
        'a0': 0,
        'u0': 0,
        't0': 0
    }
    fv_initials = [fv0_ini] #x, v, 0 integral action, 0 acceleration
    platoon_size = len(fv_initials) #excluding leader
    #will use the following to instantiate various decentralized MPC for each CAV
    model_params = {'h': 1,
                    'd_min': 2,
                    'L_prec': 4.5,
                    'm': VEHICLE_MASS,
                    'tau': 0.5,
                    'window_size': 3,
                    'dt': dt
                    }

    mpc_config = {
            'n_horizon': 10,
            't_step': dt, #set equal to dt for model training
            'n_robust': 0, #for scenario based mpc -> see mpc.set_uncertainty_values()
            'store_full_solution': True,
            'collocation_deg': 2, #default 2nd-degree polynomial to approximate the state trajectories
            'collocation_ni': 1, #default
            'nlpsol_opts': {'ipopt.linear_solver': 'MA27',
                            'ipopt.print_level':0, 'print_time':0}
            
            }
    
    opt_params = {
        'Q': np.diag([1000, 10, 100]), #PDI [100, 10, 100]
        'P': np.diag([0, 0, 0]), #TODO: INvestigate terminal cost
        'R': 0.1,
        'u_max': 5*VEHICLE_MASS,
        'u_min': -8*VEHICLE_MASS,
        #TODO: Realistic Constraints
    }

    sim_config = {
        't_step': dt,
        'reltol': 1e-10, #default
        'abstol': 1e-10 #default
    }
    #*---

    script_dir = os.path.dirname(os.path.abspath(__file__))

    pinn_model_path = os.path.join(script_dir, "../../models/onnx/pinn_FC_udds_hwycol_70%_windowsize3.onnx")
    scalerX_path = os.path.join(script_dir, "../../models/scalers/scalerX_FC_udds_hwycol_70%_windowsize3.save")
    scalerY_path = os.path.join(script_dir, "../../models/scalers/scalerY_FC_udds_hwycol_70%_windowsize3.save")
    # Building platoon...
    #same model for every vehicle (homogeneous platoon)
    mpc_model = DiscreteWindowedPINN(pinn_model_path, model_params,
                                     scalerX_path=scalerX_path,
                                     scalerY_path=scalerY_path)
    plant_model = ThirdOrderModel(model_params)
    print("Pinn (MPC) model control input and states:", mpc_model.u.keys(), mpc_model.x.keys())
    print("Pinn (MPC) model time varying parameters (provided):", mpc_model.tvp.keys())
    print("Plant model control input and states:", plant_model.u.keys(), plant_model.x.keys())
    print("Plant model time varying parameters (provided):", plant_model.tvp.keys())
    
    platoon = []
    for i in range(0, platoon_size):
        print(f"Setting MPC for follower {i}...")

        mpc = setupDMPC(mpc_model, mpc_config, opt_params, fn_get_prec_state, platoon, i)
        print("\n", mpc.settings)
        sim = setupSim(plant_model, sim_config, fn_get_prec_state, platoon, i)
        fv = FV(mpc, sim, VEHICLE_MASS, model_params['window_size'])
        #MPC init conditions
        fv_ini = fv_initials[i]
        fv.set_initial_conditions(fv0_ini)
        print(f"\n(FV{i}) MPC initial conditions: {fv.mpc.x0}")
        print(f"\n(FV{i}) SIM (actual) initial conditions: {fv.sim.x0}")
        platoon.append(fv) #add to platton        
        
        #for plotting
        mpc_graphics, sim_graphics = setup_graphics(fv.mpc.data, fv.sim.data)
        fv.set_graphics(mpc_graphics, sim_graphics)

    #Control loop:
    fv0 = platoon[0]
    for i, t_value in enumerate(t):
        fv0.mpc_step()
        #! DEBUG
        print(f"DEBUG: at t={t_value:.1f}")
        fv0.print_latest_mpc()
        #!---

        #* Have to do update of tvp before calling sim,
        #* for some reason simulator does all calculation before updating tvp, except for t=0 obv.
        #* The controller does the opposite.
        #* That's why there was a problem with the calculated gap vs the last actual gap calculated by the sim,
        #* because this last one was stored using the previous leader position. So the update has 
        #* to be in the middle to sync the two,
        #* and eliminate the discrepancy that was causing a steady-state error. 

        lv_curr_state['v'] = lv_speed(t_value)
        lv_curr_state['x'] += lv_curr_state['v']*dt

        fv0.sim_step(t_value)
        #!DEBUG
        #fv0.print_latest_sim()
        #!---

    save_results([fv0.mpc, fv0.sim], overwrite=True)

    fv0.mpc_graphics.plot_predictions(t_ind=0) #Additionally exemplify predictions
    plot(fv0.sim_graphics, name="(sim)")

    plt.show()
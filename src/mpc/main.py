import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from do_mpc.simulator import Simulator
from do_mpc.controller import MPC
from modelling import SecondOrderPINNmodel, SecondOrderIdealPlant, ThirdOrderPlant
from controller import setupDMPC, setupSim
from plotting import setup_graphics, plot
from do_mpc.data import save_results
class FV:
    def __init__(self, sim_initial: np.ndarray, mpc_initial: np.ndarray, mpc: MPC, sim: Simulator):
        self.state = sim_initial
        self.mpc = mpc
        self.sim = sim
        self.mpc.x0 = mpc_initial
        self.sim.x0 = sim_initial
        self.mpc.set_initial_guess()
        print(self.state)
    def set_graphics(self, mpc_graphics, sim_graphics):
        self.mpc_graphics = mpc_graphics
        self.sim_graphics = sim_graphics
    
    def sim_step(self, u):
        #reset with current plant state and fetch Ie from the previous mpc state
        self.state = self.sim.make_step(u)

    def mpc_step(self):
        u = self.mpc.make_step(self.state[:-1]) #ignore last state -> acceleration 
        return u

if __name__ == "__main__":
    #*CONFIG---
    VEHICLE_MASS = 1284.0
    dt = 0.1
    L_prec = 4.5
    t_samp = np.array([0, 5, 10, 15, 20]) #time check points
    t_end = t_samp[-1]
    noise_std = 0 #TODO
    t = np.arange(start=t_samp[0], stop=t_end+dt, step=dt) #end at t_end seconds
    
    #TODO: add driving cycle for leader
    lv_samp = np.array([0, 1, -3, 0.1, -0.3]) #leader acc checkpoint 
    lv_acc = interp1d(t_samp, lv_samp, kind='quadratic') #quadratic interpolation -> no drivetrain limitation for now
    lv_x0 = 30 #start at arbitrary position
    lv_v0 = 50 / 3.6
    lv_curr_state = { #initial state - Leader Vehicle
        'x': lv_x0, 
        'v': lv_v0,
        'a': lv_acc(0)
    }

    def fn_get_prec_state(platoon: list, vehicle_idx: int):
        if vehicle_idx == 0:
            # For the first follower, the preceding vehicle is the leader.
            x_prec = lv_curr_state['x']
            v_prec = lv_curr_state['v']
            a_prec = lv_curr_state['a']
        else:
            # For other followers, the preceding vehicle is in the platoon list.
            prec = platoon[vehicle_idx - 1]
            x_prec = prec.state[0]
            v_prec = prec.state[1]
            a_prec = prec.state[2]

        return x_prec, v_prec, a_prec

    #Followers
    ini_gap = L_prec + 20 
    fv_v0 = 40/3.6
    #*Plant state: [x, v, u, a]
    sim_initial_state = np.array([lv_x0 - ini_gap, fv_v0, 0, 0])
    #*MPC state: [x, v, u]
    mpc_initial_state = np.array([lv_x0 - ini_gap, fv_v0, 0])

    fv_initials = [(sim_initial_state, mpc_initial_state)] # Store as a tuple
    platoon_size = len(fv_initials) #excluding leader
    #will use the following to instantiate various decentralized MPC for each CAV
    model_params = {'h': 1,
                    'd_min': 2,
                    'L_prec': 4.5,
                    'm': VEHICLE_MASS,
                    'tau': 0.3,
                    }

    mpc_config = {
            'n_horizon': 15,
            't_step': dt, #set equal to dt for model training
            'n_robust': 0, #for scenario based mpc -> see mpc.set_uncertainty_values()
            'store_full_solution': True,
            'collocation_deg': 2, #default 2nd-degree polynomial to approximate the state trajectories
            'collocation_ni': 1, #default
            'nlpsol_opts': {'ipopt.linear_solver': 'MA27',
                           'ipopt.print_level':0, 'print_time':0}
            }
    
    opt_params = {
        'Q': [3e4, 1.5e3], #spacing error, de/dt -> relative velocity  
        'Qu': [2e-3], #relative to u magnitude (input acceleration)
        'P': np.diag([0, 0, 0]), #TODO: Investigate terminal cost
        'R': 5e-5, #-> relative to du/dt (control variable)
        'u_max': 4.208*VEHICLE_MASS,
        'u_min': -8*VEHICLE_MASS,
        'v_max': 140/3.6
        #TODO: Realistic Constraints
    }

    sim_config = {
        't_step': dt,
        'reltol': 1e-6,
        'abstol': 1e-6, 
    }
    #*---

    script_dir = os.path.dirname(os.path.abspath(__file__))

    pinn_model_path = os.path.join(script_dir, "../../models/onnx/" \
    "pinn_FC_noWindow_udds_hwycol_fullsplit_alpha0.25_features3.onnx")
    """ scalerX_path = os.path.join(script_dir, "../../models/scalers/scalerX_" \
    "FC_noWindow_udds_hwycol_nycccol_70%.save")
    scalerY_path = os.path.join(script_dir, "../../models/scalers/scalerY_" \
    "FC_noWindow_udds_hwycol_nycccol_70%.save") """
    
    # Building platoon...
    #same model for every vehicle (homogeneous platoon)
    mpc_model = SecondOrderPINNmodel(pinn_model_path, model_params,
                                     scalerX_path=None,
                                     scalerY_path=None)
    plant_model = ThirdOrderPlant(model_params)
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
        sim_init, mpc_init = fv_initials[i]
        platoon.append(FV(sim_init, mpc_init, mpc, sim)) #create follower and add to platoon

    for i, fv in enumerate(platoon):
        print(f"\n(FV{i}) Set Initial Conditions:")
        print(str(fv.mpc.x0['x']) + "m", str(fv.mpc.x0['v']*3.6) + "km/h") #, str(fv.sim.x0['a']) + "m/s^2")
        mpc_graphics, sim_graphics = setup_graphics(fv.mpc.data, fv.sim.data)
        fv.set_graphics(mpc_graphics, sim_graphics)
        
        
    #Control loop:
    fv0 = platoon[0]
    for i, t_value in enumerate(t):
        du = fv0.mpc_step()

        #!DEBUG
        print(f"DEBUG: MPC state at t={t_value:.1f}")
        mpc_x = fv0.mpc.data['_x', 'x'][-1]
        mpc_x_prec = fv0.mpc.data['_tvp', 'x_prec'][-1]
        mpc_v = fv0.mpc.data['_x', 'v'][-1]
        mpc_v_prec = fv0.mpc.data['_tvp', 'v_prec'][-1]
        mpc_gap = fv0.mpc.data['_aux', 'd'][-1]
        mpc_desired_gap = fv0.mpc.data['_aux', 'd_ref'][-1]
        u = fv0.mpc.data['_x', 'u'][-1]

        print(f"  Ego position (x): {mpc_x}")
        print(f"  Ego velocity (v): {mpc_v}")
        print(f"  Preceeding position (x_prec): {mpc_x_prec}")
        print(f"  Preceeding velocity (v_prec): {mpc_v_prec}")
        print(f"  Desired gap (d_ref): {mpc_desired_gap}")
        print(f"  Actual gap (d): {mpc_gap}")
        print(f"  Gap error (d - d_ref): {mpc_gap - mpc_desired_gap}")
        print(f"  Chosen acceleration (u): {u/VEHICLE_MASS}")
        print(f" Error matrix (e, de, u): {fv0.mpc.data['_aux', 'E'][-1]}")

        #* Have to do update of tvp before calling sim,
        #* for some reason simulator does all calculation before updating tvp, except for t=0 obv.
        #* The controller does the opposite.
        #* That's why there was a problem with the calculated gap vs the last actual gap calculated by the sim,
        #* because this last one was stored using the previous leader position. So the update has to be in the middle to sync the two,
        #* and eliminate the discrepancy that was causing a steady-state error. 

        lv_curr_state['a'] = lv_acc(t_value) #a[k]
        lv_vk = lv_curr_state['v'] #v[k]
        lv_curr_state['x'] += (lv_vk)*dt + 0.5*lv_curr_state['a']*dt**2 #v[k+1]
        lv_curr_state['v'] += lv_curr_state['a']*dt #v[k+1]

        fv0.sim_step(du)        

        #!DEBUG
        """ print(f"DEBUG: Simulator state at t={t_value:.1f}")
        sim_x = fv0.sim.data['_x', 'x'][-1]
        sim_x_prec = fv0.sim.data['_tvp', 'x_prec'][-1]
        sim_v = fv0.sim.data['_x', 'v'][-1]
        sim_v_prec = fv0.sim.data['_tvp', 'v_prec'][-1]
        sim_gap = fv0.sim.data['_aux', 'd'][-1]
        
        print(f"  Ego position (x): {sim_x}")
        print(f"  Ego velocity (v): {sim_v}")
        print(f"  Preceeding position (x_prec): {sim_x_prec}")
        print(f"  Preceeding position (v_prec): {sim_v_prec}")
        print(f"  Calculated gap: {sim_x_prec - sim_x - L_prec}", f"using {fv0.sim.model.aux['d']}")
        print(f"  Actual gap (d): {sim_gap}\n") """
    
    save_results([fv0.mpc], overwrite=True)
    plot(mpc_graphics, pred_t=10)

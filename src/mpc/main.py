import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from do_mpc.simulator import Simulator
from do_mpc.controller import MPC
from modelling import SecondOrderPINNmodel, SecondOrderIdeal
from controller import setupDMPC, setupSim
from plotting import setup_graphics, plot
from do_mpc.data import save_results
class FV:
     def __init__(self, fv_initial: np.ndarray, mpc: MPC, sim: Simulator):
        self.state = fv_initial
        self.mpc = mpc
        self.sim = sim
        self.mpc.x0 = fv_initial
        self.sim.x0 = fv_initial
        self.mpc.set_initial_guess()

     def set_graphics(self, mpc_graphics, sim_graphics):
        self.mpc_graphics = mpc_graphics
        self.sim_graphics = sim_graphics

if __name__ == "__main__":
    #*CONFIG---
    VEHICLE_MASS_KG = 1500
    h = 1
    d_min = 2
    L_prec = 4.5 #same len for every CAV
    dt = 0.1
    t_samp = np.array([0, 5, 10, 15, 20]) #time check points
    t_end = t_samp[-1]
    noise_std = 0 #TODO
    t = np.arange(start=t_samp[0], stop=t_end+dt, step=dt) #end at t_end seconds
    #TODO: add driving cycle for leader
    lv_samp = np.array([50, 50, 50, 50, 50]) / 3.6 #leader speed check points 
    lv_speed = interp1d(t_samp, lv_samp, kind='quadratic') #quadratic interpolation -> no drivetrain limitation for now
    lv_x0 = 200
    lv_curr_state = { #initial state - Leader Vehicle
        'x': lv_x0,
        'v': lv_speed(0)
    }

    def get_lv_state(t_now):
        x,v = lv_curr_state['x'], lv_curr_state['v'] 
        #print(x,v)
        return x,v

    #Followers
    ini_gap = L_prec + 20 
    fv_v0 = 40/3.6
    fv_initials = [np.array([lv_x0 - ini_gap, fv_v0])]
    platoon_size = len(fv_initials) #excluding leader
    #will use the following to instantiate various decentralized MPC for each CAV
    model_params = {'h': h,
                    'd_min': d_min,
                    'L_prec': L_prec,
                    'm': VEHICLE_MASS_KG}

    mpc_config = {
            'n_horizon': 10,
            't_step': dt, #set equal to dt for model training
            'n_robust': 0, #for scenario based mpc -> see mpc.set_uncertainty_values()
            'store_full_solution': True,
            'collocation_deg': 2, #default 2nd-degree polynomial to approximate the state trajectories
            'collocation_ni': 1, #default
            }
    
    opt_params = {
        'Q': np.diag([100, 10]),
        'P': 100,
        'R': 0.0001,
        #TODO: Realistic Constraints
    }

    sim_config = {
        't_step': dt,
        'reltol': 1e-10, #default
        'abstol': 1e-10 #default
    }
    #*---

    script_dir = os.path.dirname(os.path.abspath(__file__))

    pinn_model_path = os.path.join(script_dir, "../models/onnx/pinn_model_udds_10%.onnx")
    scalerX_path = os.path.join(script_dir, "../models/scalers/scalerX_model_udds_10%.save")
    scalerY_path = os.path.join(script_dir, "../models/scalers/scalerY_model_udds_10%.save")
    results_path = os.path.join(script_dir, ".results/")

    # Building platoon...
    #same model for every vehicle (homogeneous platoon)
    mpc_model = SecondOrderPINNmodel(pinn_model_path, model_params,
                                     scalerX_path=scalerX_path,
                                     scalerY_path=scalerY_path)
    plant_model = SecondOrderIdeal(model_params)
    print("Pinn model control input and states:", mpc_model.u.keys(), mpc_model.x.keys())
    print("Pinn model time varying parameters (provided):", mpc_model.tvp.keys())

    platoon = []
    for i in range(0, platoon_size):
        print(f"Setting MPC for follower {i}...")
        if i==0:
            fn_get_prec = get_lv_state #set leader function
        else:
            #! not implemented
            raise NotImplementedError("Setup for followers beyond the leader is not implemented yet.")
        
        mpc = setupDMPC(mpc_model, mpc_config, opt_params, fn_get_prec)
        mpc.settings.set_linear_solver(solver_name='MA27') #boosts speed supposedly
        print("\n", mpc.settings)

        sim = setupSim(plant_model, sim_config, get_prec_state=fn_get_prec)
        platoon.append(FV(fv_initials[i], mpc, sim)) #create follower and add to platoon

    #test:
    for i, fv in enumerate(platoon):
        print(f"\n(FV{i}) Set Initial Conditions:")
        print(str(fv.mpc.x0['x']) + "m", str(fv.mpc.x0['v']*3.6) + "km/h")
        mpc_graphics, sim_graphics = setup_graphics(fv.mpc.data, fv.sim.data)
        fv.set_graphics(mpc_graphics, sim_graphics)
        
        
    #Control loop:
    fv0 = platoon[0]
    fv0.mpc.settings.supress_ipopt_output()
    for i, t_value in enumerate(t):
        u = fv0.mpc.make_step(fv0.state)
        #!DEBUG
        """ print(f"DEBUG: MPC state at t={t_value:.1f}")
        mpc_x = fv0.mpc.data['_x', 'x'][-1]
        mpc_x_prec = fv0.mpc.data['_tvp', 'x_prec'][-1]
        mpc_v = fv0.mpc.data['_x', 'v'][-1]
        mpc_v_prec = fv0.mpc.data['_tvp', 'v_prec'][-1]
        mpc_gap = fv0.mpc.data['_aux', 'd'][-1]
        
        print(f"  Ego position (x): {mpc_x}")
        print(f"  Ego velocity (v): {mpc_v}")
        print(f"  Preceeding position (x_prec): {mpc_x_prec}")
        print(f"  Preceeding position (v_prec): {mpc_v_prec}")
        print(f"  Calculated gap: {mpc_x_prec - mpc_x - L_prec}", f"using {fv0.mpc.model.aux['d']}")
        print(f"  Actual gap (d): {mpc_gap}") """

        #* Have to do update of tvp before calling sim,
        #* for some reason simulator does all calculation before updating tvp, except for t=0 obv.
        #* The controller does the opposite.
        #* That's why there was a problem with the calculated gap vs the last actual gap calculated by the sim,
        #* because this last one was stored using the previous leader position. So the update has to be in the middle to sync the two,
        #* and eliminate the discrepancy that was causing a steady-state error. 

        lv_curr_state['v'] = lv_speed(t_value) 
        lv_curr_state['x'] += lv_curr_state['v'] * dt
        fv0.state = fv0.sim.make_step(u)
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
    
    save_results([fv0.mpc, fv0.sim], overwrite=True, result_path=results_path)

    fv0.mpc_graphics.plot_predictions(t_ind=0) #Additionally exemplify predictions
    plot(fv0.sim_graphics, name="(sim)")

    plt.show()
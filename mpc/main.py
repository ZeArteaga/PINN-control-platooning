import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from do_mpc.simulator import Simulator
from do_mpc.controller import MPC
from modelling import SecondOrderCarModel, SecondOrderIdeal
from controller import setupDMPC, setupSim
from plotting import setup_graphics

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
    d_min = 0
    L_prec = 4.5 #same len for every CAV
    dt = 0.1
    t_end = 20
    noise_std = 0
    t_samp = np.array([0, 5, 10, 13, 18, 20]) #time check points
    t = np.linspace(t_samp[0], t_samp[-1], int(t_end/dt)+1) #end at t_end seconds

    # Leader vehicle TODO: create leader model
    lv_samp = np.array([50, 50, 50, 10, 10, 10]) / 3.6 #leader speed check points 
    lv_speed = interp1d(t_samp, lv_samp, kind='quadratic') #quadratic interpolation -> no drivetrain limitation for now
    lv_x0 = 200
    lv_curr_state = { #initial state - Leader Vehicle
        'x': lv_x0,
        'v': float(lv_speed(0))
    }

    def get_lv_state(t_now):
            return lv_curr_state['x'], lv_curr_state['v']

    #Followers
    ini_gap = L_prec + 20 
    fv_v0 = 40/3.6
    fv_initials = [np.array([lv_x0 - ini_gap, fv_v0])]
    platoon_size = len(fv_initials) #excluding leader
    #will use the following to instantiate various decentralized MPC for each CAV
    model_params = {'h': h,
                    'd_min': d_min,
                    'L_prec': L_prec,
                    'dt': dt}

    mpc_config = {
            'n_horizon': 20,
            't_step': dt, #set equal to dt for model training
            'n_robust': 0, #for scenario based mpc -> see mpc.set_uncertainty_values()
            #...not really online refinement of the PINN parameters which is the goal
            'store_full_solution': True,
            'Q': 10,
            'R': 0.1,
            'd_min': 1}
    #*---

    # Building platoon...
    #same model for every vehicle (homogeneous platoon)
    pinn_model = SecondOrderCarModel("./models/onnx/pinn_c0_c1.onnx", model_params)
    plant_model = SecondOrderIdeal(model_params)
    print("Pinn model control input and states:", pinn_model.u.keys(), pinn_model.x.keys())
    print("Pinn model time varying parameters (provided):", pinn_model.tvp.keys())

    platoon = []
    for i in range(0, platoon_size):
        print(f"Setting MPC for follower {i}...")
        if i==0:
            get_prec_state = get_lv_state #set leader function
        else:
            #! not implemented
            raise NotImplementedError("MPC setup for followers beyond the leader is not implemented yet.")
        mpc = setupDMPC(pinn_model, mpc_config, get_prec_state)
        print("\n", mpc.settings)
        sim = setupSim(plant_model, t_step=dt, get_prec_state=get_prec_state)
        platoon.append(FV(fv_initials[i], mpc, sim)) #create follower and add to platoon

    #test:
    for fv in platoon:
        print(f"\n(FV{i}) Set Initial Conditions:")
        print(str(fv.mpc.x0['x_i']) + "m", str(fv.mpc.x0['v_i']*3.6) + "km/h")
        mpc_graphics, sim_graphics = setup_graphics(mpc.data, sim.data)
        fv.set_graphics(mpc_graphics, sim_graphics)

    #Control loop:
    fv0 = platoon[0]
    for i, t_value in enumerate(t):
        lv_curr_state['v'] = float(50/3.6) #!DEBUG
        lv_curr_state['x'] += lv_curr_state['v'] * dt
        u = fv0.mpc.make_step(fv0.state)
        fv0.state = fv0.sim.make_step(u/VEHICLE_MASS_KG) #TODO: PINN INPUT FOR NOW IS NOT MASS NORMALIZED (Force)

    fv0.mpc_graphics.plot_predictions(t_ind=100) #Additionally exemplify predictions for t=0
    fv0.sim_graphics.plot_results()
    fv0.sim_graphics.reset_axes()
    plt.show()

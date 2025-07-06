from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator
import numpy as np

from carla_sim.Core import Platoon, Vehicle

def setupDMPC(model: Model, config: dict, opt_params: dict, get_prec_state, platoon: Platoon, fv: Vehicle) -> MPC:
     #*MPC----
    mpc = MPC(model)
    mpc.set_param(**config)

    error_vec = model.aux['E']
    terminal_vec = model.aux['E_term']
    Q: list = opt_params['Q']
    Qu: list = opt_params['Qu']
    P = opt_params['P']
    R = opt_params['R']
    W = np.diag(Q+Qu) 
    lterm = error_vec.T @ W @ error_vec  #*LAGRANGE/ERROR PENALTY
    mterm = terminal_vec.T @ P @ terminal_vec  #*TERMINAL COST 
    mpc.set_objective(lterm=lterm, mterm=mterm) #, mterm=mterm)
    mpc.set_rterm(delta_u=R) #*ACTUATOR PENALTYs

    #*BOUNDS/CONSTRAINTS -> simple bound for output acc but could define non-linear one
    u_min = opt_params.get('u_min', -np.inf)
    u_max = opt_params.get('u_max', np.inf)
    mpc.bounds['lower', '_x', 'u'] = u_min
    mpc.bounds['upper', '_x', 'u'] = u_max

    v_min = opt_params.get('v_min', 0.0)
    v_max = opt_params.get('v_max', 120/3.6) # e.g., max velocity
    mpc.bounds['lower', '_x', 'v'] = v_min
    mpc.bounds['upper', '_x', 'v'] = v_max
     
    #set hard safety constraint. Note: there is only an upper bound argument so need to do the negative for lower:
    mpc.set_nl_cons('safety_dist_constraint', -model.aux['d'], ub=-opt_params.get('d_min', 1),
                     soft_constraint=False)
    
    #need to define time varying parameter callback over prediction horizon
    tvp_template = mpc.get_tvp_template()
    
    def tvp_fun(t_now): #create time varying parameter fetch function
        v_prec = get_prec_state(platoon, fv) #*get true (sensor) gap and prec vehicle speed (V2V)
        #print(f"  What mpc has called: {d}, {v_prec}")
        for k in range(mpc.settings.n_horizon+1):
                dt = mpc.settings.t_step
                t_pred = t_now + k * dt 
                tvp_template['_tvp',k,'t'] = t_pred
                tvp_template['_tvp',k,'v_prec'] = v_prec #*...and hold v constant 
                #tvp_template['_tvp',k,'x_prec'] = x_prec + k*dt*v_prec
        return tvp_template
    
    mpc.set_tvp_fun(tvp_fun)
    
    mpc.setup()
    return mpc

def setupSim(model: Model, sim_config: dict, get_prec_state, platoon: Platoon, fv: Vehicle) -> Simulator:
    sim = Simulator(model)
    sim.set_param(**sim_config)
    tvp_template = sim.get_tvp_template() #have to do this again, using the mpc one does not work
    
    def tvp_fun(t_now): #create time varying parameter fetch function
        v_prec = get_prec_state(platoon, fv) #*get preceeding vehicle state...
        tvp_template['t'] = t_now 
        tvp_template['v_prec'] = v_prec
        #tvp_template['x_prec'] = x_prec
        return tvp_template
    
    sim.set_tvp_fun(tvp_fun)
    
    sim.setup()
    return sim
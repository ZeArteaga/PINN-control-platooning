from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator
import numpy as np

def setupDMPC(model: Model, config: dict, get_prec_state) -> MPC:
     #*MPC----
    mpc = MPC(model)
    mpc.set_param(
        n_horizon=config['n_horizon'],
        t_step=config['t_step'],
        n_robust=config['n_robust'],
        store_full_solution=config.get('store_full_solution', True) #true default value
    )

    error_matrix = model.aux['E']
    gap_error = model.aux['e']

    Q = config['Q']
    P = config.get('P', 1)
    R = config['R'] 
    lterm = error_matrix.T @ Q @ error_matrix  #*LAGRANGE/ERROR PENALTY
    mterm = P *gap_error**2  #*TERMINAL COST 
    mpc.set_objective(lterm=lterm, mterm=mterm) #, mterm=mterm)
    mpc.set_rterm(u=R) #*ACTUATOR PENALTY

    #*BOUNDS/CONSTRAINTS -> simple bound for thrust force but could define non-linear one
    u_min = config.get('u_min', -np.inf)
    u_max = config.get('u_max', np.inf)
    mpc.bounds['lower', '_u', 'u'] = u_min
    mpc.bounds['upper', '_u', 'u'] = u_max

    v_min = config.get('v_min', 0.0)
    v_max = config.get('v_max', 120/3.6) # e.g., max velocity
    mpc.bounds['lower', '_x', 'v'] = v_min
    mpc.bounds['upper', '_x', 'v'] = v_max
    
    #position has no bounds i guess but spacing:
 
    #TODO: try to do this instead later, instead of specifying d_min in objective
    """ #set hard safety constraint. Note: there is only an upper bound argument so need to do the negative for lower:
    mpc.set_nl_cons('safety_dist_constraint', -model.aux['e_i'], ub=-config.get('d_min', 1),
                     soft_constraint=False)  """
    
    #need to define time varying parameter callback over prediction horizon
    tvp_template = mpc.get_tvp_template()
    
    def tvp_fun(t_now): #create time varying parameter fetch function
        x_prec, v_prec = get_prec_state(t_now) #*get preceeding vehicle state...
        print(f"  What mpc has called: {x_prec}, {v_prec}")
        for k in range(mpc.settings.n_horizon+1):
                dt = mpc.settings.t_step
                t_pred = t_now + k * dt 
                tvp_template['_tvp',k,'t'] = t_pred
                tvp_template['_tvp',k,'v_prec'] = v_prec #*...and hold v constant 
                tvp_template['_tvp',k,'x_prec'] = x_prec + k*dt*v_prec  #*keep updating position on constant velocity 
        return tvp_template
    
    mpc.set_tvp_fun(tvp_fun)
    
    mpc.setup()
    return mpc

def setupSim(model: Model, t_step, get_prec_state) -> Simulator:
    sim = Simulator(model)
    sim.set_param(t_step=t_step)
    tvp_template = sim.get_tvp_template() #have to do this again, using the mpc one does not work
    
    def tvp_fun(t_now): #create time varying parameter fetch function
        x_prec, v_prec = get_prec_state(t_now) #*get preceeding vehicle state...
        print(f"  What sim has called: {x_prec}, {v_prec}")
        tvp_template['t'] = t_now 
        tvp_template['x_prec'] = x_prec #*though this time no prediction is made since this sim model (plant)
        tvp_template['v_prec'] = v_prec
        return tvp_template
    
    sim.set_tvp_fun(tvp_fun)
    
    sim.setup()
    return sim
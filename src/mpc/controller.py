from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator
from carla_sim.Core import Platoon, Vehicle
import numpy as np

def setupDMPC(model: Model, config: dict, opt_params: dict, get_prec_state, platoon: Platoon, vehicle: Vehicle) -> MPC:
     #*MPC----
    mpc = MPC(model)
    mpc.set_param(**config)

    error_matrix = model.aux['E']
    gap_error = model.aux['e']

    Q = opt_params['Q']
    P = opt_params.get('P', 1)
    R = opt_params['R'] 
    lterm = error_matrix.T @ Q @ error_matrix  #*LAGRANGE/ERROR PENALTY
    mterm = P *gap_error**2  #*TERMINAL COST 
    mpc.set_objective(lterm=lterm, mterm=mterm) #, mterm=mterm)
    mpc.set_rterm(u=R) #*ACTUATOR PENALTY

    #*BOUNDS/CONSTRAINTS -> simple bound for thrust force but could define non-linear one
    u_min = opt_params.get('u_min', -np.inf)
    u_max = opt_params.get('u_max', np.inf)
    mpc.bounds['lower', '_u', 'u'] = u_min
    mpc.bounds['upper', '_u', 'u'] = u_max

    v_min = opt_params.get('v_min', 0.0)
    v_max = opt_params.get('v_max', 140/3.6) # e.g., max velocity
    mpc.bounds['lower', '_x', 'v'] = v_min
    mpc.bounds['upper', '_x', 'v'] = v_max
    
    #position has no bounds i guess but spacing:
 
    #TODO: try to do this instead later, instead of specifying d_min in objective
    """ #set hard safety constraint. Note: there is only an upper bound argument so need to do the negative for lower:
    mpc.set_nl_cons('safety_dist_constraint', -model.aux['e_i'], ub=-opt_params.get('d_min', 1),
                     soft_constraint=False)  """
    
    #need to define time varying parameter callback over prediction horizon
    tvp_template = mpc.get_tvp_template()
    
    def tvp_fun(t_now): #create time varying parameter fetch function
        v_prec = get_prec_state(platoon, vehicle) #*get true (sensor) gap and prec vehicle speed (V2V)
        #print(f"  What mpc has called: {d}, {v_prec}")
        for k in range(mpc.settings.n_horizon+1):
                dt = mpc.settings.t_step
                t_pred = t_now + k * dt 
                tvp_template['_tvp',k,'t'] = t_pred
                tvp_template['_tvp',k,'v_prec'] = v_prec #*...and hold v constant 
        return tvp_template
    
    mpc.set_tvp_fun(tvp_fun)
    
    mpc.setup()
    return mpc

def setupSim(model: Model, sim_config: dict, get_prec_state) -> Simulator:
    sim = Simulator(model)
    sim.set_param(**sim_config)
    tvp_template = sim.get_tvp_template() #have to do this again, using the mpc one does not work
    
    def tvp_fun(t_now): #create time varying parameter fetch function
        x_prec, v_prec = get_prec_state(t_now) #*get preceeding vehicle state...
        #print(f"  What sim has called: {x_prec}, {v_prec}")
        tvp_template['t'] = t_now 
        tvp_template['v_prec'] = v_prec
        return tvp_template
    
    sim.set_tvp_fun(tvp_fun)
    
    sim.setup()
    return sim
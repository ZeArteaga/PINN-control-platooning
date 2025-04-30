from do_mpc.sysid import ONNXConversion
from do_mpc.model import Model
from do_mpc.controller import MPC
import numpy as np
import onnx
import casadi

onnx_model = onnx.load("models/onnx/pinn_c0_c1.onnx")
casadi_converter = ONNXConversion(onnx_model)
#*TEST CASADI CONVERSION ----:
# print(casadi_converter)
# # Inputs can be numpy arrays
# casadi_converter.convert(input=np.zeros((1,4)))
# print(casadi_converter['output'])
#*-------

#*Do-MPC model setup
model = Model('discrete') #can be continous or discrete

#Variables
""" 
Uncertain parameters (_p) are decision variables you can estimate once and hold constant over each horizon.
Time-varying parameters (_tvp) are exogenous signals you supply at every time step but do not optimize internally """

t  = model.set_variable('_tvp', 't', shape=1)  # time varying parameter (time)
u  = model.set_variable('_u', 'u', shape=1)    # control input
X = model.set_variable('_x', 'x',shape=(1,2))    # [v_k, x_k]
dt = model.set_variable('_p', 'dt', shape=1) #fixed parameter

features = casadi.horzcat(t, u, X) #correct order of features
#print(features.shape)
casadi_converter.convert(input=features)
a_next = casadi_converter['output']

#discrete‐kinematics update
v_next = X[0] + a_next*dt
x_next = X[1] + X[0]*dt + 0.5*a_next*dt**2
model.set_rhs('x', casadi.vertcat(v_next, x_next)) #equations right hand side  
model.setup() #after this cannot setup more variables
#*----

mpc = MPC(model)
setup_mpc = {
    'n_horizon': 20,
    't_step': 0.1, #set equal to dt for model training
    'n_robust': 0, #for scenario based mpc -> see mpc.set_uncertainty_values()
    #...not really online refinment of the PINN parameters which is the goal
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

print(mpc.settings)
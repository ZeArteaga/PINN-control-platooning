import onnx
import casadi as ca 
from do_mpc.sysid import ONNXConversion
from do_mpc.model import Model

def _set_model_common_params(model, const_params: dict):
    """
    Creates the basic model structure with common variables and parameters.
    """
    h = const_params["h"]
    d_min = const_params["d_min"]
    L_prec = const_params["L_prec"]
    dt = const_params["dt"]

    u = model.set_variable('_u', 'u', shape=(1, 1))
    x = model.set_variable('_x', 'x', shape=(1, 1))
    v = model.set_variable('_x', 'v', shape=(1, 1))

    t = model.set_variable('_tvp', 't', shape=(1, 1)) # PINN input feature or general time
    x_prec = model.set_variable('_tvp', 'x_prec', shape=(1, 1))
    v_prec = model.set_variable('_tvp', 'v_prec', shape=(1, 1))
    
    return model, u, x, v, t, x_prec, v_prec, dt, h, d_min, L_prec

def _apply_kinematics_define_expr(
    model: Model, 
    x, v, a_next, 
    dt, x_prec, L_prec, d_min, h, v_prec):
    # Discrete‐kinematics update
    v_next = v + a_next*dt
    x_next = x + v_next*dt
    model.set_rhs('x', x_next)
    model.set_rhs('v', v_next)

    d_ref = d_min + h * v
    d = x_prec - x - L_prec 
    e = d - d_ref
    ev = v_prec - v - h * a_next
    E = ca.vertcat(e, ev)
    
    model.set_expression('d_ref', d_ref)
    model.set_expression('d', d)
    model.set_expression('e', e)
    model.set_expression('a', a_next) # For plotting/monitoring acceleration
    model.set_expression('E', E)


def SecondOrderPINNmodel(onnx_model_path: str, const_params: dict, **kwargs) -> Model: 
    onnx_model = onnx.load(onnx_model_path)
    ca_converter = ONNXConversion(onnx_model)
    #*TEST CASadi CONVERSION ----:
    # print(ca_converter)
    # # Inputs can be numpy arrays
    # ca_converter.convert(input=np.zeros((1,4)))
    # print(ca_converter['output'])
    #*-------

    #*Do-MPC model setup
    model = Model('discrete') #can be continous or discrete

    model, u, x, v, t, x_prec, v_prec, dt, h, d_min, L_prec = _set_model_common_params(model, const_params)

    features = ca.horzcat(t, u, v, x) #correct order of features
    #print(features.shape)
    ca_converter.convert(input=features)
    a_next = ca_converter['output'] #get PINN prediction

    _apply_kinematics_define_expr(model, x, v, a_next, dt, 
                                  x_prec, L_prec, d_min, h, v_prec) 
    
    model.setup() #after this cannot setup more variables
    return model

def SecondOrderIdeal(const_params: dict) -> Model:
    model = Model('discrete') #can be continous or discrete

    model, u, x, v, t, x_prec, v_prec, dt, h, d_min, L_prec = _set_model_common_params(model, const_params)
    m = const_params["m"]
    a_next = u / m

    _apply_kinematics_define_expr(model, x, v, a_next, dt, 
                                  x_prec, L_prec, d_min, h, v_prec) 
    
    model.setup()
    return model
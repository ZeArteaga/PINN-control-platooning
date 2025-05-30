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

    u = model.set_variable('_u', 'u', shape=(1, 1))
    x = model.set_variable('_x', 'x', shape=(1, 1))
    v = model.set_variable('_x', 'v', shape=(1, 1))

    t = model.set_variable('_tvp', 't', shape=(1, 1)) # PINN input feature or general time
    x_prec = model.set_variable('_tvp', 'x_prec', shape=(1, 1))
    v_prec = model.set_variable('_tvp', 'v_prec', shape=(1, 1))
    
    return model, u, x, v, t, x_prec, v_prec, h, d_min, L_prec

def _apply_kinematics_define_expr(
    model: Model, 
    x, v, a, x_prec, L_prec, d_min, h, v_prec):
    # Discrete‐kinematics update
    dvdt = a
    dxdt = v
    model.set_rhs('x', dxdt)
    model.set_rhs('v', dvdt)

    d_ref = d_min + h * v
    d = x_prec - x - L_prec 
    e = d - d_ref
    ev = v_prec - v - h * a
    E = ca.vertcat(e, ev)
    
    model.set_expression('d_ref', d_ref)
    model.set_expression('d', d)
    model.set_expression('e', e)
    model.set_expression('a', a) # For plotting/monitoring acceleration
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
    model = Model('continuous') #can be continous or discrete (have to handle discretization through euler or others)

    model, u, x, v, t, x_prec, v_prec, h, d_min, L_prec = _set_model_common_params(model, const_params)
    d = x_prec - x
    d_ref = d_min + h*v
    features = ca.horzcat(t, u, v, d, d_ref) #correct order of features
    ca_converter.convert(input=features)
    a = ca_converter['output'] #get PINN prediction

    _apply_kinematics_define_expr(model, x, v, a, 
                                  x_prec, L_prec, d_min, h, v_prec) 
    
    model.setup() #after this cannot setup more variables
    return model

def SecondOrderIdeal(const_params: dict) -> Model:
    model = Model('continuous') #can be continous or discrete

    model, u, x, v, t, x_prec, v_prec, h, d_min, L_prec = _set_model_common_params(model, const_params)
    m = const_params["m"]
    a = u / m

    _apply_kinematics_define_expr(model, x, v, a
                                  , 
                                  x_prec, L_prec, d_min, h, v_prec) 
    
    model.setup()
    return model
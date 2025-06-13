import onnx
import joblib
import casadi as ca 
from do_mpc.sysid import ONNXConversion
from do_mpc.model import Model

def _set_model_common_params(model, const_params: dict):
    """
    Adds to the basic model structure with common variables and parameters.
    """
    h = const_params["h"]
    d_min = const_params["d_min"]

    u = model.set_variable('_u', 'u', shape=(1, 1))
    #x = model.set_variable('_x', 'x', shape=(1, 1)) assuming 1D scenario else this is isn't needed
    d = model.set_variable('_x', 'd', shape=(1, 1))
    v = model.set_variable('_x', 'v', shape=(1, 1))

    t = model.set_variable('_tvp', 't', shape=(1, 1)) # PINN input feature
    v_prec = model.set_variable('_tvp', 'v_prec', shape=(1, 1))
    
    return h, d_min, u, v, d, t, v_prec

def _apply_kinematics_define_expr(
    model: Model, 
    v, a, d, d_min, h, v_prec):
    # Discrete‐kinematics update
    """ dxdt = v 
    model.set_rhs('x', dxdt) """
    dgapdt = v_prec - v #gap
    dvdt = a #vi
    model.set_rhs('d', dgapdt)
    model.set_rhs('v', dvdt)


    d_ref = d_min + h * v 
    e = d - d_ref
    ev = v_prec - v - h * a
    E = ca.vertcat(e, ev)
    
    model.set_expression('d_ref', d_ref)
    model.set_expression('d', d)
    model.set_expression('e', e)
    model.set_expression('a', a) # For plotting/monitoring acceleration
    model.set_expression('E', E)


def SecondOrderPINNmodel(onnx_model_path: str, const_params: dict,
                         scalerX_path: str = None, scalerY_path: str = None) -> Model: 
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

    h, d_min, u, v, d, t, v_prec = _set_model_common_params(model, const_params)
    
    d_ref = d_min + h*v
    X = ca.horzcat(t, u, v, d, d_ref) #correct order of features
    if scalerX_path:
        scalerX = joblib.load(scalerX_path)
        target_min = scalerX.feature_range[0]
        #normalize manually, without breaking CASadi
        scaleX = ca.DM(scalerX.scale_).reshape((1,-1))
        minX = ca.DM(scalerX.min_).reshape((1,-1))
        X = target_min + (X-minX)*scaleX
        
    ca_converter.convert(input=X)
    a = ca_converter['output'] #get PINN prediction
    if scalerY_path:
        scalerY = joblib.load(scalerY_path)
        a = a * scalerY.scale_ + scalerY.mean_  #in this case the scaler has single values/floats

    _apply_kinematics_define_expr(model, v, a, d, d_min, h, v_prec)
    
    model.setup() #after this cannot setup more variables
    return model

def SecondOrderIdeal(const_params: dict) -> Model:
    model = Model('continuous') #can be continous or discrete

    h, d_min, u, v, d, t, v_prec = _set_model_common_params(model, const_params)
    m = const_params["m"]
    a = u / m

    _apply_kinematics_define_expr(model, v, a, d, d_min, h, v_prec)
    
    model.setup()
    return model
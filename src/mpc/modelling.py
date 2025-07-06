import onnx
import numpy as np
import joblib
import casadi as ca 
from do_mpc.sysid import ONNXConversion
from do_mpc.model import Model

def _set_model_common_params(model, const_params: dict):
    """
    Adds to the basic model structure with common variables and parameters.
    """

    model.set_variable('_x', 'x', shape=(1, 1)) #assuming 1D scenario else this is isn't needed
    #model.set_variable('_x', 'd', shape=(1, 1))
    model.set_variable('_x', 'v', shape=(1, 1))
    model.set_variable('_u', 'delta_u', shape=(1,1))
    model.set_variable('_x', 'u', shape=(1, 1))
    
    model.set_variable('_tvp', 't', shape=(1, 1)) # PINN input feature
    model.set_variable('_tvp', 'x_prec', shape=(1, 1))
    model.set_variable('_tvp', 'v_prec', shape=(1, 1))
    
    return model

def _define_expressions(model: Model, d_min, h, L_prec, a):
    """Defines expressions common to both simulator and controller for monitoring."""
    x = model.x['x']
    v = model.x['v']
    x_prec = model.tvp['x_prec']
    v_prec = model.tvp['v_prec']
    d = x_prec - x - L_prec
    d_ref = d_min + h * v
    error_spacing = d - d_ref
    de = v_prec - v - h * a # Damping term for CTH policy
    error_rel_v = v_prec - v

    model.set_expression('d_ref', d_ref)
    model.set_expression('d', d)
    model.set_expression('e', error_spacing)
    model.set_expression('e_rel_v', error_rel_v)

    E = ca.vertcat(error_spacing, de, model.x['u']) # State cost vector
    E_term = ca.vertcat(error_spacing, error_rel_v, model.x['u']) # Terminal cost vector
    
    model.set_expression('E', E)
    model.set_expression('E_term', E_term)

    return model

def SecondOrderPINNmodel(onnx_model_path: str, const_params: dict,
                         scalerX_path: str = None, scalerY_path: str = None) -> Model: 
    onnx_model = onnx.load(onnx_model_path)    
    ca_converter = ONNXConversion(onnx_model)
    #*TEST CASadi CONVERSION ----:
    print(ca_converter)
    # # Inputs can be numpy arrays
    # ca_converter.convert(input=np.zeros((1,3)))
    # print(ca_converter['output'])
    #*-------

    #*Do-MPC model setup
    model = Model('continuous') #can be continous or discrete (have to handle discretization through euler or others)

    model = _set_model_common_params(model, const_params)

    h = const_params["h"]
    d_min = const_params["d_min"]
    L_prec = const_params['L_prec']

    X = ca.horzcat(model.tvp['t'], model.x['u'], model.x['v']) #correct order of features
    if scalerX_path:
        scalerX = joblib.load(scalerX_path)
        target_min = scalerX.feature_range[0]
        #normalize manually, without breaking CASadi
        scaleX = ca.DM(scalerX.scale_).reshape((1,-1))
        print(f"ScalerX scale: {scaleX}")
        minX = ca.DM(scalerX.min_).reshape((1,-1))
        print(f"ScalerX min: {minX}")
        X = target_min + (X-minX)*scaleX
    
    ca_converter.convert(args_0=X)
    a = ca_converter['output'] #get PINN prediction
    if scalerY_path:
        scalerY = joblib.load(scalerY_path)
        print(f"ScalerY scale: {scalerY.scale_}")
        print(f"ScalerY mean: {scalerY.mean_}")
        a = a * scalerY.scale_.item() + scalerY.mean_.item()  #in this case the scaler has single values/floats

    model = _define_expressions(model, d_min, h, L_prec, a)    
    
    dxdt = model.x['v']
    model.set_rhs('x', dxdt)
    #dgapdt = v_prec - v #gap
    dvdt = a 
    #model.set_rhs('d', dgapdt)
    model.set_rhs('v', dvdt)
    dudt = model.u['delta_u']
    model.set_rhs('u', dudt)
    e = model.aux['e']
    
    model.setup() #after this cannot setup more variables
    return model

def SecondOrderIdealPlant(const_params: dict) -> Model:
    model = Model('continuous') #can be continous or discrete

    model = _set_model_common_params(model, const_params)
    h = const_params["h"]
    d_min = const_params["d_min"]
    L_prec = const_params['L_prec']
    m = const_params["m"]

    dxdt = model.x['v']
    model.set_rhs('x', dxdt)
    #dgapdt = v_prec - v #gap
    a = model.x['u'] / m
    dvdt = a 
    #model.set_rhs('d', dgapdt)
    model.set_rhs('v', dvdt)
    dudt = model.u['delta_u']
    model.set_rhs('u', dudt)
    model = _define_expressions(model, d_min, h, L_prec, a)    
    e = model.aux['e']

    model.setup()
    return model

def ThirdOrderPlant(const_params: dict) -> Model:
    model = Model('continuous')
    h = const_params["h"]
    d_min = const_params["d_min"]
    L_prec = const_params['L_prec']
    tau = const_params["tau"]
    m = const_params["m"]
    model = _set_model_common_params(model, const_params)
    
    a = model.set_variable('_x', 'a', shape=(1,1))
    a_ref = model.x['u']/m
    dadt = 1/tau*(a_ref - a) 
    model.set_rhs('a', dadt)

    model = _define_expressions(model, d_min, h, L_prec, a)    

    dxdt = model.x['v']
    model.set_rhs('x', dxdt)
    #dgapdt = v_prec - v #gap
    dvdt = a 
    #model.set_rhs('d', dgapdt)
    model.set_rhs('v', dvdt)
    e = model.aux['e']
    dudt = model.u['delta_u']
    model.set_rhs('u', dudt)

    model.setup()
    return model
    
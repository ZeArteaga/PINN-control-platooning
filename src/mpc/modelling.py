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
    model.set_variable('_u', 'u', shape=(1, 1))
    model.set_variable('_x', 'x', shape=(1, 1)) #assuming 1D scenario else this is isn't needed
    #model.set_variable('_x', 'd', shape=(1, 1))
    model.set_variable('_x', 'v', shape=(1, 1))
    model.set_variable('_x', 'Ie', shape=(1, 1))

    model.set_variable('_tvp', 't', shape=(1, 1)) # PINN input feature
    model.set_variable('_tvp', 'x_prec', shape=(1, 1))
    model.set_variable('_tvp', 'v_prec', shape=(1, 1))
    
    return model

def policy_define_expr(
    model: Model, a, d_min, h, L_prec):
    # Discrete‐kinematics update
    x = model.x['x']
    v = model.x['v']
    x_prec = model.tvp['x_prec']
    v_prec = model.tvp['v_prec']

    d = x_prec - x - L_prec
    d_ref = d_min + h * v 
    error_spacing = d - d_ref
    de = v_prec - v - h * a
    Ie = model.x['Ie']
    E = ca.vertcat(error_spacing, de, Ie)
    
    error_rel_v = v_prec - v #for terminal cost
    terminal_vec = ca.vertcat(error_spacing, error_rel_v, Ie)
    

    model.set_expression('d_ref', d_ref)
    model.set_expression('d', d)
    model.set_expression('e', error_spacing)
    model.set_expression('e_rel_v', error_rel_v)
    model.set_expression('a', a) # For plotting/monitoring acceleration
    model.set_expression('E', E)
    model.set_expression('E_term', terminal_vec)

    return model

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

    model = _set_model_common_params(model, const_params)
    h = const_params["h"]
    d_min = const_params["d_min"]
    L_prec = const_params['L_prec']
    d_ref = d_min + h*model.x['v']
    d = model.tvp['x_prec'] - model.x['x'] - L_prec

    X = ca.horzcat(model.tvp['t'], model.u['u'], model.x['v'], d, d_ref, model.tvp['v_prec']) #correct order of features
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

    model = policy_define_expr(model, a, d_min, h, L_prec)
    dxdt = model.x['v']
    model.set_rhs('x', dxdt)
    #dgapdt = v_prec - v #gap
    dvdt = a 
    #model.set_rhs('d', dgapdt)
    model.set_rhs('v', dvdt)
    model.set_rhs("Ie", model.aux['e']) #set integral action
    
    model.setup() #after this cannot setup more variables
    return model

def DiscreteWindowedPINN(onnx_model_path: str, const_params: dict,
                         scalerX_path: str = None, scalerY_path: str = None) -> Model:
    onnx_model = onnx.load(onnx_model_path)    
    ca_converter = ONNXConversion(onnx_model)

    #*Do-MPC model setup
    model = Model('discrete')
    
    model = _set_model_common_params(model, const_params)

    h = const_params["h"]
    d_min = const_params["d_min"]
    L_prec = const_params['L_prec']
    dt = const_params['dt'] #since this one is discrete we need to perform integration manually

    window_size = const_params['window_size']
    X = ca.horzcat(model.tvp['t'], model.u['u'], model.x['v']) #*ensure correct order of features
    n_features = 3

    for i in range(1, window_size):
        model.set_variable('_x', 't_km'+ str(i), shape=(1, 1)) 
        model.set_variable('_x', 'v_km'+ str(i), shape=(1, 1))  #v[k-i]
        model.set_variable('_x', 'u_km'+ str(i), shape=(1, 1))  #u[k-i]
        old_feat = ca.horzcat(model.x['t_km'+ str(i)], model.x['u_km'+ str(i)], model.x['v_km'+ str(i)])
        X = ca.vertcat(old_feat, X)

    if scalerX_path:
        scalerX = joblib.load(scalerX_path)
        target_min = scalerX.feature_range[0]
        #normalize manually, without breaking CASadi
        scaleX = ca.DM(scalerX.scale_).reshape((1,-1))
        minX = ca.DM(scalerX.min_).reshape((1,-1))
        minX= ca.repmat(minX, window_size, 1) #replicate min value over window
        scaleX = ca.repmat(scaleX, window_size, 1)
        X = target_min + (X - minX) * scaleX

    X = X.reshape((1, -1)) #flatten window

    ca_converter.convert(input=X)
    
    a = ca_converter['output'] #get PINN prediction
    if scalerY_path:
        scalerY = joblib.load(scalerY_path)
        a = a * scalerY.scale_ + scalerY.mean_  #in this case the scaler has single values/floats

    model = policy_define_expr(model, a, d_min, h, L_prec)

    v_next = model.x['v'] + a*dt
    x_next = model.x['x'] + model.x['v']*dt + 0.5*a*dt**2
    Ie_next = model.x['Ie'] + model.aux['e']*dt
    model.set_rhs('v', v_next)
    model.set_rhs('x', x_next)
    model.set_rhs("Ie", Ie_next) #set integral action

    for i in range(1, window_size):
        if i==1:
            model.set_rhs('t_km'+ str(i), model.tvp['t'])
            model.set_rhs('v_km'+ str(i), model.x['v'])
            model.set_rhs('u_km'+ str(i), model.u['u'])
        else:
            model.set_rhs('t_km'+ str(i), model.x['t_km'+ str(i-1)])
            model.set_rhs('v_km'+ str(i), model.x['v_km'+ str(i-1)])
            model.set_rhs('u_km'+ str(i), model.x['u_km'+ str(i-1)])

    model.setup() #after this cannot setup more variables
    return model

def SecondOrderIdeal(const_params: dict) -> Model:
    model = Model('continuous') #can be continous or discrete

    model = _set_model_common_params(model, const_params)
    h = const_params["h"]
    d_min = const_params["d_min"]
    L_prec = const_params['L_prec']
    m = const_params["m"]

    model = policy_define_expr(model, a, d_min, h, L_prec)

    dxdt = model.x['v']
    model.set_rhs('x', dxdt)
    #dgapdt = v_prec - v #gap
    a_target = model.u['u'] / m
    dvdt = a_target 
    #model.set_rhs('d', dgapdt)
    model.set_rhs('v', dvdt)

    model.set_rhs("Ie", model.aux['e']) #set integral action

    model.setup()
    return model

def ThirdOrderModel(const_params: dict) -> Model:
    model = Model('continuous')
    h = const_params["h"]
    d_min = const_params["d_min"]
    L_prec = const_params['L_prec']
    tau = const_params["tau"]
    m = const_params["m"]
    model = _set_model_common_params(model, const_params)
    
    a = model.set_variable('_x', 'a', shape=(1,1))
    a_target = model.u['u']/m
    dadt = 1/tau*(a_target - a) 
    model.set_rhs('a', dadt)

    model = policy_define_expr(model, a, d_min, h, L_prec)
    
    dxdt = model.x['v']
    model.set_rhs('x', dxdt)
    #dgapdt = v_prec - v #gap
    dvdt = a 
    #model.set_rhs('d', dgapdt)
    model.set_rhs('v', dvdt)

    model.set_rhs("Ie", model.aux['e']) #set integral action

    model.setup()
    return model
    
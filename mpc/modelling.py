import onnx
import casadi as ca 
from do_mpc.sysid import ONNXConversion
from do_mpc.model import Model

def SecondOrderCarModel(onnx_model_path: str, const_params: dict) -> Model: 
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

    #Likely dont need these, already defined by PINN:
    """ m = model.set_variable(var_type='_p', var_name='m')
    A = model.set_variable(var_type='_p', var_name='A')
    Cd = model.set_variable(var_type='_p', var_name='Cd')
    p = model.set_variable(var_type='_p', var_name='rho')
    c0 = model.set_variable(var_type='_p', var_name='c0')
    c1 = model.set_variable(var_type='_p', var_name='c1')
    th = model.set_variable(var_type='_p', var_name='th')
    g = model.set_variable(var_type='_p', var_name='g') 

    phi = ca.vertcat(1, v_i, v_i**2)
    B = ca.vertcat(
        c0*g*ca.cos(th)+g*ca.sin(th),
        c1*g*ca.cos(th),
        1/2*p/m*Cd*A
    )
    """
    #*spacing policy parameters
    h = const_params["h"]
    d_min = const_params["d_min"]
    L_prec = const_params["L_prec"]
    dt = const_params["dt"]

    #*Variables
    u_i  = model.set_variable('_u', 'u_i', shape=(1,1)) # control input
    x_i = model.set_variable('_x', 'x_i',shape=(1,1))
    v_i = model.set_variable('_x', 'v_i',shape=(1,1))

    #*Time-varying parameters
    t  = model.set_variable('_tvp', 't', shape=(1,1))  # PINN input feature 
    x_prec = model.set_variable('_tvp', 'x_prec', shape=(1,1)) #preceeding (i-1) CAV position
    v_prec = model.set_variable('_tvp', 'v_prec', shape=(1,1)) # "" "" "" velocity

    features = ca.horzcat(t, u_i, v_i, x_i) #correct order of features
    #print(features.shape)
    ca_converter.convert(input=features)
    a_i_next = ca_converter['output'] #get PINN prediction

    #discrete‐kinematics update
    v_i_next = v_i + a_i_next*dt
    x_i_next = x_i + v_i*dt + 0.5*a_i_next*dt**2
    model.set_rhs('x_i', x_i_next) #equations right hand side
    model.set_rhs('v_i', v_i_next)

    #Define error expression
    d_ref_i = d_min + h * v_i
    d_i = x_prec - x_i - L_prec #include prec CAV length for actual gap
    e_i = d_i - d_ref_i
    model.set_expression('d_ref_i', d_ref_i) #adds expression for monotoring
    model.set_expression('d_i', d_i)
    model.set_expression('e_i', e_i)

    model.setup() #after this cannot setup more variables
    return model

def SecondOrderIdeal(const_params: dict) -> Model:
    model = Model('discrete')

    h = const_params["h"]
    d_min = const_params["d_min"]
    L_prec = const_params["L_prec"]
    dt = const_params["dt"]

    #*Variables
    u_i  = model.set_variable('_u', 'u_i', shape=(1,1)) # control input
    x_i = model.set_variable('_x', 'x_i',shape=(1,1))
    v_i = model.set_variable('_x', 'v_i',shape=(1,1))

    #*Time-varying parameters
    t  = model.set_variable('_tvp', 't', shape=(1,1)) 
    x_prec = model.set_variable('_tvp', 'x_prec', shape=(1,1)) #preceeding (i-1) CAV position
    v_prec = model.set_variable('_tvp', 'v_prec', shape=(1,1)) # "" "" "" velocity

    a_i_next = u_i

    #discrete‐kinematics update
    v_i_next = v_i + a_i_next*dt
    x_i_next = x_i + v_i*dt + 0.5*a_i_next*dt**2
    model.set_rhs('x_i', x_i_next) #equations right hand side
    model.set_rhs('v_i', v_i_next)

    #Define error expression
    d_ref_i = d_min + h * v_i
    d_i = x_prec - x_i - L_prec #include prec CAV length for actual gap
    e_i = d_i - d_ref_i
    model.set_expression('d_ref_i', d_ref_i) #adds expression for monotoring
    model.set_expression('d_i', d_i)
    model.set_expression('e_i', e_i)

    model.setup() #after this cannot setup more variables
    return model
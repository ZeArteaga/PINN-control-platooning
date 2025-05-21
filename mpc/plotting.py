import matplotlib.pyplot as plt
import matplotlib as mpl
from do_mpc.graphics import Graphics

def setup_graphics(mpc_data, sim_data):
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.grid'] = True

    mpc_graphics = Graphics(mpc_data)
    sim_graphics = Graphics(sim_data)

    n_subplot = 4
    fig, ax = plt.subplots(n_subplot, sharex=True, figsize=(16, 9))
    fig.align_ylabels()

    for g, name in zip([sim_graphics, mpc_graphics], ["(sim)", "(mpc)"]):
        g.add_line(var_type='_x', var_name='x', axis=ax[0], label="x_i "+ name)
        g.add_line(var_type='_tvp', var_name='x_prec', axis=ax[0], label="x_i-1 "+ name)
        g.add_line(var_type='_aux', var_name='d_ref', axis=ax[1], label="target gap " + name)
        g.add_line(var_type='_aux', var_name='d', axis=ax[1], label="actual gap " + name)
        g.add_line(var_type='_aux', var_name='e', axis=ax[1], label="error " + name)
        g.add_line(var_type='_x', var_name='v', axis=ax[2], label="v_i " + name)
        g.add_line(var_type='_tvp', var_name='v_prec', axis=ax[2], label="v_i-1 " + name)
        g.add_line(var_type='_u', var_name='u', axis=ax[3], label="Control input - Thrust (N) " + name)

    ax[0].set_ylabel('Position (m)')
    ax[1].set_ylabel('(m)')
    ax[2].set_ylabel('Velocity (m/s)')

    for i in range(0,n_subplot):
        ax[i].legend()
        ax[i].set_xlabel('Time [s]')
    
    fig.tight_layout()

    return mpc_graphics, sim_graphics
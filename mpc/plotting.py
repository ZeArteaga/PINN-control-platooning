import matplotlib.pyplot as plt
import matplotlib as mpl
from do_mpc.graphics import Graphics

def setup_graphics(mpc_data, sim_data):
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['axes.grid'] = True

    mpc_graphics = Graphics(mpc_data)
    sim_graphics = Graphics(sim_data)

    n_subplot = 4
    fig, ax = plt.subplots(n_subplot, sharex=True, figsize=(16, 9))
    fig.align_ylabels()

    for g in [sim_graphics, mpc_graphics]:
        g.add_line(var_type='_x', var_name='x_i', axis=ax[0], label="x_i")
        g.add_line(var_type='_aux', var_name='d_ref_i', axis=ax[1], label="target gap")
        g.add_line(var_type='_aux', var_name='d_i', axis=ax[1], label="actual gap")
        g.add_line(var_type='_aux', var_name='e_i', axis=ax[1], label="error")
        g.add_line(var_type='_x', var_name='v_i', axis=ax[2], label="v_i")
        g.add_line(var_type='_tvp', var_name='v_prec', axis=ax[2], label="v_i-1")
        g.add_line(var_type='_u', var_name='u_i', axis=ax[3], label="u_i")

    ax[0].set_ylabel('Position (m)')
    ax[1].set_ylabel('(m)')
    ax[2].set_ylabel('Velocity (m/s)')
    ax[3].set_ylabel('Control input - thrust force (N)')
    for i in range(0,n_subplot):
        ax[i].legend()
        ax[i].set_xlabel('Time [s]')
    
    fig.tight_layout()

    return mpc_graphics, sim_graphics
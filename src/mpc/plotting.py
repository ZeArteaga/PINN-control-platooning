import matplotlib.pyplot as plt
import matplotlib as mpl
from do_mpc.graphics import Graphics

def setup_graphics(mpc_data, sim_data):
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['axes.grid'] = True

    mpc_graphics = Graphics(mpc_data)
    sim_graphics = Graphics(sim_data)

    return mpc_graphics, sim_graphics

def setup_plot(g: Graphics, ax: plt.axes, with_label: bool=True):
    def label(l): return l if with_label else None
    # Define colors for each variable
    colors = {
        'x': 'C0', 'x_prec': 'C1',
        'd_ref': 'C1', 'd': 'C2', 'e': 'C3',
        'v': 'C4', 'v_prec': 'C5',
        'u': 'C6', 'delta_u': 'C6'
    }

    g.add_line(var_type='_x', var_name='x', axis=ax[0], label=label("Ego CAV Position"), color=colors['x'])
    g.add_line(var_type='_tvp', var_name='x_prec', axis=ax[0], label=label("Preceeding CAV Position"), color=colors['x_prec'])

    g.add_line(var_type='_aux', var_name='d_ref', axis=ax[1], label=label("target gap"), color=colors['d_ref'])
    g.add_line(var_type='_aux', var_name='d', axis=ax[1], label=label("actual gap"), color=colors['d'])
    g.add_line(var_type='_aux', var_name='e', axis=ax[1], label=label("spacing error"), color=colors['e'])

    g.add_line(var_type='_x', var_name='v', axis=ax[2], label=label("Ego CAV Speed"), color=colors['v'])
    g.add_line(var_type='_tvp', var_name='v_prec', axis=ax[2], label=label("Preceeding CAV Speed"), color=colors['v_prec'])

    g.add_line(var_type='_x', var_name='u', axis=ax[3], label=label("control input $u(t)$ (Thrust/Brake)"), color=colors['u'])
    
    ax[0].set_ylabel('Position (m)')
    ax[1].set_ylabel('(m)')
    ax[2].set_ylabel('Speed (m/s)')
    ax[3].set_ylabel('Force (N)')

    for axis in ax:
        axis.legend(loc="upper right")
        axis.set_xlabel('Time [s]')

    return ax

def plot(sim_graphics, mpc_graphics, pred_t=None):
    n_subplot = 4
    fig, ax = plt.subplots(n_subplot, sharex=True, figsize=(16, 9))
    fig.align_ylabels()

    setup_plot(sim_graphics, ax)
    setup_plot(mpc_graphics, ax, with_label=False)
    sim_graphics.plot_results()
    if pred_t is not None:
        mpc_graphics.plot_predictions(t_ind=pred_t)

    plt.tight_layout()
    plt.show()
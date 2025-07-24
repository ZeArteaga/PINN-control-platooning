import matplotlib.pyplot as plt
import matplotlib as mpl
from do_mpc.graphics import Graphics

def setup_graphics(data):
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['axes.grid'] = True

    graphics = Graphics(data)

    return graphics

def setup_plot(g: Graphics, ax: plt.axes, ax3_twin: plt.axes, with_label: bool=True):
    def label(l): return l if with_label else None
    # Define colors for each variable
    colors = {
        'x': 'C0', 'x_prec': 'C1',
        'd_ref': 'C1', 'd': 'C2', 'e': 'C3',
        'v': 'C4', 'v_prec': 'C5',
        'u': 'C6', 'delta_u': 'C6',
        'a_ref': 'C7'
    }

    g.add_line(var_type='_x', var_name='x', axis=ax[0], label=label("$x_0$"), color=colors['x'])
    g.add_line(var_type='_tvp', var_name='x_prec', axis=ax[0], label=label("$x_{lv}$"), color=colors['x_prec'])
    
    g.add_line(var_type='_aux', var_name='d_ref', axis=ax[1], label=label("$d^*_0$"), color=colors['d_ref'])
    g.add_line(var_type='_aux', var_name='d', axis=ax[1], label=label("$d_0$"), color=colors['d'])
    g.add_line(var_type='_aux', var_name='e', axis=ax[1], label=label("$e_{s,0}$"), color=colors['e'])
    
    g.add_line(var_type='_aux', var_name='v_kmh', axis=ax[2], label=label("$v_0$"), color=colors['v'])
    g.add_line(var_type='_aux', var_name='v_prec_kmh', axis=ax[2], label=label("$v_{lv}$"), color=colors['v_prec'])
    
    g.add_line(var_type='_aux', var_name='a_ref', axis=ax[3], color=colors['a_ref'])
    g.add_line(var_type='_x', var_name='u', axis=ax3_twin, label=label("$u_0$"), color=colors['u'])
    
    ax[0].set_ylabel('Absolute Position (m)')
    ax[1].set_ylabel('Relative Position (m)')
    ax[2].set_ylabel('Speed (km/h)')
    ax3_twin.set_ylabel('Force (N)')
    ax[3].set_ylabel('Acceleration (m/s$^2$)')

    for i, axis in enumerate(ax):
        if i == 3:
            ax3_twin.legend(loc='upper right')
        else:
            axis.legend(loc="upper right")
        axis.set_xlabel('Time (s)')

    return ax

def plot(mpc_graphics, sim_graphics = None, pred_t=None):
    n_subplot = 4
    fig, ax = plt.subplots(n_subplot, sharex=True, figsize=(16, 9))
    ax3_twin = ax[3].twinx()
    fig.align_ylabels()

    if sim_graphics:
        setup_plot(sim_graphics, ax, ax3_twin)
        setup_plot(mpc_graphics, ax, ax3_twin, with_label=False)
        sim_graphics.plot_results()
    else:
        setup_plot(mpc_graphics, ax, ax3_twin, with_label=True)
        mpc_graphics.plot_results()
    if pred_t is not None:
        #predicitons
        mpc_graphics.plot_predictions(t_ind=pred_t)
        for line in mpc_graphics.pred_lines.full:
            line.set_label('_nolegend_') 
        
        # Update legends on all axes, including the twin
        for i, a in enumerate(ax):
            if i == 3:
                ax3_twin.legend(loc='upper right')
            else:
                a.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
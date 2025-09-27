import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
import numpy as np
from scipy.interpolate import interp1d

COL = {
    'd_ref':     'tab:orange',
    'd_actual':  'tab:green',
    'd_err':     'tab:red',

    'v_follower':'tab:purple',
    'v_prec':    'tab:brown',

    'u':         'tab:blue', 
    'a_out':     'tab:pink',
    'a_ref':     'tab:cyan',
}

def _zoh(mpc_t, y):
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return interp1d(
        np.asarray(mpc_t).flatten(), y, kind='zero', axis=0,
        bounds_error=False, fill_value='extrapolate'
    )

def plot_follower_results(data, follower_id):
    sim_data = data['sim']
    mpc_data = data['mpc']

    sim_time = sim_data["time"].flatten()
    mpc_time = mpc_data["time"].flatten()
    aux_keys = mpc_data['aux'].keys()
    tvp_keys = mpc_data['tvp'].keys()

    fig, ax = plt.subplots(3, sharex=True, figsize=(16, 9))
    #fig.suptitle(f'Results for Follower Vehicle {follower_id}')

    # 1) Spacing/Gap
    if 'd_ref' in aux_keys:
        d_ref_zoh = _zoh(mpc_time, mpc_data['aux']['d_ref'])(sim_time)
        ax[0].plot(sim_time, d_ref_zoh, label='Target spacing (MPC)', color=COL['d_ref'])
    sim_d = sim_data.get('gap') or sim_data.get('d')
    if sim_d is not None:
        ax[0].plot(sim_time, sim_d, label='Actual spacing', color=COL['d_actual'])
    if 'e' in aux_keys:
        e_zoh = _zoh(mpc_time, mpc_data['aux']['e'])(sim_time)
        ax[0].plot(sim_time, e_zoh, label='Spacing error (MPC)', color=COL['d_err'])
    ax[0].set_ylabel('Gap (m)')
    ax[0].legend(loc="lower left")

    # 2) Velocity
    sim_v = sim_data.get('v')
    if sim_v is not None:
        ax[1].plot(sim_time, np.asarray(sim_v, float)*3.6, label="Follower vehicle's speed", color=COL['v_follower'])
    if 'v_prec' in tvp_keys:
        v_prec_zoh = _zoh(mpc_time, mpc_data['tvp']['v_prec'])(sim_time)
        ax[1].plot(sim_time, v_prec_zoh*3.6, label="Preceding vehicle's speed (MPC)", color=COL['v_prec'])
    ax[1].set_ylabel('Velocity (km/h)')
    ax[1].legend(loc="lower left")

    # 3) Input (u) and accelerations on twinx
    # Left axis: force
    sim_u = sim_data.get('u')
    left_color = COL['u']
    if sim_u is not None:
        ax[2].plot(sim_time, np.asarray(sim_u, float), label='Input longitudinal force $u$', color=left_color, zorder=3)
    ax[2].set_ylabel('Force (N)', color=left_color)
    ax[2].tick_params(axis='y', colors=left_color)
    ax[2].spines['left'].set_color(left_color)
    ax[2].spines['left'].set_linewidth(3)

    # Right axis: accelerations (reference and measured)
    ax_r = ax[2].twinx()
    right_color = COL['a_out']  # prefer measured accel color for the axis
    sim_acc = sim_data.get('a_out') or sim_data.get('acc')
    if sim_acc is not None:
        ax_r.plot(sim_time, np.asarray(sim_acc, float), label='Output acceleration', color=COL['a_out'], zorder=1)
    sim_a_ref = sim_data.get('acc_ref')
    if sim_a_ref is not None:
        ax_r.plot(sim_time, np.asarray(sim_a_ref, float), label='Input equiv. acceleration', color=COL['a_ref'], zorder=3)
        # If only ref is plotted, color axis with ref tone
        if sim_acc is None:
            right_color = COL['a_ref']
    ax_r.set_ylabel('Acceleration (m/s²)', color="black")
    ax_r.tick_params(axis='y', colors=COL['a_ref'])
    ax_r.spines['right'].set_color(right_color)
    ax_r.spines['right'].set_linewidth(3)

    # Combined legend
    h1, l1 = ax[2].get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    leg = ax[2].legend(h1 + h2, l1 + l2, loc="best", frameon=True, facecolor='white')
    leg.set_zorder(100)
    leg.set_alpha(1)

    plt.tight_layout()
    fig.supxlabel('Time (s)')
    plt.show()

def plot_platoon_results(all_data):
    fig, ax = plt.subplots(2, sharex=True, figsize=(16, 9))
    #fig.suptitle('Platoon Performance')

    # --- Plot 1: Velocities ---
    ax[0].set_title('CAV Velocities')
    ax[0].set_ylabel('Velocity (km/h)')

    # Leader velocity
    if all_data:
        leader = all_data[0]['data']['sim']
        t = np.asarray(leader['time']).flatten()
        ax[0].plot(t, np.array(leader['v']) * 3.6, label='Leader', color='black')

        # Followers' velocities
        for item in all_data[1:]:
            idx = item['index'] - 1
            sim = item['data']['sim']
            if 'v' in sim:
                ax[0].plot(t, np.asarray(sim['v'], float).flatten() * 3.6, label=f'Follower {idx}')
        ax[0].legend(loc="upper left")

        # --- Plot 2: Spacing Error (gap - d_ref) ---
        ax[1].set_title('Spacing Error (Actual Gap - Target Gap)')
        ax[1].set_ylabel('Spacing Error (m)')
        ax[1].axhline(0, color='black', linestyle='--', linewidth=1)

        for item in all_data[1:]:
            idx = item['index']
            sim = item['data']['sim']
            mpc = item['data']['mpc']
            sim_t = np.asarray(sim.get('time', []), float).reshape(-1)
            mpc_t = np.asarray(mpc.get('time', []), float).reshape(-1)


            d = sim.get('gap') or sim.get('d')
            if d is None or sim_t.size == 0:
                continue
            d = np.asarray(d, float).reshape(-1)
            if 'aux' in mpc.keys():
                if "d_ref" in mpc['aux'].keys():
                    dref = _zoh(mpc_t, mpc['aux']['d_ref'])(sim_t).reshape(-1)
                    err = d - dref
                    ax[1].plot(sim_t, err, label=f'Follower {idx-1}')
                    print(f"Follower {idx-1} spacing error l2-norm: {np.linalg.norm(err, 2)}")

        ax[1].legend(loc="upper left")

    plt.tight_layout()
    fig.supxlabel('Time (s)')
    plt.show()

def plot_platoon_path(all_data):
    """
    Plots the XY trajectories of all vehicles in the platoon.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    #ax.set_title('Platoon Trajectories')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.yaxis.set_inverted(True) #carla uses left handed coord system
    ax.set_aspect('equal', adjustable='box') # Ensures X and Y axes are scaled the same
    ax.grid(True)

    if all_data:
        leader_data = all_data[0]['data']['sim']
        if 'xy' in leader_data and leader_data['xy']:
            x_coords, y_coords = zip(*leader_data['xy'])
            ax.plot(x_coords, y_coords, label='Leader', color='black', linewidth=2)
            start_x, start_y = x_coords[0], y_coords[0]
            ax.plot(start_x, start_y, '*', markersize=15, color='gold', markeredgecolor='black', label='Leader Spawning Point', zorder=5)
            
            # Plot Finish Marker (red 'x')
            finish_x, finish_y = x_coords[-1], y_coords[-1]
            ax.plot(finish_x, finish_y, 'X', markersize=12, color='red', label='Leader Destination', zorder=5)
    #    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
        for i, item in enumerate(all_data[1:]):
            follower_idx = item.get('index', i + 1) - 1 # Get follower index (0, 1, 2...)
            sim_data = item.get('data', {}).get('sim', {})
            if 'xy' in sim_data and sim_data['xy']:
                x_coords, y_coords = zip(*sim_data['xy'])
                #color = color_cycle[i % len(color_cycle)] # Cycle through colors
                ax.plot(x_coords, y_coords, label=f'Follower {follower_idx}')

    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

def main():
    results_dir = os.path.join(os.path.dirname(__file__), 'results/')
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found.")
        return

    mpl.rcParams['font.size'] = 12
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams['axes.grid'] = True

    all_vehicle_data = []
    for file_name in sorted(os.listdir(results_dir)):
        if file_name.startswith("vehicle_") and file_name.endswith(".pkl"):
            v_id = int(file_name.split('_')[-1].split('.')[0])
            file_path = os.path.join(results_dir, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            all_vehicle_data.append({'index': v_id, 'data': data})

    #individual results
    for item in all_vehicle_data[1:]:
        print(f"Plotting individual results for platoon vehicle {item['index']}")
        plot_follower_results(item['data'], item['index']-1)

    #platoon results
    if all_vehicle_data:
        print("\nPlotting collective platoon results...")
        plot_platoon_results(all_vehicle_data)
        plot_platoon_path(all_vehicle_data)

if __name__ == '__main__':
    main()

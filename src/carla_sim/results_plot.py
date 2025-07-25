import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
import numpy as np

def plot_follower_results(data, follower_id):
    time = data['_time'].flatten()
    aux_keys = data['aux'].keys()
    x_keys = data['x'].keys()
    print(x_keys)
    tvp_keys = data['tvp'].keys()
    u_keys = data['u'].keys()

    fig, ax = plt.subplots(3, sharex=True, figsize=(16, 9))
    fig.suptitle(f'Results for Follower Vehicle {follower_id}')
    
    # 1. Gap (actual and target)
    if 'd_ref' in aux_keys:
        ax[0].plot(time, data['aux']['d_ref'], label='Target gap', color='C1', linestyle='--')
    if 'd' in aux_keys:
        ax[0].plot(time, data['x']['d'], label='Actual gap', color='C2')
    if 'e' in aux_keys:
        ax[0].plot(time, data['aux']['e'], label='Gap error', color='C3')
    ax[0].set_ylabel('Gap (m)')
    ax[0].legend()

    # 2. Velocity (follower and leader)
    if 'v' in x_keys:
        ax[1].plot(time, data['x']['v']*3.6, label='Follower vehicle\'s speed', color='C4')
    if 'v_prec' in tvp_keys:
        ax[1].plot(time, data['tvp']['v_prec']*3.6, label='Preceeding vehicle\'s speed', color='C5', linestyle='--')
    ax[1].set_ylabel('Velocity (km/h)')
    ax[1].legend()

    # 3. Acceleration (if available)
    if 'u' in u_keys:
        ax[2].plot(time, data['u']['u'], label='Reference input force (u)', color='C6')
    #if 'delta_u' in u_keys:
        #ax[2].plot(time, data['u']['delta_u'], label='Control variable (du/dt)', color='C7')
    ax[2].set_ylabel('Force (N)')
    ax[2].legend()

    plt.tight_layout()
    fig.supxlabel('Time (s)')

    plt.show()

def plot_platoon_results(all_data):
    fig, ax = plt.subplots(2, sharex=True, figsize=(16, 9))
    fig.suptitle('Platoon Performance')

    # --- Plot 1: Velocities ---
    ax[0].set_title('Vehicle Velocities')
    ax[0].set_ylabel('Velocity (km/h)')
    
    # Plot leader's velocity from the first follower's data
    if all_data:
        first_follower_data = all_data[0]['data']
        time = first_follower_data['_time'].flatten()
        if 'v_prec' in first_follower_data['tvp']:
            ax[0].plot(time, first_follower_data['tvp']['v_prec'] * 3.6, label='Leader', color='black', linestyle='--')

    # Plot followers' velocities
    for item in all_data:
        follower_id = item['id']
        data = item['data']
        time = data['_time'].flatten()
        if 'v' in data['x']:
            ax[0].plot(time, data['x']['v'] * 3.6, label=f'Follower {follower_id}')
    ax[0].legend()

    # --- Plot 2: Spacing Error ---
    ax[1].set_title('Follower Spacing Error (Actual Gap - Target Gap)')
    ax[1].set_ylabel('Spacing Error (m)')
    ax[1].axhline(0, color='black', linestyle='--', linewidth=1) # Zero error line

    for item in all_data:
        follower_id = item['id']
        data = item['data']
        time = data['_time'].flatten()
        # Calculate spacing error if not present
        if 'e' not in data['aux'] and 'd' in data['aux'] and 'd_ref' in data['aux']:
             data['aux']['e'] = data['aux']['d'] - data['aux']['d_ref']

        if 'e' in data['aux']:
            ax[1].plot(time, data['aux']['e'], label=f'Follower {follower_id}')
    ax[1].legend()

    plt.tight_layout()
    fig.supxlabel('Time (s)')
    plt.show()


def main():
    results_dir = os.path.join(os.path.dirname(__file__), 'results/')
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found.")
        return

    mpl.rcParams['font.size'] = 12
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.grid'] = True

    all_follower_data = []
    # First, collect all data and sort by follower ID
    for file_name in sorted(os.listdir(results_dir)):
        if file_name.startswith("follower_") and file_name.endswith(".pkl"):
            follower_id = int(file_name.split('_')[-1].split('.')[0])
            file_path = os.path.join(results_dir, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            all_follower_data.append({'id': follower_id, 'data': data})

    #individual results
    for item in all_follower_data:
        print(f"Plotting individual results for follower {item['id']}")
        plot_follower_results(item['data'], item['id'])

    #platoon results
    if all_follower_data:
        print("\nPlotting collective platoon results...")
        plot_platoon_results(all_follower_data)


if __name__ == '__main__':
    main()

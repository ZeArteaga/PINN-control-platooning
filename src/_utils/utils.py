import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

def load_yaml_file(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def calc_req_input_from_acc(v: np.ndarray, a: np.ndarray, params: dict) -> np.ndarray:
    c0 = params["c0"]
    c1 = params["c1"]
    m = params["m"]
    g = params["g"]
    road_grade = params["road_grade"]
    p = params["p"]
    Cd = params["Cd"]
    Af = params["Af"]

    Fr = (c0 + c1*v)*(m*g*np.cos(road_grade))
    Fa = (p*Cd*Af*v**2)/2 #drag
    Fg = m*g*np.sin(road_grade)
    a_input = a + (Fr + Fa + Fg)/m
    return a_input

def save_trajectory_plot(data: dict, path: str, traj_id: str) -> pd.DataFrame:
    df = pd.DataFrame(data, dtype=np.float32)
    df.to_csv(path, index=False)
    print(f"Saved CACC trajectory {traj_id} to {path}")


    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(10)]
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(f"CACC Follower/Leader Simulation: {traj_id}", fontsize=20)

    # 1. Vehicle Positions
    axs[0, 0].plot(df['t'], df['fv0_x'], label='Follower Position (m)', color=colors[0])
    axs[0, 0].plot(df['t'], df['lv_x'], label='Leader Position (m)', color=colors[1])
    axs[0, 0].set_title("Vehicle Positions")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Position (m)")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # 2. Vehicle Velocities
    axs[0, 1].plot(df['t'], df['fv0_v'] * 3.6, label='Follower Velocity (km/h)', color=colors[0])
    axs[0, 1].plot(df['t'], df['lv_v'] * 3.6, label='Leader Velocity (km/h)', color=colors[1])
    if 'fv0_v_noise' in df:
        axs[0, 1].scatter(df['t'], df['fv0_v_noise'] * 3.6, label='Follower Noisy Velocity (km/h)', color=colors[2], marker='x', s=1)
    if 'lv_v_noise' in df:
        axs[0, 1].scatter(df['t'], df['lv_v_noise'] * 3.6, label='Leader Noisy Velocity (km/h)', color=colors[3], marker='x', s=1)
    axs[0, 1].set_title("Vehicle Velocities")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Velocity (km/h)")
    axs[0, 1].legend()
    axs[0, 1].grid()

    # 3. Follower Acceleration and Input
    color_acc = colors[0]
    color_acc_noise = colors[2]
    color_u = colors[4]
    axs[1, 0].plot(df['t'], df['fv0_a'], label='Follower Acceleration (m/s²)', color=color_acc)
    if 'fv0_a_noise' in df:
        axs[1, 0].scatter(df['t'], df['fv0_a_noise'], label='Noisy Acceleration (m/s²)', color=color_acc_noise, marker='x', s=1)
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Acceleration (m/s²)")
    axs[1, 0].tick_params(axis="y", colors=color_acc)
    axs[1, 0].plot(df['t'], df['fv0_u'], label='Control input u (m/s²)', color=color_u, linestyle='--')
    axs[1, 0].tick_params(axis="y", colors=color_u)
    axs[1, 0].legend()
    axs[1, 0].grid()

    # 4. Spacing
    axs[1, 1].plot(df['t'], df['d_fv0_lv'], label='Actual Spacing (m)', color=colors[0])
    if 'd_fv0_lv_noise' in df:
        axs[1, 1].scatter(df['t'], df['d_fv0_lv_noise'], label='Noisy Spacing (m)', color=colors[2], marker='x', s=1)
    axs[1, 1].plot(df['t'], df['d*_fv0_lv'], label='Target Spacing (m)', color=colors[5], linestyle='--')
    axs[1, 1].set_title("Inter-Vehicle Spacing")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Spacing (m)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--')

    plt.tight_layout()
    plt.show()

    return df
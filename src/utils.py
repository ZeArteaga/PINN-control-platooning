import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    u = a*m + Fr + Fa + Fg
    return u

def save_trajectory_plot(data: dict, path: str, traj_id: str) -> pd.DataFrame:
    df = pd.DataFrame(data, dtype=np.float32)
    df.to_csv(path, index=False)
    print(f"Saved CACC trajectory {traj_id} to {path}")

    fig, axs = plt.subplots(2, 2, figsize=(16, 9)) # Adjusted for 3x2 layout
    fig.suptitle(f"CACC Follower/Leader Simulation: {traj_id}", fontsize=14)

    #Follower and Leader Position
    axs[0,0].plot(df['t'], df['fv0_x'], label='Follower Position (m)')
    axs[0,1].plot(df['t'], df['fv0_v_kmh'], label='Follower Velocity (km/h)')
    axs[0,0].plot(df['t'], df['lv_x'], label='Leader Position (m)')
    axs[0,0].set_title("Vehicle Positions")
    axs[0,0].set_xlabel("Time (s)")
    axs[0,0].set_ylabel("Position (m)")
    axs[0,0].legend()
    axs[0,0].grid(True)

    #Follower and Leader Velocity
    axs[0,1].plot(df['t'], df['lv_v'] * 3.6, label='Leader Velocity (km/h)') # Convert m/s to km/h for leader
    axs[0,1].set_title("Vehicle Velocities")
    axs[0,1].set_xlabel("Time (s)")
    axs[0,1].set_ylabel("Velocity (km/h)")
    axs[0,1].legend()
    axs[0,1].grid(True)

    #Follower Acceleration
    color_acc = "blue"
    axs[1,0].plot(df['t'], df['fv0_a'], label='Follower Acceleration (m/s^2)', color=color_acc)
    axs[1,0].set_xlabel("Time (s)")
    axs[1,0].set_ylabel("Acceleration (m/s^2)")
    axs[1,0].tick_params(axis="y", colors=color_acc)

    #Follower Required Input Force
    color_u = "red"
    ax_u = axs[1,0].twinx()
    ax_u.plot(df['t'], df['fv0_u'], label='Follower Input Force (N)', color=color_u)
    ax_u.set_title("Follower Required Input Force to produce acceleration")
    ax_u.set_ylabel("Force (N)")
    axs[1,0].tick_params(axis="y", colors=color_u)
    ax_u.grid(True)

    #Spacing
    axs[1,1].plot(df['t'], df['d_fv0_lv'], label='Actual Spacing (m)')
    axs[1,1].plot(df['t'], df['d*_fv0_lv'], label='Target Spacing (m)', linestyle='--')
    axs[1,1].set_title("Inter-Vehicle Spacing")
    axs[1,1].set_xlabel("Time (s)")
    axs[1,1].set_ylabel("Spacing (m)")
    axs[1,1].legend()
    axs[1,1].grid(True)
    

    plt.tight_layout()
    plt.show()

    return df
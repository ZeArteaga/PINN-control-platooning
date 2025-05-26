import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

def load_driving_cycle(file_path: str, dt: float, b_plot: bool = False) -> tuple:
    """
    Load a driving cycle from a file, interpolate it, and return position and velocity functions.
    
    Args:
        file_path (str): Path to the driving cycle file.
        dt (float): Simulation time step.
        b_plot (bool): Whether to plot the velocity and position.

    Returns:
        tuple: (t_sim, x_interp, v_interp)
            - t_sim: Simulation time points.
            - x_interp: Interpolated position function.
            - v_interp: Interpolated velocity function.
    """
    df = pd.read_csv(file_path, sep="\t", skiprows=2, names=["time", "speed_mph"])
    df["speed_mps"] = df["speed_mph"] * 0.44704  # Convert mph to m/s

    v_interp = interp1d(df["time"], df["speed_mps"], kind="quadratic")

    t_end = df["time"].to_numpy()[-1]
    t_sim = np.arange(0, t_end, dt)
    v_sim = v_interp(t_sim)

    x_sim = cumulative_trapezoid(v_sim, t_sim, initial=0)
    x_interp = interp1d(t_sim, x_sim, kind="quadratic")

    if b_plot:
        plot_driving_cycle(t_sim, v_sim, x_sim)

    return t_sim, x_interp, v_interp

def plot_driving_cycle(t_sim, v_sim, x_sim):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Loaded driving cycle")
    axs[0].plot(t_sim, v_sim, label="Velocity (m/s)", color="blue")
    axs[0].set_title("Interpolated Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(t_sim, x_sim, label="Position (m)", color="green")
    axs[1].set_title("Integrated Position")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Position (m)")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()
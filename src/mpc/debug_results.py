from do_mpc.data import load_results
from plotting import setup_graphics, plot
import matplotlib.pyplot as plt
import argparse
import numpy as np

""" print(fv0.sim.data['_aux', 'e_i'][int(t_end/dt), 0])
print(fv0.mpc.data['_aux', 'e_i'][int(t_end/dt)-1]) """
parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help="The name of the results .pkl file to debug/plot.")
args = parser.parse_args()

file = "results/"+args.filename
results = load_results(file)

mpc_data = results["mpc"]
#sim_data = results["simulator"]

# --- Performance Metrics Calculation ---
spacing_error = mpc_data['_aux', 'e']
control_input_u = mpc_data['_x', 'u']
dt = mpc_data.meta_data['t_step']

# 1. Spacing Error Metrics
initial_error_val = spacing_error[0][0] # Extract scalar from array
min_error = np.min(spacing_error)
max_error = np.max(spacing_error)
average_error = np.mean(spacing_error)

# 2. Overshoot
# Overshoot occurs when the error crosses zero and becomes negative.
overshoot = -min_error if min_error < 0 else 0.0
percent_overshoot = (overshoot / initial_error_val) * 100 if initial_error_val > 0 else 0.0

# 3. Steady-State Error (SSE)
# Calculate SSE over the last 1 second of the simulation
ss_window = int(1 / dt) # Number of samples in the last 1 second
if len(spacing_error) > ss_window:
    steady_state_error = np.mean(spacing_error[-ss_window:])
else:
    steady_state_error = np.mean(spacing_error)

# 4. Settling Time
# Time to enter and stay within a tolerance band (e.g., +/- 0.5m)
settling_tolerance = 0.5  # meters
unsettled_indices = np.where(np.abs(spacing_error) > settling_tolerance)[0]

if len(unsettled_indices) == 0:
    # If error was always within tolerance, settling time is arguably 0
    settling_time = 0.0
else:
    # The last time the error was outside the tolerance band
    last_unsettled_index = unsettled_indices[-1]
    settling_time = (last_unsettled_index + 1) * dt

# 5. Control Effort
rms_control_effort = np.sqrt(np.mean(np.square(control_input_u)))

print("\n--- Performance Metrics ---")
print(f"Initial Spacing Error: {initial_error_val:.2f} m")
print(f"Average Spacing Error: {average_error:.4f} m")
print(f"Maximum Spacing Error: {max_error:.2f} m")
print(f"Minimum Spacing Error: {min_error:.2f} m")
print(f"Overshoot: {overshoot:.2f} m ({percent_overshoot:.2f}%)")
print(f"Steady-State Error (last 1s): {steady_state_error:.4f} m")
print(f"Settling Time (to +/- {settling_tolerance}m): {settling_time:.2f} s")
print(f"Control Effort (RMS of Force): {rms_control_effort:.2f} N")
print("---------------------------\n")

mpc_graphics = setup_graphics(mpc_data)
plot(mpc_graphics, pred_t=25) # for predictions store_full_solution must be set to True

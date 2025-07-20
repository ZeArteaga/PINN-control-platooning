import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models import PinnModel

models_dir = "../models"
cycle_dir = "../data/driving_cycles"

model_files = [f for f in os.listdir(models_dir) if "alpha" in f and f.endswith(".keras")]
# Sort model files by alpha value (ascending data weight)
model_files.sort(key=lambda x: float(x.split("alpha")[1].split("_")[0]))
cycle_files = [f for f in os.listdir(cycle_dir) if f.endswith(".csv")]

punn_file = [f for f in os.listdir(models_dir) if f.startswith("punn_") and f.endswith(".keras")]
punn_model = None
if punn_file:
    punn_model = load_model(os.path.join(models_dir, punn_file[0]), compile=False)

results = []

for cycle_file in cycle_files:
    traj_id = os.path.splitext(cycle_file)[0]
    df = pd.read_csv(os.path.join(cycle_dir, cycle_file))
    X = df[["t", "fv0_u", "fv0_v_noise"]].to_numpy()
    Y = df[["fv0_a"]].to_numpy()

    # Baseline PUNN
    if punn_model:
        pred_a_punn = punn_model.predict(X, verbose=0)
        mse_a_punn = mean_squared_error(Y, pred_a_punn)
        mae_a_punn = mean_absolute_error(Y, pred_a_punn)
        print(f"{traj_id} - PUNN: MSE={mse_a_punn:.3e}, MAE={mae_a_punn:.3e}")
        results.append({
            "model": "PUNN",
            "alpha": None,
            "trajectory": traj_id,
            "mse_a": f"{mse_a_punn:.3e}",
            "mae_a": f"{mae_a_punn:.3e}"
        })

    # PINN models
    for model_file in model_files:
        alpha = model_file.split("alpha")[1].split("_")[0]
        model = load_model(os.path.join(models_dir, model_file), compile=False, 
                           custom_objects={"PinnModel": PinnModel})
        pred_a = model.predict(X, verbose=0)
        mse_a = mean_squared_error(Y, pred_a)
        mae_a = mean_absolute_error(Y, pred_a)
        print(f"{traj_id} - Alpha {alpha}: MSE={mse_a:.3e}, MAE={mae_a:.3e}")
        results.append({
            "model": model_file,
            "alpha": float(alpha),
            "trajectory": traj_id,
            "mse_a": f"{mse_a:.3e}",
            "mae_a": f"{mae_a:.3e}"
        })

results_df = pd.DataFrame(results)

# Plot all MSE curves for all trajectories in one plot
import itertools
markers = itertools.cycle(['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h', 'x'])
colors = itertools.cycle(plt.cm.tab10.colors)


fig, ax = plt.subplots(figsize=(10, 7))
test_trajs = ["us06", "profile_0", "profile_1", "profile_2", "profile_3"]
for traj_id in test_trajs:
    # Find matching trajectory by substring
    matched_trajs = [t for t in results_df["trajectory"].unique() if traj_id in t]
    if not matched_trajs:
        print(f"Trajectory containing '{traj_id}' not found in results.")
        continue
    matched_traj = matched_trajs[0]
    df_pinn = results_df[(results_df["trajectory"] == matched_traj) & (results_df["model"] != "PUNN")].sort_values("alpha")
    df_punn = results_df[(results_df["trajectory"] == matched_traj) & (results_df["model"] == "PUNN")]
    marker = next(markers)
    color = next(colors)
    # Combine PINN points and PUNN at alpha=1
    x = list(df_pinn["alpha"])
    y = list(df_pinn["mse_a"].astype(float))
    if not df_punn.empty:
        punn_alpha = 1
        punn_mse = float(df_punn["mse_a"].iloc[0])
        if punn_alpha not in x:
            x.append(punn_alpha)
            y.append(punn_mse)
    ax.plot(x, y, marker=marker, color=color, label=f"{traj_id}")

ax.set_xlabel("$\\alpha$")
ax.set_ylabel("MSE (log scale)")
ax.set_yscale('log')
ax.set_title("PINN MSE curves for all trajectories (with PUNN baselines)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(models_dir, "all_test_trajectories_mse_vs_alpha.png"))
plt.close()

print("Alpha comparison complete. All MSE curves plotted.")

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
print(results_df)

# Plotting

# Plotting (PUNN as a point, PINN as a curve)
for traj_id in results_df["trajectory"].unique():
    df_traj = results_df[results_df["trajectory"] == traj_id]
    pinn_df = df_traj[df_traj["model"] != "PUNN"].sort_values("alpha")
    punn_df = df_traj[df_traj["model"] == "PUNN"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # MSE plot
    ax1.plot(pinn_df["alpha"], pinn_df["mse_a"].astype(float), marker="o", label="PINN MSE")
    if not punn_df.empty:
        ax1.axhline(float(punn_df["mse_a"].iloc[0]), color="red", linestyle="--", label="PUNN MSE")
        ax1.scatter([1], [float(punn_df["mse_a"].iloc[0])], color="red", label="PUNN MSE (point)")
    ax1.set_xlabel("$\\alpha$")
    ax1.set_ylabel("MSE")
    ax1.set_title(f"MSE")
    ax1.legend()
    ax1.grid(True)

    # MAE plot
    ax2.plot(pinn_df["alpha"], pinn_df["mae_a"].astype(float), marker="o", label="PINN MAE")
    if not punn_df.empty:
        ax2.axhline(float(punn_df["mae_a"].iloc[0]), color="red", linestyle="--", label="PUNN MAE")
        ax2.scatter([1], [float(punn_df["mae_a"].iloc[0])], color="red", label="PUNN MAE (point)")
    ax1.set_xlabel("$\\alpha$")
    ax2.set_ylabel("MAE")
    ax2.set_title(f"MAE")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, f"{traj_id}_metrics_vs_alpha.png"))
    plt.close()

print("Alpha comparison complete.")

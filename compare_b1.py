import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# Initialize ROS 2 Humble type store
typestore = get_typestore(Stores.ROS2_HUMBLE)

def get_bag_data(bag_path):
    topics = ['/robot_gt', '/robot_estimated_odometry']
    data = {t: [] for t in topics}
    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for conn, t, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            p = msg.pose.pose.position
            data[conn.topic].append([t * 1e-9, p.x, p.y])
    return {t: pd.DataFrame(d, columns=['t', 'x', 'y']) for t, d in data.items()}

def compute_stability_metrics(t_series, errors, max_error_threshold=1.0, ignore_initial_s=2.0):
    n = len(errors)
    t_rel = t_series - t_series.iloc[0]
    tail_start = max(int(n * 0.9), n - 10)
    steady = float(np.median(errors[tail_start:]))
    tol = max(0.1 * steady, 0.02)
    mask = t_rel >= ignore_initial_s
    if np.any(mask):
        max_error = float(np.max(errors[mask]))
    else:
        max_error = float(np.max(errors))

    conv_time = np.nan
    for i in range(n):
        if errors[i] <= steady + tol:
            t_i = t_rel.iloc[i]
            j = i
            while j < n and t_rel.iloc[j] <= t_i + 1.0:
                if errors[j] > steady + tol:
                    break
                j += 1
            if j == n or t_rel.iloc[j - 1] >= t_i + 1.0:
                conv_time = float(t_i)
                break

    if np.isnan(conv_time):
        conv_time = float(t_rel.iloc[-1])

    slope = np.polyfit(t_rel, errors, 1)[0] if n >= 2 else 0.0
    err_std = float(np.std(errors))
    early_med = float(np.median(errors[:max(10, n // 10)]))
    late_med = float(np.median(errors[tail_start:]))

    if max_error > max_error_threshold:
        behavior = "Unstable"
    else:
        behavior = "Stable"

    return conv_time, behavior, max_error

# --- Configuration ---
# Match these exactly to your task list
b1_folders = ["bag_B1_P0001", "bag_B1_P001", "bag_B1_P01", "bag_B1_P1", "bag_B1_P"]
noise_values = [0.0001, 0.001, 0.01, 0.1, 1.0]
rmse_values = []
stability_values = []
stability_labels = []
summary = []

# --- Processing ---
plt.figure(figsize=(8, 8)) # Start Trajectory Figure
colors = plt.cm.viridis(np.linspace(0, 1, len(b1_folders)))

for i, folder in enumerate(b1_folders):
    print(f"Processing {folder}...")
    dfs = get_bag_data(folder)
    gt = dfs['/robot_gt']
    est = dfs['/robot_estimated_odometry']

    if gt.empty or est.empty: continue

    # Calculate RMSE
    est_x_interp = np.interp(gt['t'], est['t'], est['x'])
    est_y_interp = np.interp(gt['t'], est['t'], est['y'])
    err = np.sqrt((gt['x'] - est_x_interp)**2 + (gt['y'] - est_y_interp)**2)
    rmse = np.sqrt(np.mean(err**2))
    stability_time, stability_label, max_error = compute_stability_metrics(gt['t'], err)
    rmse_values.append(rmse)
    stability_values.append(stability_time)
    stability_labels.append(stability_label)
    summary.append({
        "process_noise": noise_values[i],
        "rmse": rmse,
        "max_error": max_error,
        "behavior": f"{stability_label} (max error <= 1.0m)" if stability_label == "Stable" else stability_label,
    })

    # Add to Trajectory Plot
    plt.plot(est['x'], est['y'], color=colors[i], label=f"Estimated (Noise {noise_values[i]})", alpha=0.8)

# Finalize Trajectory Plot
plt.plot(gt['x'], gt['y'], 'k-', label='Ground Truth', linewidth=1.0, alpha=0.8)
plt.title("Task B1: Trajectory Comparison (XY Plane)")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.axis('equal')
plt.xlim(-7.5, 7.5)
plt.ylim(-7.5, 7.5)
plt.grid(True, alpha=0.3)
plt.savefig("B1_Trajectories.png", dpi=300)
print("Saved: B1_Trajectories.png")
plt.close()

# --- 1b. SUMMARY TABLE ---
df = pd.DataFrame(summary)
df.to_csv("B1_Summary.csv", index=False)
print("Saved: B1_Summary.csv")

# --- Localization Error Over Time (all noise levels) ---
plt.figure(figsize=(10, 6))
colors_err = plt.cm.cividis(np.linspace(0.1, 0.9, len(b1_folders)))
highlight_color = "#ff2db2"
bright_map = {
    0.1: "#00c6ff",
    0.01: "#ffd400",
    0.001: "#00e676",
}
for i, folder in enumerate(b1_folders):
    dfs = get_bag_data(folder)
    gt = dfs['/robot_gt']
    est = dfs['/robot_estimated_odometry']
    if gt.empty or est.empty:
        continue
    est_x_interp = np.interp(gt['t'], est['t'], est['x'])
    est_y_interp = np.interp(gt['t'], est['t'], est['y'])
    err = np.sqrt((gt['x'] - est_x_interp)**2 + (gt['y'] - est_y_interp)**2)
    rmse = float(np.sqrt(np.mean(err**2)))
    t_rel = gt['t'] - gt['t'].iloc[0]
    color = bright_map.get(noise_values[i], colors_err[i])
    if noise_values[i] == 1.0:
        color = highlight_color
    plt.plot(t_rel, err, color=color, linewidth=1.8,
             label=f"Noise {noise_values[i]} (RMSE {rmse:.4f} m)")

plt.title("Task B1: Localization Error Over Time (Euclidean)")
plt.xlabel("Time [s]")
plt.ylabel("Euclidean Error [m]")
plt.xlim(0, 120)
plt.ylim(0, 2)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("B1_Error_Over_Time.png", dpi=300)
print("Saved: B1_Error_Over_Time.png")
plt.close()

# --- Generate RMSE Trend Plot ---
plt.figure(figsize=(10, 6))
plt.plot(noise_values, rmse_values, 'o-', color='blue', linewidth=2, markersize=10)
plt.xscale('log')
plt.title("Task B1: RMSE vs. Process Noise (Q)")
plt.xlabel("Process Noise Covariance (Log Scale)")
plt.ylabel("RMSE [meters]")
plt.grid(True, which="both", linestyle='--', alpha=0.5)

# Highlight optimal point
opt_idx = np.argmin(rmse_values)
plt.scatter([noise_values[opt_idx]], [rmse_values[opt_idx]], s=140, color='black', zorder=5)
plt.annotate(f'Optimal Q={noise_values[opt_idx]}\nRMSE={rmse_values[opt_idx]:.4f} m',
             xy=(noise_values[opt_idx], rmse_values[opt_idx]),
             xytext=(noise_values[opt_idx] * 1.8, rmse_values[opt_idx] + 0.12),
             textcoords='data',
             arrowprops=dict(arrowstyle='->', color='black', lw=2),
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.9))

plt.savefig("B1_RMSE_Comparison.png", dpi=300)
print("Saved: B1_RMSE_Comparison.png")

# --- Combined RMSE + Stability Plot ---
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(noise_values, rmse_values, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='RMSE')
ax1.set_xscale('log')
ax1.set_xlabel("Process Noise Covariance (Log Scale)")
ax1.set_ylabel("RMSE [meters]", color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.grid(True, which="both", linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(noise_values, stability_values, 's--', color='#d62728', linewidth=2, markersize=8, label='Stability (convergence time)')
ax2.set_ylabel("Convergence Time [s]", color='#d62728')
ax2.tick_params(axis='y', labelcolor='#d62728')

plt.title("Task B1: RMSE and Stability vs. Process Noise (Q)")
lines = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')
plt.savefig("B1_RMSE_Stability.png", dpi=300)
print("Saved: B1_RMSE_Stability.png")

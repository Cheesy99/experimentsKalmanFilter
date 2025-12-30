import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

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

# --- Configuration for B3 scenarios ---
scenarios = [
    ("S1_Baseline", "bag_B3_S1_Baseline"),
    ("S2_LargeCurve", "bag_B3_S2_LargeCurve"),
    ("S3_HighNoise", "bag_B3_S3_HighNoise"),
    ("S4_SmallObs", "bag_B3_S4_SmallObs"),
    ("S5_LargeObs", "bag_B3_S5_LargeObs"),
]

summary = []
stability_labels = []
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# --- 1. TRAJECTORY COMPARISON ---
plt.figure(figsize=(9, 9))
for i, (label, folder) in enumerate(scenarios):
    dfs = get_bag_data(folder)
    gt = dfs['/robot_gt']
    est = dfs['/robot_estimated_odometry']
    if gt.empty or est.empty:
        continue

    est_x_interp = np.interp(gt['t'], est['t'], est['x'])
    est_y_interp = np.interp(gt['t'], est['t'], est['y'])
    errors = np.sqrt((gt['x'] - est_x_interp)**2 + (gt['y'] - est_y_interp)**2)

    rmse = float(np.sqrt(np.mean(errors**2)))
    stability, stability_label, max_error = compute_stability_metrics(gt['t'], errors)
    stability_labels.append(stability_label)

    summary.append({
        "scenario": label,
        "rmse": rmse,
        "max_error": max_error,
        "behavior": f"{stability_label} (max error <= 1.0m)" if stability_label == "Stable" else stability_label,
    })

    plt.plot(est['x'], est['y'], color=colors[i], label=f"{label} (RMSE {rmse:.4f} m)", alpha=0.8)

plt.plot(gt['x'], gt['y'], 'k-', label='Ground Truth', linewidth=2.5)
plt.title("Task B3: Scenario Trajectory Comparison (XY Plane)")
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.legend(loc='upper right', frameon=True)
plt.axis('equal')
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig("B3_Trajectories.png", dpi=300, bbox_inches='tight')
plt.close()

# --- 2. SUMMARY TABLE ---
df = pd.DataFrame(summary).sort_values("scenario")
df.to_csv("B3_Summary.csv", index=False)
print("Saved: B3_Summary.csv")

# --- 3. RMSE BAR PLOT ---
plt.figure(figsize=(7, 4))
plt.bar(df["scenario"], df["rmse"], color=colors)
plt.title("Task B3: RMSE Comparison")
plt.ylabel("Error [m]")
plt.xticks(rotation=35)
plt.tight_layout()
plt.savefig("B3_Metrics_Comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: B3_Metrics_Comparison.png")

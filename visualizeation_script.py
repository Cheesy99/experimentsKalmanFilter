import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

typestore = get_typestore(Stores.ROS2_HUMBLE)

def extract_bag_data(bag_path):
    topics = ['/robot_gt', '/robot_noisy', '/robot_estimated_odometry']
    data = {t: [] for t in topics}
    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for conn, t, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            p = msg.pose.pose.position
            data[conn.topic].append([t * 1e-9, p.x, p.y])
    return {t: pd.DataFrame(d, columns=['t', 'x', 'y']) for t, d in data.items()}

# --- Processing ---
bag_folders = sorted([f for f in os.listdir('.') if f.startswith('bag_')])
summary_data = []
os.makedirs("plots", exist_ok=True)

for folder in bag_folders:
    dfs = extract_bag_data(folder)
    gt = dfs['/robot_gt']
    est = dfs['/robot_estimated_odometry']
    
    if gt.empty or est.empty: continue

    # Sync and calculate Euclidean Error
    est_x_interp = np.interp(gt['t'], est['t'], est['x'])
    est_y_interp = np.interp(gt['t'], est['t'], est['y'])
    
    errors = np.sqrt((gt['x'] - est_x_interp)**2 + (gt['y'] - est_y_interp)**2)
    rmse = np.sqrt(np.mean(errors**2))
    max_error = np.max(errors)
    
    summary_data.append({'experiment': folder, 'rmse': rmse, 'max_error': max_error})

    # ==========================================
    # 1. NEW: EUCLIDEAN ERROR VS TIME PLOT (Page 2 of PDF)
    # ==========================================
    plt.figure(figsize=(10, 4))
    plt.plot(gt['t'] - gt['t'].iloc[0], errors, color='red', linewidth=1.5)
    plt.title(f"Euclidean Position Error: {folder}")
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m]")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/{folder}_error_time.png")
    plt.close()

    # ==========================================
    # 2. TRAJECTORY PLOT (Page 1 of PDF)
    # ==========================================
    plt.figure(figsize=(7, 7))
    plt.plot(gt['x'], gt['y'], 'k-', label='Ground Truth', alpha=0.8)
    plt.plot(dfs['/robot_noisy']['x'], dfs['/robot_noisy']['y'], 'orange', label='Noisy Odom', alpha=0.4)
    plt.plot(est['x'], est['y'], 'g-', label='UKF Estimate', linewidth=2)
    plt.title(f"Trajectory: {folder}\nRMSE: {rmse:.4f}m")
    plt.legend()
    plt.axis('equal')
    plt.savefig(f"plots/{folder}_trajectory.png")
    plt.close()

# --- FINAL SUMMARY TABLES (Page 3 & 7 of PDF) ---
df_sum = pd.DataFrame(summary_data)

def print_section(df, prefix, title):
    print(f"\n--- {title} ---")
    subset = df[df['experiment'].str.contains(prefix)].copy()
    print(subset[['experiment', 'rmse', 'max_error']].to_string(index=False))

print_section(df_sum, "B1", "Task B1: Process Noise Results")
print_section(df_sum, "B2", "Task B2: Measurement Noise Results")
print_section(df_sum, "B3", "Task B3: Scenario Comparison Table")

# Save CSV for B3 specifically for the report
df_sum[df_sum['experiment'].str.contains("B3")].to_csv("plots/Task_B3_Summary.csv")
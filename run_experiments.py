import subprocess
import time
import os
import signal

# --- CONFIGURATION ---
# Path to the landmark file in your current experiments folder
LANDMARKS_PATH = os.path.abspath("landmarks.csv")
SIM_DURATION = 120  # 2 minute per test

# Task Scenarios: (Name, p_xy, p_theta, m_xy, obs_radius, robot_radius, robot_m_noise)
tasks = [
    # --- Task B1: Process Noise Variation ---
    ("B1_P0001", 0.0001, 0.01, 0.1, 15.0, 5.0, 0.01),
    ("B1_P001",  0.001,  0.01, 0.1, 15.0, 5.0, 0.01),
    ("B1_P01",   0.01,   0.01,  0.1, 15.0, 5.0, 0.01), # Baseline
    ("B1_P1",    0.1,    0.01,  0.1, 15.0, 5.0, 0.01),
    ("B1_P",    1.0,    0.01,  0.1, 15.0, 5.0, 0.01),

    # --- Task B2: Measurement Noise Variation ---
    ("B2_M0001",  0.01,   0.01,  0.0001, 15.0, 5.0, 0.01),
    ("B2_M001",   0.01,   0.01,  0.001, 15.0, 5.0, 0.01),
    ("B2_M01",    0.01,   0.01,  0.01,  15.0, 5.0, 0.01),
    ("B2_M1",    0.01,   0.01,  0.1,  15.0, 5.0, 0.01),
    ("B2_M",   0.01,   0.01,  1.0,  15.0, 5.0, 0.01),

    # --- Task B3: Simulation Parameter Variation ---
    ("B3_S1_Baseline",    0.01, 0.01, 0.1, 5.0,  5.0,  0.01),
    ("B3_S2_LargeCurve",  0.01, 0.01, 0.1, 5.0,  10.0, 0.01),
    ("B3_S3_HighNoise",   0.01, 0.01, 0.1, 5.0,  5.0,  0.5), # High robot sensor noise
    ("B3_S4_SmallObs",    0.01, 0.01, 0.1, 2.0,  5.0,  0.01),
    ("B3_S5_LargeObs",    0.01, 0.01, 0.1, 15.0, 5.0,  0.01),
]

def run_test(name, p_xy, p_theta, m_xy, obs_radius, rob_radius, rob_noise):
    print(f"\n>>> STARTING EXPERIMENT: {name}")
    
    # 1. Start fake_robot using its launch file to pass params correctly
    robot_cmd = [
        "ros2", "launch", "fake_robot", "fake_robot.launch.py",
        f"landmarks_file:={LANDMARKS_PATH}",
        f"circle_radius:={rob_radius}",
        f"measurement_noise_variance:={rob_noise}"
    ]
    robot_proc = subprocess.Popen(robot_cmd, preexec_fn=os.setsid)
    
    # 2. Start UKF Positioning Node
    ukf_cmd = [
        "ros2", "launch", "kalman_positioning", "positioning.launch.py",
        f"landmarks_csv_path:={LANDMARKS_PATH}",
        f"process_noise_xy:={p_xy}",
        f"process_noise_theta:={p_theta}",
        f"measurement_noise_xy:={m_xy}",
        f"observation_radius:={obs_radius}"
    ]
    ukf_proc = subprocess.Popen(ukf_cmd, preexec_fn=os.setsid)
    
    # 3. Record topics needed for analysis
    bag_name = f"bag_{name}"
    bag_cmd = [
        "ros2", "bag", "record", 
        "/robot_noisy", "/robot_estimated_odometry", "/robot_gt",
        "-o", bag_name
    ]
    bag_proc = subprocess.Popen(bag_cmd, preexec_fn=os.setsid)
    
    # Wait for the data collection
    time.sleep(SIM_DURATION)
    
    # 4. Shutdown processes cleanly
    print(f">>> FINISHING EXPERIMENT: {name}")
    for proc in [bag_proc, ukf_proc, robot_proc]:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait()

if __name__ == "__main__":
    for task_params in tasks:
        run_test(*task_params)
    print("\n[DONE] All experiments completed. Bags are ready for CSV conversion.")
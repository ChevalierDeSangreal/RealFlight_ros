import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Wedge

# Set font to serif (Times-like font) for all text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Define paths for three different systems
script_dir = os.path.dirname(os.path.abspath(__file__))
realflight_log = os.path.join(os.path.dirname(script_dir), 'test_log')
elastic_log = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'Elastic-Tracker', 'test_log')
visplanner_log = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'visPlanner', 'test_log')

# Data sources configuration
data_sources = {
    'AgileTracker': realflight_log,
    'Elastic-Tracker': elastic_log,
    'visPlanner': visplanner_log
}

# Trajectory types to process
trajectory_types = ['circle', 'typeD', 'type8']

# Function to compute relative position in drone frame
def compute_relative_position(drone_x, drone_y, drone_z, drone_roll, drone_pitch, drone_yaw, 
                              target_x, target_y, target_z):
    """
    Transform target position from world frame to drone body frame.
    Uses full 3D rotation with roll, pitch, yaw (ZYX Euler angles).
    In body frame: x-axis points forward, y-axis points left, z-axis points up
    """
    # Relative position in world frame
    dx_world = target_x - drone_x
    dy_world = target_y - drone_y
    dz_world = target_z - drone_z
    
    # Compute rotation matrix components (ZYX Euler angle convention)
    cos_roll = np.cos(drone_roll)
    sin_roll = np.sin(drone_roll)
    cos_pitch = np.cos(drone_pitch)
    sin_pitch = np.sin(drone_pitch)
    cos_yaw = np.cos(drone_yaw)
    sin_yaw = np.sin(drone_yaw)
    
    # Transform to drone body frame using full 3D rotation matrix
    # This matches the calculation in visualize_tracking.py
    dx_body = (cos_yaw * cos_pitch * dx_world + 
               sin_yaw * cos_pitch * dy_world - 
               sin_pitch * dz_world)
    
    dy_body = ((cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll) * dx_world +
               (sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll) * dy_world +
               cos_pitch * sin_roll * dz_world)
    
    dz_body = ((cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll) * dx_world +
               (sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll) * dy_world +
               cos_pitch * cos_roll * dz_world)
    
    return dx_body, dy_body, dz_body

# Collect all relative positions for all systems and trajectories
print("=== Loading all trajectory data ===")
all_rel_positions = {}

for system_name, log_dir in data_sources.items():
    all_rel_positions[system_name] = {}
    
    for traj_type in trajectory_types:
        csv_file = os.path.join(log_dir, f'{traj_type}.csv')
        
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping {system_name} - {traj_type}")
            continue
        
        # Read data
        data = pd.read_csv(csv_file)
        data = data.dropna()
        data_sorted = data.sort_values('timestamp').reset_index(drop=True)
        
        # Filter: only keep data when target is moving
        target_vx = data_sorted['target_vx'].values
        target_vy = data_sorted['target_vy'].values
        target_vz = data_sorted['target_vz'].values
        target_velocity_magnitude = np.sqrt(target_vx**2 + target_vy**2 + target_vz**2)
        
        # Use velocity threshold to determine if target is moving
        velocity_threshold = 0.05  # m/s
        is_moving = target_velocity_magnitude > velocity_threshold
        
        # Find when target stops moving
        stop_moving_indices = []
        for i in range(1, len(is_moving)):
            if is_moving[i-1] and not is_moving[i]:
                stop_moving_indices.append(i)
        
        # Truncate data: only keep data before target stops moving
        # For type8 trajectory, extend 3 seconds after stopping
        if stop_moving_indices:
            cutoff_index = stop_moving_indices[0]
            
            # For type8 trajectory, add 5 seconds of data after stopping
            if traj_type == 'type8':
                # Get timestamp when target stops
                stop_timestamp = data_sorted.iloc[cutoff_index]['timestamp']
                # Find data within 5 seconds after stopping
                extended_cutoff_index = cutoff_index
                for i in range(cutoff_index, len(data_sorted)):
                    if data_sorted.iloc[i]['timestamp'] - stop_timestamp <= 0.50:
                        extended_cutoff_index = i + 1
                    else:
                        break
                cutoff_index = extended_cutoff_index
                print(f"[{system_name}] {traj_type}: Using {len(data_sorted[:cutoff_index])} points (extended 5s after target stops)")
            else:
                print(f"[{system_name}] {traj_type}: Using {cutoff_index} points (before target stops)")
            
            data_sorted = data_sorted.iloc[:cutoff_index].reset_index(drop=True)
        else:
            print(f"[{system_name}] {traj_type}: Using all {len(data_sorted)} points (target always moving)")
        
        if len(data_sorted) == 0:
            continue
        
        # Extract positions
        drone_x = data_sorted['drone_x'].values
        drone_y = data_sorted['drone_y'].values
        drone_z = data_sorted['drone_z'].values
        drone_roll = data_sorted['drone_roll'].values
        drone_pitch = data_sorted['drone_pitch'].values
        drone_yaw = data_sorted['drone_yaw'].values
        target_x = data_sorted['target_x'].values
        target_y = data_sorted['target_y'].values
        target_z = data_sorted['target_z'].values
        
        # Compute relative positions in drone frame
        rel_x_traj = []
        rel_y_traj = []
        for i in range(len(drone_x)):
            rel_x_i, rel_y_i, rel_z_i = compute_relative_position(
                drone_x[i], drone_y[i], drone_z[i],
                drone_roll[i], drone_pitch[i], drone_yaw[i], 
                target_x[i], target_y[i], target_z[i]
            )
            rel_x_traj.append(rel_x_i)
            rel_y_traj.append(rel_y_i)
        
        # Store trajectory for this type
        all_rel_positions[system_name][traj_type] = {
            'rel_x': np.array(rel_x_traj),
            'rel_y': np.array(rel_y_traj)
        }

# Print statistics
for system_name in all_rel_positions:
    total_points = sum(len(all_rel_positions[system_name][traj]['rel_x']) 
                      for traj in all_rel_positions[system_name])
    print(f"[{system_name}] Total data points: {total_points}")

# Create single figure with 3 subplots
print("\n=== Generating visualization ===")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define colors for different trajectory types
traj_colors = {
    'circle': 'red',
    'typeD': 'green',
    'type8': 'blue'
}

# Plot for each system
for idx, system_name in enumerate(['AgileTracker', 'Elastic-Tracker', 'visPlanner']):
    ax = axes[idx]
    
    if system_name not in all_rel_positions or len(all_rel_positions[system_name]) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        title = 'AgileTracker (Ours)' if system_name == 'AgileTracker' else system_name
        ax.set_title(title)
        continue
    
    # Set up the square plot area
    # Drone at origin (0, 0) in body frame, facing right (positive x direction)
    # Position origin at center-left of the plot by asymmetric axis limits
    drone_x_pos = 0.0  # Drone at origin
    drone_y_pos = 0.0  # Drone at origin
    
    # Asymmetric x-axis: more space in front (positive x) than behind
    # Symmetric y-axis: equal space left and right
    ax.set_xlim(-2.0, 4.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal')
    
    # Plot trajectories for each type
    for traj_type in trajectory_types:
        if traj_type not in all_rel_positions[system_name]:
            continue
        
        rel_x = all_rel_positions[system_name][traj_type]['rel_x']
        rel_y = all_rel_positions[system_name][traj_type]['rel_y']
        
        if len(rel_x) > 0:
            ax.plot(rel_x, rel_y, 
                   color=traj_colors[traj_type], 
                   linewidth=2, 
                   alpha=0.7,
                   label=traj_type)
    
    # Draw FOV sector (120 degrees, 2m radius)
    fov_angle = 120  # degrees
    fov_radius = 2.0  # meters
    
    # Wedge: center, radius, theta1, theta2
    # Drone faces right (0 degrees), so sector is from -60 to +60 degrees
    wedge = Wedge(
        (drone_x_pos, drone_y_pos), 
        fov_radius, 
        -fov_angle/2, 
        fov_angle/2,
        fill=False,
        edgecolor='red',
        linewidth=1.0,
        linestyle='-',
        zorder=5
    )
    ax.add_patch(wedge)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    # Add "(Ours)" suffix for AgileTracker
    title = 'AgileTracker (Ours)' if system_name == 'AgileTracker' else system_name
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    print(f"[{system_name}] Plotted {len(all_rel_positions[system_name])} trajectories")

# Adjust subplot layout
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.92, wspace=0.25)

# Save figure
output_dir = os.path.join(os.path.dirname(script_dir), 'figs')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'target_position_all.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to: {output_path}")

# Show plot
plt.show()

print("\n=== Visualization completed ===")

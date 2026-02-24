import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d
import numpy as np
import os

# Set font to serif (Times-like font) for all text
# Use 'DejaVu Serif' or 'Liberation Serif' as fallback for Linux systems
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Define data paths
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(os.path.dirname(script_dir), 'test_log')

data_files = {
    'Circle': os.path.join(log_dir, 'circle.csv'),
    'Type D': os.path.join(log_dir, 'typeD.csv'),
    'Type 8': os.path.join(log_dir, 'type8.csv')
}

# Create figure with 3 subplots arranged horizontally
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Color map for velocity-based coloring
cmap = plt.cm.jet

# Store min and max velocities across all trajectories for consistent colorbar
all_velocities = []

# First pass: collect all velocities to determine global min/max
# Truncate data to stop 20cm before the end to avoid trajectory closure
processed_data = {}
distance_before_end = 0.2  # meters (20cm)

for title, filepath in data_files.items():
    data = pd.read_csv(filepath)
    data = data.dropna()
    data_sorted = data.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate cumulative trajectory distance from the end
    target_x = data_sorted['target_x'].values
    target_y = data_sorted['target_y'].values
    target_z = data_sorted['target_z'].values
    
    # Calculate distances between consecutive points
    distances = np.sqrt(np.diff(target_x)**2 + np.diff(target_y)**2 + np.diff(target_z)**2)
    
    # Calculate cumulative distance from the end (reversed)
    cumulative_dist_from_end = np.zeros(len(target_x))
    cumulative_dist_from_end[-1] = 0
    for i in range(len(distances) - 1, -1, -1):
        cumulative_dist_from_end[i] = cumulative_dist_from_end[i+1] + distances[i]
    
    # Find the index where we're 20cm before the end
    # We want the last point that is at least 20cm from the end
    cutoff_index = np.where(cumulative_dist_from_end >= distance_before_end)[0]
    if len(cutoff_index) > 0:
        cutoff_index = cutoff_index[-1]  # Take the LAST index that satisfies the condition
    else:
        cutoff_index = len(target_x)  # Use all data if trajectory is too short
    
    # Ensure we have at least 10 points for meaningful visualization
    if cutoff_index < 10:
        cutoff_index = min(10, len(target_x))
    
    data_truncated = data_sorted.iloc[:cutoff_index].reset_index(drop=True)
    
    print(f"[{title}] Using {len(data_truncated)}/{len(data_sorted)} points (stopped 20cm before end)")
    
    # Skip if not enough data points
    if len(data_truncated) < 2:
        print(f"  Warning: Not enough data points for {title}, skipping")
        continue
    
    # Calculate target velocity magnitude from position differences (more reliable)
    target_x = data_truncated['target_x'].values
    target_y = data_truncated['target_y'].values
    target_z = data_truncated['target_z'].values
    timestamps = data_truncated['timestamp'].values
    
    # Calculate velocity using central differences (more accurate)
    velocity_magnitude = np.zeros(len(target_x))
    for i in range(len(target_x)):
        if i == 0:
            # Forward difference for first point
            dt = timestamps[i+1] - timestamps[i]
            dx = target_x[i+1] - target_x[i]
            dy = target_y[i+1] - target_y[i]
            dz = target_z[i+1] - target_z[i]
        elif i == len(target_x) - 1:
            # Backward difference for last point
            dt = timestamps[i] - timestamps[i-1]
            dx = target_x[i] - target_x[i-1]
            dy = target_y[i] - target_y[i-1]
            dz = target_z[i] - target_z[i-1]
        else:
            # Central difference for middle points
            dt = timestamps[i+1] - timestamps[i-1]
            dx = target_x[i+1] - target_x[i-1]
            dy = target_y[i+1] - target_y[i-1]
            dz = target_z[i+1] - target_z[i-1]
        
        velocity_magnitude[i] = np.sqrt(dx**2 + dy**2 + dz**2) / dt
    
    # Apply Gaussian smoothing to velocity for smoother color transitions
    from scipy.ndimage import gaussian_filter1d
    velocity_magnitude = gaussian_filter1d(velocity_magnitude, sigma=3)
    
    # Store processed data
    processed_data[title] = {
        'data': data_truncated,
        'velocity': velocity_magnitude
    }
    
    # Collect velocities for global range
    all_velocities.extend(velocity_magnitude)

# Get global velocity range
if len(all_velocities) == 0:
    print("Error: No valid velocity data collected!")
    exit(1)

v_min = np.min(all_velocities)
v_max = np.max(all_velocities)
print(f"\nGlobal velocity range: {v_min:.4f} - {v_max:.4f} m/s")

# Process each trajectory
for idx, (title, filepath) in enumerate(data_files.items()):
    # Get pre-processed data
    data_sorted = processed_data[title]['data']
    velocity_magnitude = processed_data[title]['velocity']
    
    # Extract target positions (top view: X-Y plane)
    target_x = data_sorted['target_x'].values
    target_y = data_sorted['target_y'].values
    
    # Create line segments for continuous trajectory
    points = np.array([target_x, target_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create LineCollection with velocity-based coloring
    ax = axes[idx]
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(v_min, v_max))
    # Use average of velocities at segment endpoints for smoother transitions
    segment_velocities = (velocity_magnitude[:-1] + velocity_magnitude[1:]) / 2
    lc.set_array(segment_velocities)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{title} Trajectory')
    
    # Set axis limits with padding based on this trajectory's own data
    x_range = target_x.max() - target_x.min()
    y_range = target_y.max() - target_y.min()
    padding = 0.1
    
    # Calculate center and range to maintain square aspect
    mid_x = (target_x.max() + target_x.min()) / 2
    mid_y = (target_y.max() + target_y.min()) / 2
    max_range = max(x_range, y_range) * (1 + padding)
    
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    
    # Set equal aspect ratio with fixed box size
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=0.3)

# Add colorbar on the right side of the figure
# Adjust subplot positions to make room for colorbar and prevent overlap
fig.subplots_adjust(left=0.05, right=0.88, bottom=0.1, top=0.92, wspace=0.25)

# Add colorbar with proper positioning
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(v_min, v_max)), 
                    ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, aspect=30)
cbar.set_label('Velocity (m/s)', rotation=270, labelpad=20)

# Save figure (don't use tight_layout as we manually adjusted)
output_dir = os.path.join(os.path.dirname(script_dir), 'figs')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'target_trajectories.png')
plt.savefig(output_path, dpi=300)
print(f"Trajectory visualization saved to: {output_path}")

# Display the plot
plt.show()

# Print statistics for each trajectory
print("\n=== Target Trajectory Statistics ===")
for title, filepath in data_files.items():
    data = pd.read_csv(filepath)
    data = data.dropna()
    data_sorted = data.sort_values('timestamp').reset_index(drop=True)
    
    target_x = data_sorted['target_x'].values
    target_y = data_sorted['target_y'].values
    target_z = data_sorted['target_z'].values
    
    # Calculate velocity magnitude
    target_vx = data_sorted['target_vx'].values
    target_vy = data_sorted['target_vy'].values
    target_vz = data_sorted['target_vz'].values
    velocity_magnitude = np.sqrt(target_vx**2 + target_vy**2 + target_vz**2)
    
    # Calculate trajectory length
    distances = np.sqrt(np.diff(target_x)**2 + np.diff(target_y)**2 + np.diff(target_z)**2)
    total_length = np.sum(distances)
    
    print(f"\n[{title}]")
    print(f"  Data points: {len(data_sorted)}")
    print(f"  Time duration: {data_sorted['timestamp'].max() - data_sorted['timestamp'].min():.2f} s")
    print(f"  Trajectory length: {total_length:.4f} m")
    print(f"  X range: [{target_x.min():.4f}, {target_x.max():.4f}] m")
    print(f"  Y range: [{target_y.min():.4f}, {target_y.max():.4f}] m")
    print(f"  Z range: [{target_z.min():.4f}, {target_z.max():.4f}] m")
    print(f"  Velocity: min={velocity_magnitude.min():.4f}, max={velocity_magnitude.max():.4f}, mean={velocity_magnitude.mean():.4f} m/s")


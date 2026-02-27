#!/usr/bin/env python3
"""
Visualize neural network thrust output and drone altitude changes under payload variations

Comparison analysis:
1. Payload increase (circle_add_weight.csv)
2. Payload decrease (circle_reduce_weight.csv)

Display the first 10 seconds after target starts moving:
- Neural network thrust output vs time
- Drone altitude vs time
"""

import os
import sys

# Check dependencies
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required Python library - {e}")
    print("\nPlease install dependencies:")
    print("  pip install pandas matplotlib numpy")
    print("or:")
    print("  conda install pandas matplotlib numpy")
    sys.exit(1)

# Set font to serif (Times-like font) for all text - consistent with other scripts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "../test_log")

# Data file paths
ADD_WEIGHT_CSV = os.path.join(LOG_DIR, "circle_add_weight.csv")
REDUCE_WEIGHT_CSV = os.path.join(LOG_DIR, "circle_reduce_weight.csv")

# Output path - save to figs directory
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_process_data(csv_path, max_time=10.0):
    """
    Load and process CSV data
    
    Args:
        csv_path: Path to CSV file
        max_time: Only load data from first max_time seconds
    
    Returns:
        Processed DataFrame
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found - {csv_path}")
        return None
    
    # Read data
    df = pd.read_csv(csv_path)
    
    # Filter out NaN values
    df = df.dropna()
    
    # Keep only first max_time seconds of data
    df = df[df['timestamp'] <= max_time].copy()
    
    # Ensure data is sorted by time
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # If thrust_output column doesn't exist, estimate from vertical acceleration
    if 'thrust_output' not in df.columns:
        print(f"  Note: No thrust_output column in CSV, estimating from vertical acceleration")
        # Calculate vertical acceleration (numerical differentiation)
        df['vz_diff'] = df['drone_vz'].diff() / df['timestamp'].diff()
        # Smoothing
        df['vz_diff'] = df['vz_diff'].rolling(window=5, center=True).mean()
        
        # Thrust estimation: raw output in [-1, 1] range (tanh output)
        # Baseline is 2*0.76 - 1 ≈ 0.52 (hover thrust in raw output space)
        g = 9.81
        hover_thrust_raw = 2.0 * 0.76 - 1.0  # Convert hover thrust [0,1] back to [-1,1]
        df['thrust_output'] = hover_thrust_raw + (df['vz_diff'] / g) * 2.0
        # Limit range to [-1, 1]
        df['thrust_output'] = df['thrust_output'].clip(-1.0, 1.0)
        # Fill NaN values
        df['thrust_output'] = df['thrust_output'].fillna(hover_thrust_raw)
    
    print(f"Loaded {csv_path}")
    print(f"  Data points: {len(df)}")
    print(f"  Time range: {df['timestamp'].min():.2f}s - {df['timestamp'].max():.2f}s")
    
    return df

def plot_comparison(df_add, df_reduce):
    """
    Plot comparison: thrust output and altitude changes
    
    Args:
        df_add: Data with payload increase
        df_reduce: Data with payload decrease
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Color configuration
    color_add = '#d62728'      # Red - payload increase
    color_reduce = '#2ca02c'   # Green - payload decrease
    color_target = '#7f7f7f'   # Gray - target altitude
    
    # ==================== Subplot 1: Thrust Output ====================
    ax1.plot(df_add['timestamp'], df_add['thrust_output'], 
             color=color_add, linewidth=2, label='Payload Increase (+20%)', alpha=0.8)
    ax1.plot(df_reduce['timestamp'], df_reduce['thrust_output'], 
             color=color_reduce, linewidth=2, label='Payload Decrease (-20%)', alpha=0.8)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Thrust Output (Raw Network Output [-1, 1])', fontsize=12)
    ax1.set_title('Neural Network Thrust Output', fontsize=14, pad=10)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-1.0, 1.0)  # Set y-axis limits to match raw output range
    
    # ==================== Subplot 2: Altitude Changes ====================
    # Add shaded region for unpunished altitude range (1.2 ± 0.3)
    ax2.axhspan(1.2 - 0.3, 1.2 + 0.3, alpha=0.15, color='gray', 
                label='Unpunished Altitude Range (1.2±0.3 m)')
    
    ax2.plot(df_add['timestamp'], df_add['drone_z'], 
             color=color_add, linewidth=2, label='Drone Altitude (Payload Increase)', alpha=0.8)
    ax2.plot(df_reduce['timestamp'], df_reduce['drone_z'], 
             color=color_reduce, linewidth=2, label='Drone Altitude (Payload Decrease)', alpha=0.8)
    
    # Plot target altitude
    ax2.plot(df_add['timestamp'], df_add['target_z'], 
             color=color_target, linewidth=2, linestyle='--', 
             label='Target Altitude', alpha=0.7)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Altitude (m)', fontsize=12)
    ax2.set_title('Drone Altitude', fontsize=14, pad=10)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "weight_change_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_path}")
    
    plt.show()

def plot_thrust_response(df_add, df_reduce):
    """
    Plot detailed thrust response analysis
    
    Args:
        df_add: Data with payload increase
        df_reduce: Data with payload decrease
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    color_add = '#d62728'
    color_reduce = '#2ca02c'
    
    # ==================== Subplot 1: Thrust Output Time Series ====================
    ax = axes[0, 0]
    ax.plot(df_add['timestamp'], df_add['thrust_output'], 
            color=color_add, linewidth=2, label='Payload Increase', alpha=0.8)
    ax.plot(df_reduce['timestamp'], df_reduce['thrust_output'], 
            color=color_reduce, linewidth=2, label='Payload Decrease', alpha=0.8)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Thrust Output (Raw [-1, 1])', fontsize=11)
    ax.set_title('Thrust Output Time Series', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.0, 1.0)
    
    # ==================== Subplot 2: Thrust Output Distribution ====================
    ax = axes[0, 1]
    ax.hist(df_add['thrust_output'], bins=30, color=color_add, alpha=0.6, 
            label='Payload Increase', density=True)
    ax.hist(df_reduce['thrust_output'], bins=30, color=color_reduce, alpha=0.6, 
            label='Payload Decrease', density=True)
    ax.set_xlabel('Thrust Output (Raw [-1, 1])', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Thrust Output Distribution', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ==================== Subplot 3: Altitude Error ====================
    ax = axes[1, 0]
    # Add shaded region for unpunished altitude range
    ax.axhspan(-0.3, 0.3, alpha=0.15, color='gray', 
               label='Unpunished Error Range (±0.3 m)')
    
    height_error_add = df_add['drone_z'] - df_add['target_z']
    height_error_reduce = df_reduce['drone_z'] - df_reduce['target_z']
    
    ax.plot(df_add['timestamp'], height_error_add, 
            color=color_add, linewidth=2, label='Payload Increase', alpha=0.8)
    ax.plot(df_reduce['timestamp'], height_error_reduce, 
            color=color_reduce, linewidth=2, label='Payload Decrease', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Altitude Error (m)', fontsize=11)
    ax.set_title('Altitude Tracking Error (Drone - Target)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    # ==================== Subplot 4: Vertical Velocity ====================
    ax = axes[1, 1]
    ax.plot(df_add['timestamp'], df_add['drone_vz'], 
            color=color_add, linewidth=2, label='Drone (Payload Increase)', alpha=0.8)
    ax.plot(df_reduce['timestamp'], df_reduce['drone_vz'], 
            color=color_reduce, linewidth=2, label='Drone (Payload Decrease)', alpha=0.8)
    ax.plot(df_add['timestamp'], df_add['target_vz'], 
            color='gray', linewidth=2, linestyle='--', label='Target', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Vertical Velocity (m/s)', fontsize=11)
    ax.set_title('Vertical Velocity Comparison', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "thrust_response_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    
    plt.show()

def main():
    """Main function"""
    print("=" * 60)
    print("Payload Variation Robustness Visualization Analysis")
    print("=" * 60)
    
    # Load data (first 10 seconds only)
    print("\nLoading data...")
    df_add = load_and_process_data(ADD_WEIGHT_CSV, max_time=10.0)
    df_reduce = load_and_process_data(REDUCE_WEIGHT_CSV, max_time=10.0)
    
    if df_add is None or df_reduce is None:
        print("\nError: Unable to load data files")
        return
    
    # Check data quality
    has_direct_thrust = 'thrust_output' in pd.read_csv(ADD_WEIGHT_CSV, nrows=1).columns
    if not has_direct_thrust:
        print("\n⚠  Note: CSV files do not contain direct thrust_output data")
        print("   Will estimate thrust requirement from vertical acceleration (as fallback)")
        print("\nTo record actual neural network thrust output, please:")
        print("1. Ensure track_test_node publishes /neural_network/thrust_output topic")
        print("2. Ensure tracking_visualizer subscribes and records this topic")
        print("3. Re-run experiments\n")
    
    # Print data summary
    print("\n" + "=" * 60)
    print("Data Summary:")
    print("=" * 60)
    
    print(f"\nPayload Increase (circle_add_weight.csv):")
    print(f"  Thrust Output: Mean={df_add['thrust_output'].mean():.4f}, "
          f"Std={df_add['thrust_output'].std():.4f}, "
          f"Range=[{df_add['thrust_output'].min():.4f}, {df_add['thrust_output'].max():.4f}]")
    print(f"  Altitude Error: Mean={np.abs(df_add['drone_z'] - df_add['target_z']).mean():.4f}m, "
          f"Max={np.abs(df_add['drone_z'] - df_add['target_z']).max():.4f}m")
    
    print(f"\nPayload Decrease (circle_reduce_weight.csv):")
    print(f"  Thrust Output: Mean={df_reduce['thrust_output'].mean():.4f}, "
          f"Std={df_reduce['thrust_output'].std():.4f}, "
          f"Range=[{df_reduce['thrust_output'].min():.4f}, {df_reduce['thrust_output'].max():.4f}]")
    print(f"  Altitude Error: Mean={np.abs(df_reduce['drone_z'] - df_reduce['target_z']).mean():.4f}m, "
          f"Max={np.abs(df_reduce['drone_z'] - df_reduce['target_z']).max():.4f}m")
    
    # Plot comparison
    print("\nPlotting comparison...")
    plot_comparison(df_add, df_reduce)
    
    # Plot detailed analysis
    print("\nPlotting detailed analysis...")
    plot_thrust_response(df_add, df_reduce)
    
    print("\n" + "=" * 60)
    print("✓ Visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

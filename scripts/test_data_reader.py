#!/usr/bin/env python3
"""
数据分析脚本：对比三种系统在三种轨迹上的性能指标
生成适合论文使用的统计表格
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ====== 配置路径 ======
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
parent_dir = os.path.dirname(project_root)

# 三种系统的日志路径
DATA_SOURCES = {
    'AgileTracker': os.path.join(project_root, 'test_log'),
    'Elastic-Tracker': os.path.join(parent_dir, 'Elastic-Tracker', 'test_log'),
    'visPlanner': os.path.join(parent_dir, 'visPlanner', 'test_log'),
}

# 三种轨迹类型
TRAJECTORY_TYPES = ['circle', 'typeD', 'type8']

TRAJECTORY_NAMES = {
    'circle': 'Circle',
    'typeD': 'Type-D',
    'type8': 'Figure-8',
}


def compute_relative_position(drone_x, drone_y, drone_z, drone_roll, drone_pitch, drone_yaw, 
                              target_x, target_y, target_z):
    """
    Transform target position from world frame to drone body frame.
    Uses full 3D rotation with roll, pitch, yaw (ZYX Euler angles).
    """
    dx_world = target_x - drone_x
    dy_world = target_y - drone_y
    dz_world = target_z - drone_z
    
    cos_roll = np.cos(drone_roll)
    sin_roll = np.sin(drone_roll)
    cos_pitch = np.cos(drone_pitch)
    sin_pitch = np.sin(drone_pitch)
    cos_yaw = np.cos(drone_yaw)
    sin_yaw = np.sin(drone_yaw)
    
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


def central_diff_velocity_2d(x: np.ndarray, y: np.ndarray, ts: np.ndarray, sigma=3):
    """
    对位置序列 (x, y) 用中心差分求水平速度分量，
    首末点分别用前向/后向差分，结果经高斯平滑。
    """
    n = len(x)
    vx = np.empty(n)
    vy = np.empty(n)

    for i in range(n):
        if i == 0:
            dt = ts[1] - ts[0]
            vx[i] = (x[1] - x[0]) / dt
            vy[i] = (y[1] - y[0]) / dt
        elif i == n - 1:
            dt = ts[i] - ts[i - 1]
            vx[i] = (x[i] - x[i - 1]) / dt
            vy[i] = (y[i] - y[i - 1]) / dt
        else:
            dt = ts[i + 1] - ts[i - 1]
            vx[i] = (x[i + 1] - x[i - 1]) / dt
            vy[i] = (y[i + 1] - y[i - 1]) / dt

    vx = gaussian_filter1d(vx, sigma=sigma)
    vy = gaussian_filter1d(vy, sigma=sigma)
    return vx, vy


def calculate_viewing_angles(rel_x, rel_y, rel_z):
    """
    计算目标相对无人机的视角
    返回：水平视角、垂直视角、总视角（度）
    """
    distance = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
    horizontal_distance = np.sqrt(rel_x**2 + rel_y**2)
    
    horizontal_angle = np.degrees(np.arctan2(rel_y, rel_x))
    vertical_angle = np.degrees(np.arctan2(rel_z, horizontal_distance))
    
    if distance > 0:
        cos_angle = rel_x / distance
        total_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    else:
        total_angle = 0
    
    return horizontal_angle, vertical_angle, total_angle


def analyze_csv(csv_path, system_name, traj_type):
    """
    分析单个CSV文件，计算各种统计指标
    """
    if not os.path.exists(csv_path):
        return None
    
    # 读取数据
    data = pd.read_csv(csv_path)
    data = data.dropna()
    data_sorted = data.sort_values('timestamp').reset_index(drop=True)
    
    if len(data_sorted) < 3:
        return None
    
    # 截断数据：只保留目标运动阶段（参考 vis_target_position.py）
    target_vx = data_sorted['target_vx'].values
    target_vy = data_sorted['target_vy'].values
    target_vz = data_sorted['target_vz'].values
    target_velocity_magnitude = np.sqrt(target_vx**2 + target_vy**2 + target_vz**2)
    
    velocity_threshold = 0.05  # m/s
    is_moving = target_velocity_magnitude > velocity_threshold
    
    # 找到目标停止运动的时刻
    stop_moving_indices = []
    for i in range(1, len(is_moving)):
        if is_moving[i-1] and not is_moving[i]:
            stop_moving_indices.append(i)
    
    # 截断数据
    if stop_moving_indices:
        cutoff_index = stop_moving_indices[0]
        
        # type8 轨迹特殊处理：延长0.5秒（参考 vis_target_position.py）
        if traj_type == 'type8':
            stop_timestamp = data_sorted.iloc[cutoff_index]['timestamp']
            extended_cutoff_index = cutoff_index
            for i in range(cutoff_index, len(data_sorted)):
                if data_sorted.iloc[i]['timestamp'] - stop_timestamp <= 0.50:
                    extended_cutoff_index = i + 1
                else:
                    break
            cutoff_index = extended_cutoff_index
        
        data_sorted = data_sorted.iloc[:cutoff_index].reset_index(drop=True)
    
    if len(data_sorted) < 3:
        return None
    
    # 提取数据
    timestamps = data_sorted['timestamp'].values
    
    drone_x = data_sorted['drone_x'].values
    drone_y = data_sorted['drone_y'].values
    drone_z = data_sorted['drone_z'].values
    drone_roll = data_sorted['drone_roll'].values
    drone_pitch = data_sorted['drone_pitch'].values
    drone_yaw = data_sorted['drone_yaw'].values
    
    target_x = data_sorted['target_x'].values
    target_y = data_sorted['target_y'].values
    target_z = data_sorted['target_z'].values
    
    # 1. 计算距离统计
    distances = np.sqrt((target_x - drone_x)**2 + 
                       (target_y - drone_y)**2 + 
                       (target_z - drone_z)**2)
    
    # 2. 计算相对速度统计
    drone_vx, drone_vy = central_diff_velocity_2d(drone_x, drone_y, timestamps)
    target_vx, target_vy = central_diff_velocity_2d(target_x, target_y, timestamps)
    relative_velocity = np.sqrt((drone_vx - target_vx)**2 + (drone_vy - target_vy)**2)
    
    # 3. 计算视角差统计
    horizontal_angles = []
    vertical_angles = []
    total_angles = []
    
    for i in range(len(drone_x)):
        rel_x, rel_y, rel_z = compute_relative_position(
            drone_x[i], drone_y[i], drone_z[i],
            drone_roll[i], drone_pitch[i], drone_yaw[i], 
            target_x[i], target_y[i], target_z[i]
        )
        
        h_angle, v_angle, t_angle = calculate_viewing_angles(rel_x, rel_y, rel_z)
        horizontal_angles.append(np.abs(h_angle))
        vertical_angles.append(np.abs(v_angle))
        total_angles.append(t_angle)
    
    horizontal_angles = np.array(horizontal_angles)
    vertical_angles = np.array(vertical_angles)
    total_angles = np.array(total_angles)
    
    # 编译统计结果（删除了所有视角的最小值）
    stats = {
        'system': system_name,
        'trajectory': traj_type,
        'data_points': len(data_sorted),
        'time_duration': timestamps[-1] - timestamps[0],
        
        'distance_max': np.max(distances),
        'distance_min': np.min(distances),
        'distance_mean': np.mean(distances),
        'distance_std': np.std(distances),
        
        'relative_velocity_max': np.max(relative_velocity),
        'relative_velocity_min': np.min(relative_velocity),
        'relative_velocity_mean': np.mean(relative_velocity),
        'relative_velocity_std': np.std(relative_velocity),
        
        'total_angle_max': np.max(total_angles),
        'total_angle_mean': np.mean(total_angles),
        'total_angle_std': np.std(total_angles),
        
        'horizontal_angle_max': np.max(horizontal_angles),
        'horizontal_angle_mean': np.mean(horizontal_angles),
        'horizontal_angle_std': np.std(horizontal_angles),
        
        'vertical_angle_max': np.max(vertical_angles),
        'vertical_angle_mean': np.mean(vertical_angles),
        'vertical_angle_std': np.std(vertical_angles),
    }
    
    return stats


def print_table_distance(all_stats):
    """表1：距离统计对比表"""
    print("\n" + "="*100)
    print("表1：目标-无人机距离统计 (单位：米)")
    print("="*100)
    
    # 表头
    header = f"{'System':<18}"
    for traj in TRAJECTORY_TYPES:
        header += f"│ {TRAJECTORY_NAMES[traj]:<28}"
    print(header)
    print("─" * 100)
    
    # 每个系统的数据
    for system in ['AgileTracker', 'Elastic-Tracker', 'visPlanner']:
        # 最大距离行
        row_max = f"{system:<18}"
        for traj in TRAJECTORY_TYPES:
            key = (system, traj)
            if key in all_stats:
                val = all_stats[key]['distance_max']
                row_max += f"│ Max: {val:6.3f}               "
            else:
                row_max += f"│ Max: {'N/A':<6}               "
        print(row_max)
        
        # 最小距离行
        row_min = f"{'':<18}"
        for traj in TRAJECTORY_TYPES:
            key = (system, traj)
            if key in all_stats:
                val = all_stats[key]['distance_min']
                row_min += f"│ Min: {val:6.3f}               "
            else:
                row_min += f"│ Min: {'N/A':<6}               "
        print(row_min)
        
        # 平均距离行
        row_mean = f"{'':<18}"
        for traj in TRAJECTORY_TYPES:
            key = (system, traj)
            if key in all_stats:
                val = all_stats[key]['distance_mean']
                std = all_stats[key]['distance_std']
                row_mean += f"│ Mean: {val:5.3f} ± {std:5.3f}     "
            else:
                row_mean += f"│ Mean: {'N/A':<6}               "
        print(row_mean)
        
        if system != 'visPlanner':
            print("─" * 100)
    
    print("="*100)


def print_table_velocity(all_stats):
    """表2：相对速度统计对比表"""
    print("\n" + "="*100)
    print("表2：相对速度统计 (单位：米/秒)")
    print("="*100)
    
    # 表头
    header = f"{'System':<18}"
    for traj in TRAJECTORY_TYPES:
        header += f"│ {TRAJECTORY_NAMES[traj]:<28}"
    print(header)
    print("─" * 100)
    
    # 每个系统的数据
    for system in ['AgileTracker', 'Elastic-Tracker', 'visPlanner']:
        # 最大速度行
        row_max = f"{system:<18}"
        for traj in TRAJECTORY_TYPES:
            key = (system, traj)
            if key in all_stats:
                val = all_stats[key]['relative_velocity_max']
                row_max += f"│ Max: {val:6.3f}               "
            else:
                row_max += f"│ Max: {'N/A':<6}               "
        print(row_max)
        
        # 最小速度行
        row_min = f"{'':<18}"
        for traj in TRAJECTORY_TYPES:
            key = (system, traj)
            if key in all_stats:
                val = all_stats[key]['relative_velocity_min']
                row_min += f"│ Min: {val:6.3f}               "
            else:
                row_min += f"│ Min: {'N/A':<6}               "
        print(row_min)
        
        # 平均速度行
        row_mean = f"{'':<18}"
        for traj in TRAJECTORY_TYPES:
            key = (system, traj)
            if key in all_stats:
                val = all_stats[key]['relative_velocity_mean']
                std = all_stats[key]['relative_velocity_std']
                row_mean += f"│ Mean: {val:5.3f} ± {std:5.3f}     "
            else:
                row_mean += f"│ Mean: {'N/A':<6}               "
        print(row_mean)
        
        if system != 'visPlanner':
            print("─" * 100)
    
    print("="*100)


def print_table_angles(all_stats):
    """表3：视角统计对比表（删除了最小值）"""
    print("\n" + "="*100)
    print("表3：视角差统计 (单位：度)")
    print("="*100)
    
    # 总视角差
    print("\n【总视角差 (Total Viewing Angle Deviation)】")
    header = f"{'System':<18}"
    for traj in TRAJECTORY_TYPES:
        header += f"│ {TRAJECTORY_NAMES[traj]:<28}"
    print(header)
    print("─" * 100)
    
    for system in ['AgileTracker', 'Elastic-Tracker', 'visPlanner']:
        row_max = f"{system:<18}"
        for traj in TRAJECTORY_TYPES:
            key = (system, traj)
            if key in all_stats:
                val = all_stats[key]['total_angle_max']
                row_max += f"│ Max: {val:6.2f}°              "
            else:
                row_max += f"│ Max: {'N/A':<6}               "
        print(row_max)
        
        row_mean = f"{'':<18}"
        for traj in TRAJECTORY_TYPES:
            key = (system, traj)
            if key in all_stats:
                val = all_stats[key]['total_angle_mean']
                std = all_stats[key]['total_angle_std']
                row_mean += f"│ Mean: {val:5.2f}° ± {std:5.2f}°   "
            else:
                row_mean += f"│ Mean: {'N/A':<6}               "
        print(row_mean)
        
        if system != 'visPlanner':
            print("─" * 100)
    
    # 水平视角差
    print("\n【水平视角差 (Horizontal Viewing Angle Deviation)】")
    header = f"{'System':<18}"
    for traj in TRAJECTORY_TYPES:
        header += f"│ {TRAJECTORY_NAMES[traj]:<28}"
    print(header)
    print("─" * 100)
    
    for system in ['AgileTracker', 'Elastic-Tracker', 'visPlanner']:
        row_max = f"{system:<18}"
        for traj in TRAJECTORY_TYPES:
            key = (system, traj)
            if key in all_stats:
                val = all_stats[key]['horizontal_angle_max']
                row_max += f"│ Max: {val:6.2f}°              "
            else:
                row_max += f"│ Max: {'N/A':<6}               "
        print(row_max)
        
        row_mean = f"{'':<18}"
        for traj in TRAJECTORY_TYPES:
            key = (system, traj)
            if key in all_stats:
                val = all_stats[key]['horizontal_angle_mean']
                std = all_stats[key]['horizontal_angle_std']
                row_mean += f"│ Mean: {val:5.2f}° ± {std:5.2f}°   "
            else:
                row_mean += f"│ Mean: {'N/A':<6}               "
        print(row_mean)
        
        if system != 'visPlanner':
            print("─" * 100)
    
    # 垂直视角差
    print("\n【垂直视角差 (Vertical Viewing Angle Deviation)】")
    header = f"{'System':<18}"
    for traj in TRAJECTORY_TYPES:
        header += f"│ {TRAJECTORY_NAMES[traj]:<28}"
    print(header)
    print("─" * 100)
    
    for system in ['AgileTracker', 'Elastic-Tracker', 'visPlanner']:
        row_max = f"{system:<18}"
        for traj in TRAJECTORY_TYPES:
            key = (system, traj)
            if key in all_stats:
                val = all_stats[key]['vertical_angle_max']
                row_max += f"│ Max: {val:6.2f}°              "
            else:
                row_max += f"│ Max: {'N/A':<6}               "
        print(row_max)
        
        row_mean = f"{'':<18}"
        for traj in TRAJECTORY_TYPES:
            key = (system, traj)
            if key in all_stats:
                val = all_stats[key]['vertical_angle_mean']
                std = all_stats[key]['vertical_angle_std']
                row_mean += f"│ Mean: {val:5.2f}° ± {std:5.2f}°   "
            else:
                row_mean += f"│ Mean: {'N/A':<6}               "
        print(row_mean)
        
        if system != 'visPlanner':
            print("─" * 100)
    
    print("="*100)


def print_summary_table(all_stats):
    """表4：紧凑汇总表（只显示平均值）"""
    print("\n" + "="*100)
    print("表4：性能指标汇总 (Mean ± Std)")
    print("="*100)
    
    metrics = [
        ('距离 (m)', 'distance_mean', 'distance_std'),
        ('相对速度 (m/s)', 'relative_velocity_mean', 'relative_velocity_std'),
        ('总视角差 (°)', 'total_angle_mean', 'total_angle_std'),
        ('水平视角差 (°)', 'horizontal_angle_mean', 'horizontal_angle_std'),
        ('垂直视角差 (°)', 'vertical_angle_mean', 'vertical_angle_std'),
    ]
    
    for metric_name, mean_key, std_key in metrics:
        print(f"\n【{metric_name}】")
        header = f"{'System':<18}"
        for traj in TRAJECTORY_TYPES:
            header += f"│ {TRAJECTORY_NAMES[traj]:<20}"
        print(header)
        print("─" * 100)
        
        for system in ['AgileTracker', 'Elastic-Tracker', 'visPlanner']:
            row = f"{system:<18}"
            for traj in TRAJECTORY_TYPES:
                key = (system, traj)
                if key in all_stats:
                    mean_val = all_stats[key][mean_key]
                    std_val = all_stats[key][std_key]
                    if '°' in metric_name:
                        row += f"│ {mean_val:5.2f} ± {std_val:5.2f}      "
                    else:
                        row += f"│ {mean_val:5.3f} ± {std_val:5.3f}     "
                else:
                    row += f"│ {'N/A':<20}"
            print(row)
    
    print("="*100)


def main():
    """主函数"""
    print("="*100)
    print(" 数据分析：三种系统在三种轨迹上的性能指标对比")
    print("="*100)
    
    # 收集所有数据
    all_stats = {}
    
    print("\n正在加载数据...")
    for system_name, log_dir in DATA_SOURCES.items():
        for traj_type in TRAJECTORY_TYPES:
            csv_file = os.path.join(log_dir, f'{traj_type}.csv')
            
            if not os.path.exists(csv_file):
                print(f"  ⚠ 文件不存在: {csv_file}")
                continue
            
            stats = analyze_csv(csv_file, system_name, traj_type)
            if stats:
                all_stats[(system_name, traj_type)] = stats
                print(f"  ✓ {system_name:>16} - {traj_type:<7}: {stats['data_points']:4d} 点, {stats['time_duration']:5.1f}s")
            else:
                print(f"  ✗ {system_name:>16} - {traj_type:<7}: 分析失败")
    
    if not all_stats:
        print("\n错误：没有可用的数据")
        return
    
    # 生成所有表格
    print_table_distance(all_stats)
    print_table_velocity(all_stats)
    print_table_angles(all_stats)
    print_summary_table(all_stats)
    
    print("\n" + "="*100)
    print(" 分析完成")
    print("="*100)


if __name__ == '__main__':
    main()

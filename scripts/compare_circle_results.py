#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比有速度预测和无速度预测的性能指标
展示使用速度预测的优越性
支持：circle, typeD, type8
"""

import os
import sys
import numpy as np
import pandas as pd


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


def moving_average(data, window=5):
    """简单移动平均平滑"""
    if len(data) < window:
        return data
    result = np.copy(data)
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        result[i] = np.mean(data[start:end])
    return result


def central_diff_velocity_2d(x: np.ndarray, y: np.ndarray, ts: np.ndarray, window=5):
    """
    对位置序列 (x, y) 用中心差分求水平速度分量，
    首末点分别用前向/后向差分，结果经移动平均平滑。
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

    vx = moving_average(vx, window)
    vy = moving_average(vy, window)
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


def truncate_data_by_velocity(data_sorted, traj_type):
    """
    根据目标速度截取数据（参考 vis_target_position.py）
    - 截取到目标停止运动的时刻
    - type8 轨迹停止后额外保留 0.5 秒
    """
    target_vx = data_sorted['target_vx'].values
    target_vy = data_sorted['target_vy'].values
    target_vz = data_sorted['target_vz'].values
    target_velocity_magnitude = np.sqrt(target_vx**2 + target_vy**2 + target_vz**2)
    
    # 速度阈值判断目标是否在运动
    velocity_threshold = 0.05  # m/s
    is_moving = target_velocity_magnitude > velocity_threshold
    
    # 找到目标停止运动的时刻
    stop_moving_indices = []
    for i in range(1, len(is_moving)):
        if is_moving[i-1] and not is_moving[i]:
            stop_moving_indices.append(i)
    
    # 截取数据：只保留目标运动期间的数据
    if stop_moving_indices:
        cutoff_index = stop_moving_indices[0]
        
        # type8 轨迹特殊处理：停止后延长0.5秒
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
    
    return data_sorted


def analyze_csv(csv_path, method_name, traj_type):
    """
    分析单个CSV文件，计算各种统计指标
    使用 vis_target_position.py 中相同的时间片段截取逻辑
    """
    if not os.path.exists(csv_path):
        return None
    
    # 读取数据
    data = pd.read_csv(csv_path)
    data = data.dropna()
    data_sorted = data.sort_values('timestamp').reset_index(drop=True)
    
    if len(data_sorted) < 3:
        return None
    
    # 使用相同的时间片段截取逻辑
    data_sorted = truncate_data_by_velocity(data_sorted, traj_type)
    
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
    
    # 编译统计结果
    stats = {
        'method': method_name,
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


def print_comparison_table(stats_without, stats_with, traj_name):
    """打印对比表格"""
    
    print("\n" + "="*90)
    print(f" {traj_name} 轨迹：无速度预测 vs 有速度预测性能对比")
    print("="*90)
    
    # 表1: 距离统计
    print("\n【表1：目标-无人机距离统计 (单位：米)】")
    print("─"*90)
    print(f"{'指标':<20} │ {'无速度预测':<30} │ {'有速度预测':<30}")
    print("─"*90)
    print(f"{'最大距离':<20} │ {stats_without['distance_max']:6.3f}{'':<24} │ {stats_with['distance_max']:6.3f}")
    print(f"{'最小距离':<20} │ {stats_without['distance_min']:6.3f}{'':<24} │ {stats_with['distance_min']:6.3f}")
    print(f"{'平均距离 ± 标准差':<20} │ {stats_without['distance_mean']:6.3f} ± {stats_without['distance_std']:6.3f}{'':<13} │ {stats_with['distance_mean']:6.3f} ± {stats_with['distance_std']:6.3f}")
    
    # 计算改进
    dist_improve = (stats_without['distance_mean'] - stats_with['distance_mean']) / stats_without['distance_mean'] * 100
    print(f"\n  💡 平均距离改进: {dist_improve:+.2f}%")
    
    # 表2: 相对速度统计
    print("\n【表2：相对速度统计 (单位：米/秒)】")
    print("─"*90)
    print(f"{'指标':<20} │ {'无速度预测':<30} │ {'有速度预测':<30}")
    print("─"*90)
    print(f"{'最大相对速度':<20} │ {stats_without['relative_velocity_max']:6.3f}{'':<24} │ {stats_with['relative_velocity_max']:6.3f}")
    print(f"{'最小相对速度':<20} │ {stats_without['relative_velocity_min']:6.3f}{'':<24} │ {stats_with['relative_velocity_min']:6.3f}")
    print(f"{'平均速度 ± 标准差':<20} │ {stats_without['relative_velocity_mean']:6.3f} ± {stats_without['relative_velocity_std']:6.3f}{'':<13} │ {stats_with['relative_velocity_mean']:6.3f} ± {stats_with['relative_velocity_std']:6.3f}")
    
    # 计算改进
    vel_improve = (stats_without['relative_velocity_mean'] - stats_with['relative_velocity_mean']) / stats_without['relative_velocity_mean'] * 100
    print(f"\n  💡 平均相对速度改进: {vel_improve:+.2f}%")
    
    # 表3: 视角差统计
    print("\n【表3：视角差统计 (单位：度)】")
    print("\n  ▶ 总视角差 (Total Viewing Angle Deviation)")
    print("─"*90)
    print(f"{'指标':<20} │ {'无速度预测':<30} │ {'有速度预测':<30}")
    print("─"*90)
    print(f"{'最大总视角差':<20} │ {stats_without['total_angle_max']:6.2f}°{'':<23} │ {stats_with['total_angle_max']:6.2f}°")
    print(f"{'平均视角 ± 标准差':<20} │ {stats_without['total_angle_mean']:6.2f}° ± {stats_without['total_angle_std']:6.2f}°{'':<12} │ {stats_with['total_angle_mean']:6.2f}° ± {stats_with['total_angle_std']:6.2f}°")
    
    total_angle_improve = (stats_without['total_angle_mean'] - stats_with['total_angle_mean']) / stats_without['total_angle_mean'] * 100
    print(f"\n  💡 平均总视角差改进: {total_angle_improve:+.2f}%")
    
    print("\n  ▶ 水平视角差 (Horizontal Viewing Angle Deviation)")
    print("─"*90)
    print(f"{'指标':<20} │ {'无速度预测':<30} │ {'有速度预测':<30}")
    print("─"*90)
    print(f"{'最大水平视角差':<20} │ {stats_without['horizontal_angle_max']:6.2f}°{'':<23} │ {stats_with['horizontal_angle_max']:6.2f}°")
    print(f"{'平均视角 ± 标准差':<20} │ {stats_without['horizontal_angle_mean']:6.2f}° ± {stats_without['horizontal_angle_std']:6.2f}°{'':<12} │ {stats_with['horizontal_angle_mean']:6.2f}° ± {stats_with['horizontal_angle_std']:6.2f}°")
    
    h_angle_improve = (stats_without['horizontal_angle_mean'] - stats_with['horizontal_angle_mean']) / stats_without['horizontal_angle_mean'] * 100
    print(f"\n  💡 平均水平视角差改进: {h_angle_improve:+.2f}%")
    
    print("\n  ▶ 垂直视角差 (Vertical Viewing Angle Deviation)")
    print("─"*90)
    print(f"{'指标':<20} │ {'无速度预测':<30} │ {'有速度预测':<30}")
    print("─"*90)
    print(f"{'最大垂直视角差':<20} │ {stats_without['vertical_angle_max']:6.2f}°{'':<23} │ {stats_with['vertical_angle_max']:6.2f}°")
    print(f"{'平均视角 ± 标准差':<20} │ {stats_without['vertical_angle_mean']:6.2f}° ± {stats_without['vertical_angle_std']:6.2f}°{'':<12} │ {stats_with['vertical_angle_mean']:6.2f}° ± {stats_with['vertical_angle_std']:6.2f}°")
    
    v_angle_improve = (stats_without['vertical_angle_mean'] - stats_with['vertical_angle_mean']) / stats_without['vertical_angle_mean'] * 100
    print(f"\n  💡 平均垂直视角差改进: {v_angle_improve:+.2f}%")
    
    # 汇总表
    print("\n【表4：性能改进汇总】")
    print("─"*90)
    print(f"{'性能指标':<30} │ {'改进百分比':<20}")
    print("─"*90)
    print(f"{'平均距离':<30} │ {dist_improve:+7.2f}%")
    print(f"{'平均相对速度':<30} │ {vel_improve:+7.2f}%")
    print(f"{'平均总视角差':<30} │ {total_angle_improve:+7.2f}%")
    print(f"{'平均水平视角差':<30} │ {h_angle_improve:+7.2f}%")
    print(f"{'平均垂直视角差':<30} │ {v_angle_improve:+7.2f}%")
    
    # 距离标准差改进
    dist_std_improve = (stats_without['distance_std'] - stats_with['distance_std']) / stats_without['distance_std'] * 100
    print(f"{'距离标准差（稳定性）':<30} │ {dist_std_improve:+7.2f}%")
    
    # 视角标准差改进
    total_angle_std_improve = (stats_without['total_angle_std'] - stats_with['total_angle_std']) / stats_without['total_angle_std'] * 100
    print(f"{'总视角差标准差（稳定性）':<30} │ {total_angle_std_improve:+7.2f}%")
    
    print("\n" + "="*90)
    print(" ✓ 分析完成")
    print("="*90)


def analyze_trajectory_pair(log_dir, traj_type, traj_display_name):
    """分析一对轨迹文件（无速度预测 vs 有速度预测）"""
    
    # 文件路径
    csv_without = os.path.join(log_dir, f'{traj_type}_no_vel.csv')
    csv_with = os.path.join(log_dir, f'{traj_type}.csv')
    
    # 检查文件是否存在
    if not os.path.exists(csv_without):
        print(f"  ✗ 文件不存在: {csv_without}")
        return False
    
    if not os.path.exists(csv_with):
        print(f"  ✗ 文件不存在: {csv_with}")
        return False
    
    print(f"\n正在分析 {traj_display_name} 轨迹...")
    print(f"  步骤1: 使用 vis_target_position.py 的时间片段截取逻辑")
    
    # 第一步：读取并使用 vis_target_position.py 的截取逻辑
    data_without = pd.read_csv(csv_without).dropna().sort_values('timestamp').reset_index(drop=True)
    data_with = pd.read_csv(csv_with).dropna().sort_values('timestamp').reset_index(drop=True)
    
    data_without_truncated = truncate_data_by_velocity(data_without, traj_type)
    data_with_truncated = truncate_data_by_velocity(data_with, traj_type)
    
    time_without = data_without_truncated['timestamp'].iloc[-1] - data_without_truncated['timestamp'].iloc[0]
    time_with = data_with_truncated['timestamp'].iloc[-1] - data_with_truncated['timestamp'].iloc[0]
    
    print(f"    截取后时长: 无速度预测={time_without:.2f}s, 有速度预测={time_with:.2f}s")
    
    # 第二步：使用最短的时间长度
    min_time = min(time_without, time_with)
    print(f"  步骤2: 统一截取到最短时长 {min_time:.2f}s 确保公平对比")
    
    # 重新分析，先用 vis_target_position.py 逻辑，再截取到最短时长
    stats_without = analyze_csv_with_min_time(csv_without, '无速度预测', traj_type, min_time)
    stats_with = analyze_csv_with_min_time(csv_with, '有速度预测', traj_type, min_time)
    
    if not stats_without or not stats_with:
        print(f"  ✗ 数据分析失败")
        return False
    
    print(f"  ✓ 无速度预测: {stats_without['data_points']} 数据点, {stats_without['time_duration']:.2f}s")
    print(f"  ✓ 有速度预测: {stats_with['data_points']} 数据点, {stats_with['time_duration']:.2f}s")
    
    # 打印对比表格
    print_comparison_table(stats_without, stats_with, traj_display_name)
    
    return True


def analyze_csv_with_min_time(csv_path, method_name, traj_type, min_time):
    """
    先使用 vis_target_position.py 的截取逻辑，再截取到最短时长
    """
    if not os.path.exists(csv_path):
        return None
    
    # 读取数据
    data = pd.read_csv(csv_path)
    data = data.dropna()
    data_sorted = data.sort_values('timestamp').reset_index(drop=True)
    
    if len(data_sorted) < 3:
        return None
    
    # 第一步：使用 vis_target_position.py 的时间片段截取逻辑
    data_sorted = truncate_data_by_velocity(data_sorted, traj_type)
    
    if len(data_sorted) < 3:
        return None
    
    # 第二步：截取到最短时长
    start_time = data_sorted['timestamp'].iloc[0]
    data_sorted = data_sorted[data_sorted['timestamp'] <= start_time + min_time].reset_index(drop=True)
    
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
    
    # 编译统计结果
    stats = {
        'method': method_name,
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


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    log_dir = os.path.join(project_root, 'test_log')
    
    # 支持的轨迹类型
    trajectories = [
        ('circle', 'Circle'),
        ('typeD', 'Type-D'),
        ('type8', 'Figure-8'),
    ]
    
    # 如果有命令行参数，只分析指定的轨迹
    if len(sys.argv) > 1:
        traj_arg = sys.argv[1].lower()
        trajectories = [(t, n) for t, n in trajectories if t == traj_arg]
        if not trajectories:
            print(f"错误: 未知的轨迹类型 '{sys.argv[1]}'")
            print(f"支持的类型: circle, typeD, type8")
            return
    
    print("="*90)
    print(" 轨迹跟踪性能对比分析：无速度预测 vs 有速度预测")
    print("="*90)
    
    # 分析所有轨迹
    success_count = 0
    for traj_type, traj_name in trajectories:
        if analyze_trajectory_pair(log_dir, traj_type, traj_name):
            success_count += 1
    
    print("\n" + "="*90)
    print(f" ✓ 完成分析，成功处理 {success_count}/{len(trajectories)} 个轨迹")
    print("="*90)


if __name__ == '__main__':
    main()

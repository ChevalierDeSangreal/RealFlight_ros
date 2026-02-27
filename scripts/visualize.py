import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取数据
data = pd.read_csv('/tmp/realflight_tracking_data.csv')
# data = pd.read_csv('/home/core/wangzimo/RealFlight_ros/test_log/circle.csv')

# 打印数据基本信息
print(f"数据总行数: {len(data)}")
print(f"时间戳范围: {data['timestamp'].min():.2f} - {data['timestamp'].max():.2f}")
print(f"时间跨度: {data['timestamp'].max() - data['timestamp'].min():.2f} 秒")

# 检查是否有NaN值
print(f"缺失值检查:")
print(f"  timestamp: {data['timestamp'].isna().sum()}")
print(f"  drone_x: {data['drone_x'].isna().sum()}")
print(f"  target_x: {data['target_x'].isna().sum()}")

# 删除包含NaN的行
data = data.dropna()
print(f"删除NaN后数据行数: {len(data)}")

# 确保数据按时间排序
data_sorted = data.sort_values('timestamp').reset_index(drop=True)
time_sorted = data_sorted['timestamp'] - data_sorted['timestamp'].iloc[0]

# 检查时间戳是否有效（如果所有时间戳相同，使用数据索引作为时间轴）
if time_sorted.max() <= 0.0:
    print("警告: 所有时间戳相同，使用数据索引作为时间轴")
    # 假设采样频率为10Hz（根据visualization_rate默认值）
    # 或者根据数据量估算：如果有N个数据点，假设总时长，计算采样频率
    estimated_duration = len(data_sorted) / 10.0  # 假设10Hz采样
    time_sorted = np.arange(len(data_sorted)) / 10.0  # 使用索引/采样频率作为时间
    print(f"使用估算时间轴: 0.00 - {time_sorted.max():.2f} 秒 (假设10Hz采样)")
else:
    print(f"时间范围: {time_sorted.min():.2f} - {time_sorted.max():.2f} 秒")

# 重新计算排序后的数据
dx_sorted = data_sorted['target_x'] - data_sorted['drone_x']
dy_sorted = data_sorted['target_y'] - data_sorted['drone_y']
dz_sorted = data_sorted['target_z'] - data_sorted['drone_z']
distance_sorted = np.sqrt(dx_sorted**2 + dy_sorted**2 + dz_sorted**2)

drone_velocity_magnitude_sorted = np.sqrt(data_sorted['drone_vx']**2 + data_sorted['drone_vy']**2 + data_sorted['drone_vz']**2)
target_velocity_magnitude_sorted = np.sqrt(data_sorted['target_vx']**2 + data_sorted['target_vy']**2 + data_sorted['target_vz']**2)
velocity_diff_sorted = drone_velocity_magnitude_sorted - target_velocity_magnitude_sorted

# 计算无人机机体x轴与到目标方向的夹角
# 使用完整的roll、pitch、yaw构建旋转矩阵
drone_roll_sorted = data_sorted['drone_roll']    # ENU坐标系中的roll角
drone_pitch_sorted = data_sorted['drone_pitch']  # ENU坐标系中的pitch角
drone_yaw_sorted = data_sorted['drone_yaw']      # ENU坐标系中的yaw角

# 从无人机到目标的方向向量（归一化）
direction_norm = np.sqrt(dx_sorted**2 + dy_sorted**2 + dz_sorted**2)
direction_x_normalized = dx_sorted / (direction_norm + 1e-8)
direction_y_normalized = dy_sorted / (direction_norm + 1e-8)
direction_z_normalized = dz_sorted / (direction_norm + 1e-8)

# 使用ZYX欧拉角顺序（yaw-pitch-roll）构建旋转矩阵，将世界坐标转换到机体坐标
# R = Rz(yaw) * Ry(pitch) * Rx(roll)
# 这里我们需要逆旋转：R^T = Rx(-roll) * Ry(-pitch) * Rz(-yaw)
# 简化：直接用旋转矩阵计算

cos_roll = np.cos(drone_roll_sorted)
sin_roll = np.sin(drone_roll_sorted)
cos_pitch = np.cos(drone_pitch_sorted)
sin_pitch = np.sin(drone_pitch_sorted)
cos_yaw = np.cos(drone_yaw_sorted)
sin_yaw = np.sin(drone_yaw_sorted)

# 世界坐标系到机体坐标系的旋转矩阵（每个时间步都不同）
# 机体x轴在世界坐标系中的表示
body_x_world_x = cos_yaw * cos_pitch
body_x_world_y = sin_yaw * cos_pitch
body_x_world_z = -sin_pitch

# 将目标方向向量转换到机体坐标系
# R^T * direction_world = direction_body
direction_body_x = (cos_yaw * cos_pitch * dx_sorted + 
                   sin_yaw * cos_pitch * dy_sorted - 
                   sin_pitch * dz_sorted)
direction_body_y = ((cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll) * dx_sorted +
                   (sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll) * dy_sorted +
                   cos_pitch * sin_roll * dz_sorted)
direction_body_z = ((cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll) * dx_sorted +
                   (sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll) * dy_sorted +
                   cos_pitch * cos_roll * dz_sorted)

# 归一化机体坐标系中的方向向量
direction_body_norm = np.sqrt(direction_body_x**2 + direction_body_y**2 + direction_body_z**2)
direction_body_x_normalized = direction_body_x / (direction_body_norm + 1e-8)
direction_body_y_normalized = direction_body_y / (direction_body_norm + 1e-8)
direction_body_z_normalized = direction_body_z / (direction_body_norm + 1e-8)

# 3D夹角：机体x轴[1,0,0]与目标方向的夹角
# cos(angle) = dot([1,0,0], direction_body_normalized) = direction_body_x_normalized
cos_angle_3d = np.clip(direction_body_x_normalized, -1.0, 1.0)
angle_3d_deg_sorted = np.degrees(np.arccos(cos_angle_3d))

# 投影夹角（机体系）：
# - 在x-y平面：反映目标相对机体y方向的偏转（水平偏转，正值=右侧，负值=左侧）
# - 在x-z平面：反映目标相对机体z方向的偏转（俯仰偏转，正值=上方，负值=下方）
angle_proj_y_deg_sorted = np.degrees(np.arctan2(direction_body_y, direction_body_x))
angle_proj_z_deg_sorted = np.degrees(np.arctan2(direction_body_z, direction_body_x))

# 检测目标物体开始移动和停止移动的时刻
# 使用速度阈值来判断是否在移动（阈值设为0.05 m/s）
velocity_threshold = 0.05
is_moving = target_velocity_magnitude_sorted > velocity_threshold

# 找到开始移动的时刻（从静止到移动的转变）
start_moving_indices = []
for i in range(1, len(is_moving)):
    if not is_moving[i-1] and is_moving[i]:
        start_moving_indices.append(i)

# 找到停止移动的时刻（从移动到静止的转变）
stop_moving_indices = []
for i in range(1, len(is_moving)):
    if is_moving[i-1] and not is_moving[i]:
        stop_moving_indices.append(i)

# 提取对应的时间
start_moving_times = [time_sorted.iloc[i] for i in start_moving_indices]
stop_moving_times = [time_sorted.iloc[i] for i in stop_moving_indices]

print(f"\n【目标运动时刻】")
print(f"  开始移动时刻: {start_moving_times} 秒")
print(f"  停止移动时刻: {stop_moving_times} 秒")

# 截断数据：只保留目标物体停止移动之前的数据
if stop_moving_indices:
    # 使用第一个停止移动的索引作为截断点
    cutoff_index = stop_moving_indices[0]
    print(f"\n⚠️  数据截断：仅可视化目标停止前的数据（索引 0-{cutoff_index}）")
    
    # 截断所有数据
    data_sorted = data_sorted.iloc[:cutoff_index].reset_index(drop=True)
    time_sorted = time_sorted.iloc[:cutoff_index].reset_index(drop=True)
    distance_sorted = distance_sorted.iloc[:cutoff_index].reset_index(drop=True)
    velocity_diff_sorted = velocity_diff_sorted.iloc[:cutoff_index].reset_index(drop=True)
    angle_3d_deg_sorted = angle_3d_deg_sorted.iloc[:cutoff_index].reset_index(drop=True)
    angle_proj_y_deg_sorted = angle_proj_y_deg_sorted.iloc[:cutoff_index].reset_index(drop=True)
    angle_proj_z_deg_sorted = angle_proj_z_deg_sorted.iloc[:cutoff_index].reset_index(drop=True)
    
    # 更新停止移动时刻（截断后不再显示）
    stop_moving_times = []
    stop_moving_indices = []
    
    print(f"   截断后数据行数: {len(data_sorted)}")
    print(f"   截断后时间范围: {time_sorted.min():.2f} - {time_sorted.max():.2f} 秒")

# 绘制五个子图
fig, axes = plt.subplots(5, 1, figsize=(12, 18))

# 子图1: 到目标物体距离随时间变化
axes[0].plot(time_sorted, distance_sorted, linewidth=2, color='blue', label='Distance')
axes[0].axhline(y=distance_sorted.mean(), color='r', linestyle='--', 
                label=f'Mean: {distance_sorted.mean():.4f} m')
# 标记目标开始移动和停止移动的时刻
for t in start_moving_times:
    axes[0].axvline(x=t, color='green', linestyle=':', linewidth=2, alpha=0.7)
for t in stop_moving_times:
    axes[0].axvline(x=t, color='red', linestyle=':', linewidth=2, alpha=0.7)
# 添加图例说明
if start_moving_times:
    axes[0].axvline(x=-999, color='green', linestyle=':', linewidth=2, label='Target Start Moving')
if stop_moving_times:
    axes[0].axvline(x=-999, color='red', linestyle=':', linewidth=2, label='Target Stop Moving')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Distance (m)')
axes[0].set_title('Distance to Target vs Time')
axes[0].legend()
axes[0].grid(True)
if time_sorted.max() > time_sorted.min():
    axes[0].set_xlim([time_sorted.min(), time_sorted.max()])

# 子图2: 与目标物体速度值差异随时间变化
axes[1].plot(time_sorted, velocity_diff_sorted, linewidth=2, color='orange', label='Velocity Difference')
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
# 标记目标开始移动和停止移动的时刻
for t in start_moving_times:
    axes[1].axvline(x=t, color='green', linestyle=':', linewidth=2, alpha=0.7)
for t in stop_moving_times:
    axes[1].axvline(x=t, color='red', linestyle=':', linewidth=2, alpha=0.7)
# 添加图例说明
if start_moving_times:
    axes[1].axvline(x=-999, color='green', linestyle=':', linewidth=2, label='Target Start Moving')
if stop_moving_times:
    axes[1].axvline(x=-999, color='red', linestyle=':', linewidth=2, label='Target Stop Moving')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Velocity Difference (m/s)')
axes[1].set_title('Velocity Magnitude Difference vs Time')
axes[1].legend()
axes[1].grid(True)
if time_sorted.max() > time_sorted.min():
    axes[1].set_xlim([time_sorted.min(), time_sorted.max()])

# 子图3: 机体x轴与到目标方向的3D夹角随时间变化
axes[2].plot(time_sorted, angle_3d_deg_sorted, linewidth=2, color='green', label='Angle (3D)')
axes[2].axhline(y=angle_3d_deg_sorted.mean(), color='r', linestyle='--', 
                label=f'Mean: {angle_3d_deg_sorted.mean():.4f} deg')
# 标记目标开始移动和停止移动的时刻
for t in start_moving_times:
    axes[2].axvline(x=t, color='green', linestyle=':', linewidth=2, alpha=0.7)
for t in stop_moving_times:
    axes[2].axvline(x=t, color='red', linestyle=':', linewidth=2, alpha=0.7)
# 添加图例说明
if start_moving_times:
    axes[2].axvline(x=-999, color='green', linestyle=':', linewidth=2, label='Target Start Moving')
if stop_moving_times:
    axes[2].axvline(x=-999, color='red', linestyle=':', linewidth=2, label='Target Stop Moving')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Angle (deg)')
axes[2].set_title('Angle (3D): Body X-axis to Target Direction vs Time')
axes[2].legend()
axes[2].grid(True)
if time_sorted.max() > time_sorted.min():
    axes[2].set_xlim([time_sorted.min(), time_sorted.max()])

# 子图4: 机体x轴与目标方向在x-y平面投影的夹角（y方向投影夹角）
axes[3].plot(time_sorted, angle_proj_y_deg_sorted, linewidth=2, color='purple', label='Angle (XY projection)')
axes[3].axhline(y=angle_proj_y_deg_sorted.mean(), color='r', linestyle='--',
                label=f'Mean: {angle_proj_y_deg_sorted.mean():.4f} deg')
# 标记目标开始移动和停止移动的时刻
for t in start_moving_times:
    axes[3].axvline(x=t, color='green', linestyle=':', linewidth=2, alpha=0.7)
for t in stop_moving_times:
    axes[3].axvline(x=t, color='red', linestyle=':', linewidth=2, alpha=0.7)
# 添加图例说明
if start_moving_times:
    axes[3].axvline(x=-999, color='green', linestyle=':', linewidth=2, label='Target Start Moving')
if stop_moving_times:
    axes[3].axvline(x=-999, color='red', linestyle=':', linewidth=2, label='Target Stop Moving')
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('Angle (deg)')
axes[3].set_title('Projection Angle (XY): Body X-axis vs Target Direction Projection')
axes[3].legend()
axes[3].grid(True)
if time_sorted.max() > time_sorted.min():
    axes[3].set_xlim([time_sorted.min(), time_sorted.max()])

# 子图5: 机体x轴与目标方向在x-z平面投影的夹角（z方向投影夹角）
axes[4].plot(time_sorted, angle_proj_z_deg_sorted, linewidth=2, color='brown', label='Angle (XZ projection)')
axes[4].axhline(y=angle_proj_z_deg_sorted.mean(), color='r', linestyle='--',
                label=f'Mean: {angle_proj_z_deg_sorted.mean():.4f} deg')
# 标记目标开始移动和停止移动的时刻
for t in start_moving_times:
    axes[4].axvline(x=t, color='green', linestyle=':', linewidth=2, alpha=0.7)
for t in stop_moving_times:
    axes[4].axvline(x=t, color='red', linestyle=':', linewidth=2, alpha=0.7)
# 添加图例说明
if start_moving_times:
    axes[4].axvline(x=-999, color='green', linestyle=':', linewidth=2, label='Target Start Moving')
if stop_moving_times:
    axes[4].axvline(x=-999, color='red', linestyle=':', linewidth=2, label='Target Stop Moving')
axes[4].set_xlabel('Time (s)')
axes[4].set_ylabel('Angle (deg)')
axes[4].set_title('Projection Angle (XZ): Body X-axis vs Target Direction Projection')
axes[4].legend()
axes[4].grid(True)
if time_sorted.max() > time_sorted.min():
    axes[4].set_xlim([time_sorted.min(), time_sorted.max()])

plt.tight_layout()
# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'tracking_performance.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"图片已保存到: {output_path}")

# 打印统计信息
print("\n=== RealFlight 跟踪统计信息 ===")
print(f"\n【距离统计】")
print(f"  平均值: {distance_sorted.mean():.4f} m")
print(f"  方差:   {distance_sorted.var():.4f} m²")
print(f"  标准差: {distance_sorted.std():.4f} m")
print(f"  最大值: {distance_sorted.max():.4f} m")
print(f"  最小值: {distance_sorted.min():.4f} m")

print(f"\n【夹角统计】")
print(f"  3D夹角 平均值: {angle_3d_deg_sorted.mean():.4f}°")
print(f"  3D夹角 方差:   {angle_3d_deg_sorted.var():.4f}°²")
print(f"  3D夹角 标准差: {angle_3d_deg_sorted.std():.4f}°")
print(f"  3D夹角 最大值: {angle_3d_deg_sorted.max():.4f}°")
print(f"  3D夹角 最小值: {angle_3d_deg_sorted.min():.4f}°")

print(f"\n  XY投影夹角(对应y方向偏转) 平均值: {angle_proj_y_deg_sorted.mean():.4f}°")
print(f"  XY投影夹角 方差:   {angle_proj_y_deg_sorted.var():.4f}°²")
print(f"  XY投影夹角 标准差: {angle_proj_y_deg_sorted.std():.4f}°")
print(f"  XY投影夹角 最大值: {angle_proj_y_deg_sorted.max():.4f}°")
print(f"  XY投影夹角 最小值: {angle_proj_y_deg_sorted.min():.4f}°")

print(f"\n  XZ投影夹角(对应z方向偏转) 平均值: {angle_proj_z_deg_sorted.mean():.4f}°")
print(f"  XZ投影夹角 方差:   {angle_proj_z_deg_sorted.var():.4f}°²")
print(f"  XZ投影夹角 标准差: {angle_proj_z_deg_sorted.std():.4f}°")
print(f"  XZ投影夹角 最大值: {angle_proj_z_deg_sorted.max():.4f}°")
print(f"  XZ投影夹角 最小值: {angle_proj_z_deg_sorted.min():.4f}°")

print(f"\n【速度差异统计】")
print(f"  平均值: {velocity_diff_sorted.mean():.4f} m/s")
print(f"  标准差: {velocity_diff_sorted.std():.4f} m/s")
print(f"  最大值: {velocity_diff_sorted.max():.4f} m/s")
print(f"  最小值: {velocity_diff_sorted.min():.4f} m/s")

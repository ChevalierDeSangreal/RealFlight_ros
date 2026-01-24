# Tracking Visualizer - RealFlight 追踪可视化工具

## 概述

这个包为 RealFlight_ros 提供实时追踪结果可视化和数据记录功能，专门用于分析神经网络控制的跟踪性能。

## 主要功能

- ✅ 实时显示无人机和目标的历史轨迹
- ✅ 可视化跟踪误差（距离、速度）
- ✅ 自动保存追踪数据到 CSV 文件
- ✅ 监听状态机，只在 TRAJ 状态记录数据
- ✅ 统计分析（均值、最大值、RMS）
- ✅ RViz 3D 可视化支持

## 快速开始

### 1. 编译

```bash
cd ~/wangzimo/RealFlight_ros
catkin_make
source devel/setup.bash
```

### 2. 启动可视化

```bash
# 终端1-6：按照 RealFlight_ros 主 README 启动系统

# 终端7：启动可视化节点
roslaunch tracking_visualizer tracking_visualizer.launch
```

### 3. RViz 可视化

```bash
rviz -d $(rospack find tracking_visualizer)/config/tracking_visualization.rviz
```

## 可视化内容

在 RViz 中可以看到：

- **绿色轨迹**：无人机历史路径
- **红色轨迹**：目标历史路径
- **黄色连线**：当前跟踪误差
- **球体标记**：无人机（绿色）和目标（红色）的当前位置
- **文本显示**：实时误差数值和状态（[TRACKING] 表示在 TRAJ 状态）

## 发布话题

| 话题名称 | 消息类型 | 说明 |
|---------|---------|------|
| `/tracking_viz/drone_path` | `nav_msgs/Path` | 无人机历史轨迹 |
| `/tracking_viz/target_path` | `nav_msgs/Path` | 目标历史轨迹 |
| `/tracking_viz/error_markers` | `visualization_msgs/MarkerArray` | 误差可视化标记 |
| `/tracking_viz/distance_error` | `std_msgs/Float64` | 当前距离误差 [m] |
| `/tracking_viz/velocity_error` | `std_msgs/Float64` | 当前速度误差 [m/s] |

## 订阅话题

| 话题名称 | 消息类型 | 说明 |
|---------|---------|------|
| `/mavros/local_position/pose` | `geometry_msgs/PoseStamped` | 无人机位置 |
| `/mavros/local_position/velocity_local` | `geometry_msgs/TwistStamped` | 无人机速度 |
| `/target/position` | `geometry_msgs/PointStamped` | 目标位置 |
| `/target/velocity` | `geometry_msgs/TwistStamped` | 目标速度 |
| `/state/state_drone_0` | `std_msgs/Int32` | 状态机状态 |

## 配置参数

编辑 `config/visualizer_params.yaml`：

```yaml
# 无人机ID
drone_id: 0

# 轨迹记录
max_trajectory_points: 2000   # 最大轨迹点数
visualization_rate: 10.0      # 可视化频率 [Hz]

# 数据保存
save_to_file: true            # 是否保存数据
output_file_path: "/tmp/realflight_tracking_data.csv"

# 记录选项
only_record_in_traj: true     # 是否只在TRAJ状态时记录数据

# 颜色配置 (RGBA)
drone_color:   # 绿色
  r: 0.0
  g: 1.0
  b: 0.0
  a: 1.0

target_color:  # 红色
  r: 1.0
  g: 0.0
  b: 0.0
  a: 1.0
```

## 输出数据

CSV 文件包含以下字段：
- `timestamp` - 时间戳
- `drone_x, drone_y, drone_z` - 无人机位置
- `target_x, target_y, target_z` - 目标位置
- `drone_vx, drone_vy, drone_vz` - 无人机速度
- `target_vx, target_vy, target_vz` - 目标速度
- `distance_error` - 位置误差
- `velocity_error` - 速度误差


## 特殊功能

### 只在 TRAJ 状态记录

设置 `only_record_in_traj: true` 可以：
- 只在状态机进入 TRAJ 状态时记录数据
- 进入 TRAJ 状态时自动清空之前的轨迹
- 离开 TRAJ 状态时打印最终统计信息
- 确保数据只包含实际跟踪阶段的信息

### 状态监听

可视化节点会监听状态机状态，并在终端显示：
```
State changed: 4 -> 5
✅ Entered TRAJ state - Starting tracking data recording
...
Left TRAJ state - Stopped tracking data recording
=== Final Tracking Statistics ===
Total data points: 500
Mean distance error: 0.1234 m
Max distance error: 0.5678 m
RMS distance error: 0.2345 m
```

## 注意事项

1. 确保系统正常运行并发布位置信息
2. 数据文件在节点关闭时自动保存
3. 默认只在 TRAJ 状态记录数据，确保数据一致性
4. 可以通过修改参数调整可视化性能
5. CSV 文件默认保存在 `/tmp` 目录
6. 在对比测试时，建议使用相同的目标轨迹


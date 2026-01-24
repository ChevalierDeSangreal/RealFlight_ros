# RealFlight ROS1 版本

这是 RealFlight 项目的 ROS1 Melodic/Noetic 版本，支持在 Gazebo 中使用 MAVROS 与 PX4 SITL 进行无人机仿真控制。

**从 ROS2 版本移植**: 本项目从原 RealFlight ROS2 版本移植而来，使用 MAVROS 代替直接的 DDS 通信。

## 项目结构

```
RealFlight_ros/
├── src/
│   ├── offboard_state_machine/    # 状态机包 - 自动起飞、悬停、降落
│   ├── hover_test/                # 悬停测试包 - 简化版恒定推力控制
│   └── track_test/                # 轨迹跟踪测试包 - 圆形轨迹角速率控制
├── launch/                        # 主launch文件
├── build.sh                       # 编译脚本
├── run.sh                         # 快速启动脚本
└── README.md
```

## 快速开始

### 1. 编译项目
```bash
cd ~/RealFlight_ros
./build.sh
```

### 2. 启动仿真环境

**终端1：启动 ROS 核心**
```bash
roscore
```

**终端2：启动 PX4 SITL + Gazebo**
```bash
cd ~/PX4-Autopilot
make px4_sitl gazebo
```

**终端3：启动 MAVROS**
```bash
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
```

### 3. 运行任务

#### 任务1：悬停控制（Hover Test）

**终端4：启动状态机**
```bash
cd ~/RealFlight_ros
source devel/setup.bash
roslaunch offboard_state_machine single_drone_test.launch \
    drone_id:=0
```

**终端5：启动悬停控制节点（50Hz版本）**
```bash
cd ~/RealFlight_ros
source devel/setup.bash
roslaunch hover_test tflite_neural_control_50hz.launch drone_id:=0
```

**工作流程**：
1. 状态机自动执行：ARMING → TAKEOFF → HOVER
2. hover_test 检测到 HOVER 状态后，自动发送 TRAJ 命令
3. 状态机进入 TRAJ 状态（Rate Control 模式）
4. hover_test 开始神经网络控制（50Hz 推理，100Hz 发送控制指令）
5. 悬停 3 秒后，hover_test 发送 END_TRAJ 命令
6. 状态机进入 END_TRAJ 状态，等待 5 秒后自动降落

#### 任务2：轨迹跟踪（Track Test）

**终端4：启动状态机（Position Control模式）**
```bash
cd ~/RealFlight_ros
source devel/setup.bash
roslaunch offboard_state_machine single_drone_test.launch \
    drone_id:=0 \
    config_file:=$(find offboard_state_machine)/config/fsm_traj.yaml
```

**终端5：启动目标发布器（可选，用于话题模式）**
```bash
cd ~/RealFlight_ros
source devel/setup.bash
roslaunch track_test target_publisher.launch \
    drone_id:=0
```

**终端6：启动轨迹跟踪节点**
```bash
cd ~/RealFlight_ros
source devel/setup.bash
roslaunch track_test track_test.launch drone_id:=0
```

**工作流程**：
1. 状态机自动执行：ARMING → TAKEOFF → HOVER
2. track_test 检测到 HOVER 状态后，发送 TRAJ 命令
3. 状态机进入 TRAJ 状态（Position Control 模式）
4. track_test 开始神经网络控制，跟踪目标
   - **静态模式**：跟踪正前方固定目标（无需 target_publisher）
   - **话题模式**：跟踪 target_publisher 发布的圆周运动轨迹（需在 YAML 中设置 `use_target_topic: true`）
5. 跟踪完成后，track_test 发送 END_TRAJ 命令
6. 状态机进入 END_TRAJ 状态，等待 5 秒后自动降落

## 依赖项安装

### ROS1 环境
- **Ubuntu 18.04** + ROS Melodic 或 **Ubuntu 20.04** + ROS Noetic
```bash
# ROS Noetic (Ubuntu 20.04)
sudo apt install ros-noetic-desktop-full

# ROS Melodic (Ubuntu 18.04)  
sudo apt install ros-melodic-desktop-full
```

### PX4 和 Gazebo
```bash
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
bash ./Tools/setup/ubuntu.sh
make px4_sitl gazebo
```

### MAVROS
```bash
# ROS Noetic
sudo apt install ros-noetic-mavros ros-noetic-mavros-extras

# ROS Melodic
sudo apt install ros-melodic-mavros ros-melodic-mavros-extras

# 安装 GeographicLib 数据集
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
chmod +x install_geographiclib_datasets.sh
sudo ./install_geographiclib_datasets.sh
```

### 其他依赖
```bash
sudo apt install libeigen3-dev ros-$ROS_DISTRO-tf2*
```

## 包说明

### 1. offboard_state_machine

**功能**：自动起飞、悬停、降落状态机，支持最小 jerk 轨迹规划。

**状态流程**：
```
INIT(0) → ARMING(1) → TAKEOFF(2) → HOVER(4) → TRAJ(5) → END_TRAJ(6) → LAND(7) → DONE(8)
```

**关键参数**：
- `takeoff_alt`: 起飞高度 (默认: 1.2m)
- `traj_use_position_control`: TRAJ状态控制模式
  - `false`: Rate Control模式（用于hover_test）
  - `true`: Position Control模式（用于track_test）
- `end_traj_wait_time`: END_TRAJ状态悬停时间 (默认: 5.0s)

**配置文件**：
- `config/fsm_hover.yaml`: Rate Control模式（hover_test）
- `config/fsm_traj.yaml`: Position Control模式（track_test）

**启动**：
```bash
# Hover模式（默认）
roslaunch offboard_state_machine single_drone_test.launch drone_id:=0 mode:=sitl

# Track模式
roslaunch offboard_state_machine single_drone_test.launch \
    drone_id:=0 mode:=sitl \
    config_file:=$(find offboard_state_machine)/config/fsm_traj.yaml
```

**话题**：
- 发布: `/state/state_drone_0` - 当前状态
- 订阅: `/state/command_drone_0` - 状态切换命令

### 2. hover_test

**功能**：神经网络悬停控制（TensorFlow Lite），Rate Control模式。

**主要参数**：
- `model_path`: 模型文件路径（必需）
- `hover_duration`: 悬停时长 (默认: 3.0s)
- `action_update_period`: 推理周期 (默认: 0.02s, 50Hz)
- `control_send_period`: 控制发送周期 (默认: 0.01s, 100Hz)

**启动**：
```bash
roslaunch hover_test hover_test.launch drone_id:=0
```

**话题**：
- 订阅: `/state/state_drone_0`, `/mavros/odometry/in`, `/mavros/local_position/pose`, `/mavros/imu/data`
- 发布: `/state/command_drone_0`, `/mavros/setpoint_raw/attitude`

### 3. track_test

**功能**：神经网络目标跟踪控制（TensorFlow Lite），Position Control模式。

**目标模式**：
- **静态模式** (`use_target_topic: false`): 在正前方生成固定目标
- **话题模式** (`use_target_topic: true`): 订阅 `/target/position` 获取实时目标

**主要参数**：
- `model_path`: 模型文件路径（必需）
- `use_target_topic`: 目标模式选择 (默认: false)
- `target_offset_distance`: 静态模式目标距离 (默认: 1.0m)
- `hover_duration`: 跟踪时长 (默认: 3.0s)

**启动**：
```bash
# 静态模式
roslaunch track_test track_test.launch drone_id:=0

# 话题模式（需在YAML中设置 use_target_topic: true）
roslaunch track_test track_test.launch drone_id:=0
```

**话题**：
- 订阅: `/state/state_drone_0`, `/mavros/odometry/in`, `/mavros/local_position/pose`, `/mavros/imu/data`, `/target/position`（话题模式）
- 发布: `/state/command_drone_0`, `/mavros/setpoint_raw/local`

### 4. target_publisher

**功能**：发布圆周运动目标轨迹，用于 `track_test` 的话题模式。

**主要参数**：
- `circle_radius`: 轨迹半径 [m] (默认: 2.0)
- `circle_center_z`: 圆心高度 [m] (默认: 1.2)
- `circle_duration`: 单圈时间 [s] (默认: 20.0)
- `use_sim_time`: 使用仿真时间 (SITL设为true)

**启动**：
```bash
roslaunch track_test target_publisher.launch \
    drone_id:=0 \
    use_sim_time:=true
```

**话题**：
- 发布: `/target/position`, `/target/velocity`

### 5. tracking_visualizer

**功能**：追踪结果可视化工具，用于实时显示跟踪效果和性能分析。

**主要功能**：
- 实时显示无人机和目标的历史轨迹
- 计算和显示跟踪误差（距离、速度）
- 保存追踪数据到 CSV 文件
- 只在 TRAJ 状态时记录数据（可配置）
- 统计分析（均值、最大值、RMS）

**启动**：
```bash
# 在运行 track_test 的同时启动可视化
roslaunch tracking_visualizer tracking_visualizer.launch
```

**RViz 可视化**：
```bash
# 使用配置文件启动 RViz
rviz -d $(rospack find tracking_visualizer)/config/tracking_visualization.rviz
```

**话题**：
- 订阅: `/mavros/local_position/pose`, `/mavros/local_position/velocity_local`, `/target/position`, `/target/velocity`, `/state/state_drone_0`
- 发布: `/tracking_viz/drone_path`, `/tracking_viz/target_path`, `/tracking_viz/error_markers`, `/tracking_viz/distance_error`, `/tracking_viz/velocity_error`

**输出数据**：
- CSV 文件：`/tmp/realflight_tracking_data.csv`
- 包含时间戳、位置、速度、误差等完整信息

**可视化内容**：
- 绿色轨迹：无人机历史路径
- 红色轨迹：目标历史路径
- 黄色连线：当前跟踪误差
- 球体标记：无人机和目标的当前位置
- 文本显示：实时误差数值

**配置参数**：
- `config/visualizer_params.yaml`: 可视化参数配置
- `only_record_in_traj: true`: 只在 TRAJ 状态记录数据
- `save_to_file: true`: 自动保存数据到 CSV 文件
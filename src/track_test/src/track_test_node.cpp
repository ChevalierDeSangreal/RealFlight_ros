#include "track_test/track_test_node.hpp"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <string>
#include <std_msgs/Float64.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

// Observation space bounds (matching TrackEnvVer5/Ver6 training config)
// obs: [v_body(3), g_body(3), target_pos_body(3)]
namespace {
  constexpr float OBS_MIN[9] = {
    -20.0f, -20.0f, -20.0f,    // v_body (body-frame velocity)
    -1.0f, -1.0f, -1.0f,       // g_body (body-frame gravity direction)
    -100.0f, -100.0f, -100.0f  // target_pos_body (body-frame target position, relative)
  };
  
  constexpr float OBS_MAX[9] = {
    20.0f, 20.0f, 20.0f,       // v_body
    1.0f, 1.0f, 1.0f,          // g_body
    100.0f, 100.0f, 100.0f     // target_pos_body
  };
  
  // Normalize observation to [-1, 1] range (matching Python training code)
  void normalize_observation(std::vector<float>& obs) {
    for (size_t i = 0; i < obs.size() && i < 9; ++i) {
      obs[i] = 2.0f * (obs[i] - OBS_MIN[i]) / (OBS_MAX[i] - OBS_MIN[i]) - 1.0f;
      // Clamp to [-1, 1] to handle edge cases
      obs[i] = std::clamp(obs[i], -1.0f, 1.0f);
    }
  }
}

TrackTestNode::TrackTestNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private, int drone_id)
  : nh_(nh)
  , nh_private_(nh_private)
  , drone_id_(drone_id)
  , current_state_(FsmState::INIT)
  , waiting_hover_(false)
  , hover_command_sent_(false)
  , hover_started_(false)
  , hover_completed_(false)
  , neural_control_ready_(false)
  , hover_x_(0.0)
  , hover_y_(0.0)
  , hover_z_(0.0)
  , hover_yaw_(0.0)
  , current_x_(0.0)           
  , current_y_(0.0)           
  , current_z_(0.0)           
  , odom_ready_(false)
  , current_vx_(0.0)
  , current_vy_(0.0)
  , current_vz_(0.0)
  , current_roll_(0.0)
  , current_pitch_(0.0)
  , current_yaw_(0.0)
  , local_position_ready_(false)
  , attitude_ready_(false)
  , target_x_(0.0)
  , target_y_(0.0)
  , target_z_(0.0)
  , target_vx_(0.0)
  , target_vy_(0.0)
  , target_vz_(0.0)
  , predicted_target_vx_(0.0)
  , predicted_target_vy_(0.0)
  , predicted_target_vz_(0.0)
  , current_action_(4, 0.0f)  // Initialize with zero action [thrust, omega_x, omega_y, omega_z]
  , step_counter_(0)           // Initialize step counter
  , enable_thrust_change_(false)
  , thrust_weight_ratio_(1.0)
  , thrust_change_time_(5.0)
  , thrust_changed_(false)
{
  // Read parameters from node's private namespace (loaded from YAML in launch file)
  nh_private_.param("hover_duration", hover_duration_, 3.0);  // TRAJ control duration: 3.0s
  nh_private_.param("hover_thrust", hover_thrust_, 0.76);  // Normalized thrust for hovering
  nh_private_.param("mode_stabilization_delay", mode_stabilization_delay_, 0.6);  // Wait 0.6s for mode switch
  nh_private_.param("action_update_period", action_update_period_, 0.02);  // 50 Hz for NN inference
  nh_private_.param("control_send_period", control_send_period_, 0.01);  // 100 Hz for control commands
  
  // Model path is REQUIRED
  nh_private_.param<std::string>("model_path", model_path_, "");
  
  if (model_path_.empty()) {
    ROS_FATAL("model_path parameter is REQUIRED! Neural network control is mandatory.");
    ROS_FATAL("Please set model_path in launch file or YAML config.");
    ros::shutdown();
    return;
  }
  
  // Target generation parameters
  nh_private_.param("use_target_topic", use_target_topic_, false);
  nh_private_.param<std::string>("target_position_topic", target_position_topic_, "/target/position");
  nh_private_.param<std::string>("target_velocity_topic", target_velocity_topic_, "/target/velocity");
  nh_private_.param("target_offset_distance", target_offset_distance_, 1.0);  // Default 1m ahead
  
  // Thrust weight ratio change parameters (for testing payload changes)
  nh_private_.param("enable_thrust_change", enable_thrust_change_, false);
  nh_private_.param("thrust_weight_ratio", thrust_weight_ratio_, 1.0);
  nh_private_.param("thrust_change_time", thrust_change_time_, 5.0);
  
  // Initialize target generator
  target_generator_ = std::make_unique<TargetGenerator>(
    &nh_, 
    use_target_topic_,
    target_position_topic_,
    target_velocity_topic_);
  
  ROS_INFO("=== Trajectory Test Node for Drone %d (Neural Network Control - REQUIRED) ===", drone_id_);
  ROS_INFO("Parameters:");
  ROS_INFO("  - Model path: %s", model_path_.c_str());
  ROS_INFO("  - Hover duration: %.2f s", hover_duration_);
  ROS_INFO("  - Mode stabilization delay: %.2f s (wait for body_rate mode)", mode_stabilization_delay_);
  ROS_INFO("  - Action update period: %.3f s (%.1f Hz)", action_update_period_, 1.0/action_update_period_);
  ROS_INFO("  - Control send period: %.3f s (%.1f Hz)", control_send_period_, 1.0/control_send_period_);
  ROS_INFO("  - Target mode: %s", use_target_topic_ ? "ROS1 Topic" : "Static (front of drone)");
  if (use_target_topic_) {
    ROS_INFO("    - Position topic: %s", target_position_topic_.c_str());
    ROS_INFO("    - Velocity topic: %s", target_velocity_topic_.c_str());
  } else {
    ROS_INFO("    - Offset distance: %.2f m (in front of drone)", target_offset_distance_);
  }
  ROS_INFO("  - Thrust change: %s", enable_thrust_change_ ? "ENABLED" : "DISABLED");
  if (enable_thrust_change_) {
    ROS_INFO("    - Thrust weight ratio: %.2f (max thrust = %.0f%% of original)", 
             thrust_weight_ratio_, thrust_weight_ratio_ * 100.0);
    ROS_INFO("    - Change time: %.2f s after tracking starts", thrust_change_time_);
  }
  
  // Initialize neural network policy (REQUIRED)
  ROS_INFO("  - Initializing neural network policy...");
  policy_ = std::make_unique<TFLitePolicyInference>(model_path_);
  if (!policy_->is_initialized()) {
    ROS_FATAL("Failed to initialize neural network policy! Model path: %s", model_path_.c_str());
    ROS_FATAL("Neural network control is REQUIRED. Exiting...");
    ros::shutdown();
    return;
  }
  ROS_INFO("  ✅ Neural network policy initialized successfully");
  ROS_INFO("===================================================");
  
  // Publishers - using MAVROS AttitudeTarget for body rate control
  attitude_pub_ = nh_.advertise<mavros_msgs::AttitudeTarget>(
      "/mavros/setpoint_raw/attitude", 10);
    
  state_cmd_pub_ = nh_.advertise<std_msgs::Int32>(
      "/state/command_drone_" + std::to_string(drone_id_), 10);
  
  // Publisher for predicted target velocity (body frame)
  predicted_target_vel_pub_ = nh_.advertise<geometry_msgs::TwistStamped>(
      "/predicted_target/velocity", 10);
  
  // Publisher for neural network thrust output (normalized 0-1)
  thrust_output_pub_ = nh_.advertise<std_msgs::Float64>(
      "/neural_network/thrust_output", 10);
  
  // Subscribers
  state_sub_ = nh_.subscribe<std_msgs::Int32>(
      "/state/state_drone_" + std::to_string(drone_id_),
      10,
      &TrackTestNode::state_callback, this);
    
  local_pos_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>(
      "/mavros/local_position/pose",
      10,
      &TrackTestNode::local_pos_callback, this);
  
  local_vel_sub_ = nh_.subscribe<geometry_msgs::TwistStamped>(
      "/mavros/local_position/velocity_local",
      10,
      &TrackTestNode::local_vel_callback, this);
  
  // Two-timer architecture to avoid accumulated timing errors:
  // 1. Action update timer (50Hz): Neural network inference
  // 2. Control send timer (100Hz): High-frequency command transmission
  
  // Action update timer (50Hz for neural network inference)
  action_update_timer_ = nh_.createTimer(ros::Duration(action_update_period_), 
                                         &TrackTestNode::action_update_callback, this);
  
  // Control send timer (100Hz for sending control commands)
  control_send_timer_ = nh_.createTimer(ros::Duration(control_send_period_), 
                                         &TrackTestNode::control_send_callback, this);
  
  ROS_INFO("Timers initialized: Action update at %.0f Hz, Control send at %.0f Hz",
           1.0/action_update_period_, 1.0/control_send_period_);
}

void TrackTestNode::state_callback(const std_msgs::Int32::ConstPtr& msg)
{
  auto state = static_cast<FsmState>(msg->data);
  
  // Update current state
  current_state_ = state;

  // first HOVER detection
  if (state == FsmState::HOVER &&
      !waiting_hover_ &&
      !hover_command_sent_ &&
      !hover_started_) {
    waiting_hover_ = true;
    hover_detect_time_ = ros::Time::now();
    ROS_INFO("HOVER detected, will start hover control in 2.0s");
  }

  // Check if ready to send TRAJ command (after 2.0s delay)
  if (waiting_hover_ && !hover_started_ && !hover_completed_ &&
      (ros::Time::now() - hover_detect_time_).toSec() > 2.0) {
    
    // 对于话题模式，需要等待目标数据就绪
    if (use_target_topic_) {
      if (target_generator_->is_target_ready()) {
        // Target data ready, send TRAJ command
        ROS_INFO("Target topic data ready, sending TRAJ state command");
        
        // Get target position information for logging
        target_generator_->get_target_position(target_x_, target_y_, target_z_);
        target_generator_->get_target_velocity(target_vx_, target_vy_, target_vz_);
        
        ROS_INFO("Target position: [%.2f, %.2f, %.2f], velocity: [%.2f, %.2f, %.2f]",
                 target_x_, target_y_, target_z_, target_vx_, target_vy_, target_vz_);
        
        send_state_command(static_cast<int>(FsmState::TRAJ));
        hover_command_sent_ = true;
        hover_start_time_ = ros::Time::now();
      } else {
        // Target data not ready, print waiting message
        ROS_WARN_THROTTLE(2.0, "Waiting for target position topic data: %s (target data not ready, will not enter TRAJ state)",
                          target_position_topic_.c_str());
      }
    } else {
      // Static mode, send TRAJ command directly
      ROS_INFO("Static mode - sending TRAJ state command for hover control");
      send_state_command(static_cast<int>(FsmState::TRAJ));
      hover_command_sent_ = true;
      hover_start_time_ = ros::Time::now();
    }
  }

  // TRAJ detected - capture current position but WAIT before starting neural control
  if (state == FsmState::TRAJ && (waiting_hover_ || hover_command_sent_) && !hover_started_) {
    hover_started_ = true;
    waiting_hover_ = false;
    hover_command_sent_ = false;
    neural_control_ready_ = false;  // Reset - will be set to true after stabilization delay
    hover_start_time_ = ros::Time::now();
    step_counter_ = 0;  // Reset step counter
    thrust_changed_ = false;  // Reset thrust change flag
    
    // Capture current position for hover
    hover_x_ = current_x_;
    hover_y_ = current_y_;
    hover_z_ = current_z_;
    hover_yaw_ = current_yaw_;  // 使用当前yaw角
    
    // Initialize target using target generator
    if (!use_target_topic_) {
      // 静态模式: 在无人机正前方生成目标
      target_generator_->initialize_static_target(
        current_x_, current_y_, current_z_, current_yaw_, target_offset_distance_);
      
      // 获取生成的目标位置
      target_generator_->get_target_position(target_x_, target_y_, target_z_);
      target_generator_->get_target_velocity(target_vx_, target_vy_, target_vz_);
    } else {
      // Topic mode: target position was already obtained before sending TRAJ command, just confirm here
      if (!target_generator_->is_target_ready()) {
        ROS_ERROR("ERROR: Entered TRAJ state but target topic data is not ready! This should not happen.");
      } else {
        // Get latest target position again (may have been updated since last retrieval)
        target_generator_->get_target_position(target_x_, target_y_, target_z_);
        target_generator_->get_target_velocity(target_vx_, target_vy_, target_vz_);
        ROS_INFO("Target topic data confirmed ready");
      }
    }
    
    // Set initial hover action (will be used during stabilization period)
    {
      std::lock_guard<std::mutex> lock(action_mutex_);
      current_action_[0] = 2.0f * hover_thrust_ - 1.0f;  // Map [0,1] to [-1,1]
      current_action_[1] = 0.0f;
      current_action_[2] = 0.0f;
      current_action_[3] = 0.0f;
    }
    
    ROS_INFO("🚁 FSM entered TRAJ - Waiting %.2f s for mode stabilization...", 
             mode_stabilization_delay_);
    ROS_INFO("   Drone pos (ENU): [%.2f, %.2f, %.2f] yaw: %.2f rad | Target (ENU): [%.2f, %.2f, %.2f]",
             hover_x_, hover_y_, hover_z_, hover_yaw_, target_x_, target_y_, target_z_);
  }

  // Exited TRAJ, reset
  if (hover_started_ && state != FsmState::TRAJ) {
    hover_started_ = false;
    waiting_hover_ = false;
    hover_command_sent_ = false;
    neural_control_ready_ = false;
    ROS_INFO("Left TRAJ state, resetting");
  }
}

void TrackTestNode::local_pos_callback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
  current_x_ = msg->pose.position.x;
  current_y_ = msg->pose.position.y;
  current_z_ = msg->pose.position.z;
  
  // Convert quaternion to Euler angles (ENU frame)
  tf2::Quaternion q(
      msg->pose.orientation.x,
      msg->pose.orientation.y,
      msg->pose.orientation.z,
      msg->pose.orientation.w);
  tf2::Matrix3x3 m(q);
  m.getRPY(current_roll_, current_pitch_, current_yaw_);
  
  odom_ready_ = true;
  local_position_ready_ = true;
  attitude_ready_ = true;
}

void TrackTestNode::local_vel_callback(const geometry_msgs::TwistStamped::ConstPtr& msg)
{
  current_vx_ = msg->twist.linear.x;
  current_vy_ = msg->twist.linear.y;
  current_vz_ = msg->twist.linear.z;
}

// Action update callback (50Hz): Neural network inference
void TrackTestNode::action_update_callback(const ros::TimerEvent& event)
{
  if (!odom_ready_) {
    ROS_INFO_THROTTLE(5.0, "Waiting for odometry...");
    return;
  }
  
  // Only run in TRAJ state
  if (current_state_ == FsmState::TRAJ && hover_started_ && !hover_completed_) {
    // Calculate elapsed time from timestamp
    double elapsed = (ros::Time::now() - hover_start_time_).toSec();
    
    // Update target position from target generator
    if (use_target_topic_) {
      // Topic mode: get latest target position from target_generator
      if (target_generator_->is_target_ready()) {
        target_generator_->get_target_position(target_x_, target_y_, target_z_);
        target_generator_->get_target_velocity(target_vx_, target_vy_, target_vz_);
      } else {
        ROS_WARN_THROTTLE(1.0, "Waiting for target position topic data...");
      }
    }
    // Static mode: target position was set during initialization and remains unchanged
    
    // Check if TRAJ control duration is complete
    if (elapsed >= hover_duration_) {
      ROS_INFO("✅ TRAJ control complete (%.1f s) - sending END_TRAJ command", hover_duration_);
      send_state_command(static_cast<int>(FsmState::END_TRAJ));
      hover_completed_ = true;
      return;
    }
    
    // Check if we need to wait for mode stabilization
    if (!neural_control_ready_ && elapsed < mode_stabilization_delay_) {
      // Still in stabilization period - send safe hover action
      std::lock_guard<std::mutex> lock(action_mutex_);
      current_action_[0] = 2.0f * hover_thrust_ - 1.0f;  // Map [0,1] to [-1,1]
      current_action_[1] = 0.0f;
      current_action_[2] = 0.0f;
      current_action_[3] = 0.0f;
      
      ROS_INFO_THROTTLE(0.5, "⏳ Mode stabilization: %.2f/%.2f s - sending hover thrust",
                        elapsed, mode_stabilization_delay_);
      return;
    }
    
    // Initialize neural control after stabilization delay
    if (!neural_control_ready_ && elapsed >= mode_stabilization_delay_) {
      if (!local_position_ready_ || !attitude_ready_) {
        ROS_WARN_THROTTLE(1.0, "Waiting for position/attitude data... (pos: %s, att: %s)",
                          local_position_ready_ ? "ready" : "not ready",
                          attitude_ready_ ? "ready" : "not ready");
        return;
      }
      
      ROS_INFO("✅ Mode stabilized (%.2f s) - Initializing neural control...", elapsed);
      
      // Initialize buffer with zero observation vector
      std::vector<float> initial_obs_for_reset(OBS_DIM, 0.0f);
      
      // Hovering action (normalized): [thrust, omega_x=0, omega_y=0, omega_z=0]
      float hovering_thrust_normalized = 2.0f * hover_thrust_ - 1.0f;
      std::vector<float> hovering_action = {hovering_thrust_normalized, 0.0f, 0.0f, 0.0f};
      
      policy_->reset(initial_obs_for_reset, hovering_action);
      
      // Get ACTUAL observation for first inference
      std::vector<float> obs_raw = get_observation();
      std::vector<float> obs_normalized = obs_raw;
      normalize_observation(obs_normalized);
      
      // Immediately run inference to get first action
      std::vector<float> first_action = policy_->get_action(obs_normalized);
      
      // Update current_action_ immediately (thread-safe)
      {
        std::lock_guard<std::mutex> lock(action_mutex_);
        current_action_ = first_action;
      }
      
      neural_control_ready_ = true;
      ROS_INFO("🚀 Neural control active! (effective duration: %.1f s)", hover_duration_ - elapsed);
      return;
    }
    
    // Neural control is ready - update action from neural network
    if (neural_control_ready_) {
      if (!local_position_ready_ || !attitude_ready_) {
        ROS_WARN_THROTTLE(1.0, "Position/attitude data not ready!");
        return;
      }
      
      update_neural_action();
      
      // Log current status (throttled)
      ROS_INFO_THROTTLE(2.0, "Neural control | pos=[%.2f,%.2f,%.2f] vel=[%.2f,%.2f,%.2f] | target=[%.2f,%.2f,%.2f] | elapsed: %.1f/%.1f s",
                        current_x_, current_y_, current_z_,
                        current_vx_, current_vy_, current_vz_,
                        target_x_, target_y_, target_z_,
                        elapsed, hover_duration_);
    }
  }
}

// Control send callback (100Hz): High-frequency command transmission
void TrackTestNode::control_send_callback(const ros::TimerEvent& event)
{
  if (!odom_ready_) {
    return;
  }
  
  // Only run in TRAJ state
  if (current_state_ == FsmState::TRAJ && hover_started_ && !hover_completed_) {
    // Publish current action at high frequency
    publish_current_action();
  }
}

// Get observation vector for neural network (9D)
// Based on TrackEnvVer6 observation space - all in body frame (NED/FRD)
// Observation composition:
// 1. 机体系速度 (3) - quad velocity in body frame (FRD)
// 2. 机体系重力方向 (3) - gravity direction in body frame (FRD)
// 3. 机体系目标位置 (3) - target relative position in body frame (FRD)
// 
// Coordinate System:
// - Neural network trained with NED world frame (North-East-Down) and FRD body frame (Forward-Right-Down)
// - MAVROS provides ENU world frame (East-North-Up) and FLU body frame (Forward-Left-Up)
// - This function converts ENU -> NED -> FRD body frame for neural network input
std::vector<float> TrackTestNode::get_observation()
{
  std::vector<float> obs(OBS_DIM);
  
  // ===== STEP 1: Convert ENU world coordinates to NED world coordinates =====
  // Transformation: (x_ned, y_ned, z_ned) = (y_enu, x_enu, -z_enu)
  // MAVROS provides data in ENU frame, we need to convert to NED for neural network
  
  // Position in NED (from ENU)
  double pos_ned_x = current_y_;   // North = ENU Y
  double pos_ned_y = current_x_;   // East = ENU X
  double pos_ned_z = -current_z_;  // Down = -ENU Z
  
  // Velocity in NED (from ENU)
  double vel_ned_x = current_vy_;   // North velocity
  double vel_ned_y = current_vx_;   // East velocity
  double vel_ned_z = -current_vz_;  // Down velocity (negative of Up velocity)
  
  // Target position in NED (from ENU)
  double target_ned_x = target_y_;   // North
  double target_ned_y = target_x_;   // East
  double target_ned_z = -target_z_;  // Down
  
  // ===== STEP 2: Convert ENU Euler angles to NED Euler angles =====
  // MAVROS quaternion represents rotation from ENU world to FLU body
  // We need rotation from NED world to FRD body
  // 
  // ENU to NED Euler angle transformation:
  // roll_ned = roll_enu
  // pitch_ned = -pitch_enu
  // yaw_ned = -yaw_enu + π/2
  double roll_ned = current_roll_;
  double pitch_ned = -current_pitch_;
  double yaw_ned = -(current_yaw_ - M_PI/2.0);  // Adjust yaw reference
  
  // ===== STEP 3: Compute rotation matrix (NED world to FRD body) =====
  float cr = std::cos(roll_ned);
  float sr = std::sin(roll_ned);
  float cp = std::cos(pitch_ned);
  float sp = std::sin(pitch_ned);
  float cy = std::cos(yaw_ned);
  float sy = std::sin(yaw_ned);
  
  // Rotation matrix R (NED world to FRD body)
  // R = Rz(yaw) * Ry(pitch) * Rx(roll)
  float R[3][3];
  R[0][0] = cy * cp;
  R[0][1] = cy * sp * sr - sy * cr;
  R[0][2] = cy * sp * cr + sy * sr;
  R[1][0] = sy * cp;
  R[1][1] = sy * sp * sr + cy * cr;
  R[1][2] = sy * sp * cr - cy * sr;
  R[2][0] = -sp;
  R[2][1] = cp * sr;
  R[2][2] = cp * cr;
  
  // ===== STEP 4: Transform observations to FRD body frame =====
  // R^T transforms from NED world frame to FRD body frame
  
  // 1. Body-frame velocity (v_body_frd = R^T * v_world_ned)
  float v_world_ned[3] = {
    static_cast<float>(vel_ned_x),
    static_cast<float>(vel_ned_y),
    static_cast<float>(vel_ned_z)
  };
  obs[0] = R[0][0] * v_world_ned[0] + R[1][0] * v_world_ned[1] + R[2][0] * v_world_ned[2];
  obs[1] = R[0][1] * v_world_ned[0] + R[1][1] * v_world_ned[1] + R[2][1] * v_world_ned[2];
  obs[2] = R[0][2] * v_world_ned[0] + R[1][2] * v_world_ned[1] + R[2][2] * v_world_ned[2];
  
  // 2. Body-frame gravity direction (g_body_frd = R^T * g_world_ned)
  // In NED coordinate system, gravity points down: [0, 0, +1] (positive Z = Down)
  float g_world_ned[3] = {0.0f, 0.0f, 1.0f};  // NED: Down is positive
  obs[3] = R[0][0] * g_world_ned[0] + R[1][0] * g_world_ned[1] + R[2][0] * g_world_ned[2];
  obs[4] = R[0][1] * g_world_ned[0] + R[1][1] * g_world_ned[1] + R[2][1] * g_world_ned[2];
  obs[5] = R[0][2] * g_world_ned[0] + R[1][2] * g_world_ned[1] + R[2][2] * g_world_ned[2];
  
  // 3. Body-frame target relative position (target_pos_body_frd = R^T * (target - current)_ned)
  float target_rel_ned[3] = {
    static_cast<float>(target_ned_x - pos_ned_x),
    static_cast<float>(target_ned_y - pos_ned_y),
    static_cast<float>(target_ned_z - pos_ned_z)
  };
  obs[6] = R[0][0] * target_rel_ned[0] + R[1][0] * target_rel_ned[1] + R[2][0] * target_rel_ned[2];
  obs[7] = R[0][1] * target_rel_ned[0] + R[1][1] * target_rel_ned[1] + R[2][1] * target_rel_ned[2];
  obs[8] = R[0][2] * target_rel_ned[0] + R[1][2] * target_rel_ned[1] + R[2][2] * target_rel_ned[2];
  
  return obs;
}

// Update neural network action (called at 50Hz)
void TrackTestNode::update_neural_action()
{
  // Increment step counter
  step_counter_++;
  
  // Update target position from target generator (for topic mode)
  // This ensures we always use the latest target position for observation calculation
  if (use_target_topic_ && target_generator_->is_target_ready()) {
    target_generator_->get_target_position(target_x_, target_y_, target_z_);
    target_generator_->get_target_velocity(target_vx_, target_vy_, target_vz_);
  }
  
  // Get current observation (9D) - RAW (before normalization)
  std::vector<float> obs_raw = get_observation();
  
  // Make a copy for normalization
  std::vector<float> obs_normalized = obs_raw;
  
  // Normalize observation to [-1, 1] range (CRITICAL: must match training!)
  normalize_observation(obs_normalized);
  
  // Get full output from neural network (7D: action + aux_output)
  // Note: With separate 50Hz timer, we don't need internal action repeat anymore
  std::vector<float> full_output = policy_->get_action_and_aux(obs_normalized);
  
  // Extract action (first 4 dims) and auxiliary output (last 3 dims)
  std::vector<float> action(full_output.begin(), full_output.begin() + 4);
  std::vector<float> aux_output(full_output.begin() + 4, full_output.end());
  
  // Update current action (thread-safe)
  {
    std::lock_guard<std::mutex> lock(action_mutex_);
    current_action_ = action;
  }
  
  // Update predicted target velocity (body frame, thread-safe)
  {
    std::lock_guard<std::mutex> lock(predicted_vel_mutex_);
    predicted_target_vx_ = aux_output[0];
    predicted_target_vy_ = aux_output[1];
    predicted_target_vz_ = aux_output[2];
  }
  
  // Publish predicted target velocity (body frame)
  geometry_msgs::TwistStamped predicted_vel_msg;
  predicted_vel_msg.header.stamp = ros::Time::now();
  predicted_vel_msg.header.frame_id = "body";
  predicted_vel_msg.twist.linear.x = aux_output[0];
  predicted_vel_msg.twist.linear.y = aux_output[1];
  predicted_vel_msg.twist.linear.z = aux_output[2];
  predicted_target_vel_pub_.publish(predicted_vel_msg);
  
  // Calculate elapsed time
  double elapsed = (ros::Time::now() - hover_start_time_).toSec();
  
  // Use throttled logging to reduce print frequency (every 2 seconds)
  ROS_INFO_THROTTLE(2.0, "========== STEP %d (t=%.3fs) ==========", step_counter_, elapsed);
  
  // 原始观测（未归一化）
  ROS_INFO_THROTTLE(2.0, "[RAW OBS] v_body=[%.6f, %.6f, %.6f], g_body=[%.6f, %.6f, %.6f], target_pos_body=[%.6f, %.6f, %.6f]",
                    obs_raw[0], obs_raw[1], obs_raw[2],    // v_body
                    obs_raw[3], obs_raw[4], obs_raw[5],    // g_body
                    obs_raw[6], obs_raw[7], obs_raw[8]);   // target_pos_body
  
  // 网络原始输出（[-1, 1]范围的tanh输出）
  float thrust_raw = action[0];
  float omega_x_norm = action[1];
  float omega_y_norm = action[2];
  float omega_z_norm = action[3];
  
  ROS_INFO_THROTTLE(2.0, "[NN RAW OUTPUT] thrust_raw=%.6f, omega_x=%.6f, omega_y=%.6f, omega_z=%.6f",
                    thrust_raw, omega_x_norm, omega_y_norm, omega_z_norm);
  
  ROS_INFO_THROTTLE(2.0, "[NN AUX OUTPUT] predicted_target_v_body=[%.6f, %.6f, %.6f]",
                    aux_output[0], aux_output[1], aux_output[2]);
  
  ROS_INFO_THROTTLE(2.0, "=====================================");
}

// Publish current action (called at 100Hz)
void TrackTestNode::publish_current_action()
{
  // Read current action (thread-safe)
  std::vector<float> action;
  {
    std::lock_guard<std::mutex> lock(action_mutex_);
    action = current_action_;
  }
  
  // Action output: [thrust, omega_x, omega_y, omega_z]
  // All in range [-1, 1] from tanh activation
  float thrust_raw = action[0];
  float omega_x_norm = action[1];
  float omega_y_norm = action[2];
  float omega_z_norm = action[3];
  
  // Denormalize angular rates: [-1, 1] -> [-omega_max, omega_max]
  constexpr float OMEGA_MAX_X = 2.0f;  // Roll rate max [rad/s]
  constexpr float OMEGA_MAX_Y = 2.0f;  // Pitch rate max [rad/s]
  constexpr float OMEGA_MAX_Z = 3.0f;  // Yaw rate max [rad/s]
  
  // Neural network outputs body rates in FRD frame (NED-based training)
  // MAVROS expects body rates in FLU frame (ROS standard)
  // Coordinate transformation: FRD -> FLU
  //   roll_rate_flu = roll_rate_frd     (X axis: Forward same in both frames)
  //   pitch_rate_flu = -pitch_rate_frd  (Y axis: Right->Left, flip sign)
  //   yaw_rate_flu = -yaw_rate_frd      (Z axis: Down->Up, flip sign)
  float roll_rate_frd = omega_x_norm * OMEGA_MAX_X;
  float pitch_rate_frd = omega_y_norm * OMEGA_MAX_Y;
  float yaw_rate_frd = omega_z_norm * OMEGA_MAX_Z;
  
  // Convert FRD to FLU for MAVROS
  float roll_rate = roll_rate_frd;      // Roll rate: no change (Forward axis)
  float pitch_rate = -pitch_rate_frd;   // Pitch rate: flip sign (Right->Left)
  float yaw_rate = -yaw_rate_frd;       // Yaw rate: flip sign (Down->Up)
  
  // Thrust mapping: [-1, 1] -> [0, 1]
  float thrust_normalized = (thrust_raw + 1.0f) * 0.5f;
  
  // Calculate elapsed time from timestamp
  double elapsed = (ros::Time::now() - hover_start_time_).toSec();
  
  // Apply thrust weight ratio change if enabled and time condition is met
  if (enable_thrust_change_ && !thrust_changed_ && elapsed >= thrust_change_time_) {
    thrust_changed_ = true;
    ROS_WARN("⚠️  THRUST CHANGE ACTIVATED at t=%.2f s: Thrust ratio = %.2f (%.0f%% of original max thrust)",
             elapsed, thrust_weight_ratio_, thrust_weight_ratio_ * 100.0);
  }
  
  // Apply thrust weight ratio to the normalized thrust
  if (enable_thrust_change_ && thrust_changed_) {
    thrust_normalized *= thrust_weight_ratio_;
  }
  
  // Create MAVROS AttitudeTarget message
  mavros_msgs::AttitudeTarget msg;
  
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "base_link";
  
  // Set type_mask to use body rates
  // Bit 0-2: roll, pitch, yaw
  // Bit 3-5: roll_rate, pitch_rate, yaw_rate
  // Bit 6: thrust
  // We want to use body_rate (bits 3-5 = 0) and thrust (bit 6 = 0)
  // Ignore attitude (bits 0-2 = 1)
  msg.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ATTITUDE;  // 0b10000111 = 0x07
  
  // Body rates
  msg.body_rate.x = roll_rate;
  msg.body_rate.y = pitch_rate;
  msg.body_rate.z = yaw_rate;
  
  // Thrust (normalized 0-1)
  msg.thrust = thrust_normalized;
  
  // Orientation - not used when IGNORE_ATTITUDE is set
  msg.orientation.w = 1.0;
  msg.orientation.x = 0.0;
  msg.orientation.y = 0.0;
  msg.orientation.z = 0.0;
  
  // // Debug: Print control commands in the first 0.5 seconds
  // if (elapsed < 0.5) {
  //   ROS_INFO("[t=%.3fs] [PUBLISH@100Hz] thrust=%.3f(raw=%.3f), rates=[%.3f, %.3f, %.3f] rad/s",
  //            elapsed,
  //            thrust_normalized, thrust_raw,
  //            roll_rate, pitch_rate, yaw_rate);
  // }
  
  // Publish control commands
  attitude_pub_.publish(msg);
  
  // Publish neural network thrust output (RAW output from network in [-1, 1] range)
  // This is the direct tanh output before normalization to [0, 1]
  std_msgs::Float64 thrust_msg;
  thrust_msg.data = thrust_raw;  // Publish raw network output [-1, 1]
  thrust_output_pub_.publish(thrust_msg);
}

void TrackTestNode::send_state_command(int state)
{
  std_msgs::Int32 msg;
  msg.data = state;
  state_cmd_pub_.publish(msg);
  
  ROS_INFO("Sent state command: %d", state);
}

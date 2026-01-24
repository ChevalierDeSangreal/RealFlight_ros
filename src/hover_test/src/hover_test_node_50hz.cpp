#include "hover_test/hover_test_node_50hz.hpp"
#include <cmath>
#include <algorithm>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

// Observation space bounds (matching HoverEnv training config)
// obs: [v_body(3), g_body(3), target_distance_body(3)]
namespace {
  constexpr float OBS_MIN[9] = {
    -20.0f, -20.0f, -20.0f,    // v_body (body-frame velocity)
    -1.0f, -1.0f, -1.0f,       // g_body (body-frame gravity direction)
    -100.0f, -100.0f, -100.0f  // target_distance_body (body-frame distance to target point)
  };
  
  constexpr float OBS_MAX[9] = {
    20.0f, 20.0f, 20.0f,       // v_body
    1.0f, 1.0f, 1.0f,          // g_body
    100.0f, 100.0f, 100.0f     // target_distance_body
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

HoverTestNode50Hz::HoverTestNode50Hz(ros::NodeHandle& nh, ros::NodeHandle& nh_private, int drone_id)
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
  , pose_ready_(false)
  , current_vx_(0.0)
  , current_vy_(0.0)
  , current_vz_(0.0)
  , current_roll_(0.0)
  , current_pitch_(0.0)
  , current_yaw_(0.0)
  , target_x_(0.0)
  , target_y_(0.0)
  , target_z_(1.2)  // ENU: 1.2m altitude
  , current_action_(4, 0.0f)  // Initialize with zero action [thrust, omega_x, omega_y, omega_z]
  , step_counter_(0)           // Initialize step counter
{
  // Read parameters from node's private namespace (loaded from YAML in launch file)
  nh_private_.param("hover_duration", hover_duration_, 3.0);  // TRAJ control duration: 3.0s
  nh_private_.param("hover_thrust", hover_thrust_, 0.251);
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
  
  nh_private_.param("target_x", target_x_, 0.0);
  nh_private_.param("target_y", target_y_, 0.0);
  nh_private_.param("target_z", target_z_, 1.2);  // ENU: 1.2m altitude (from config)
  
  ROS_INFO("=== Hover Test Node 50Hz for Drone %d (Neural Network Control - REQUIRED) ===", drone_id_);
  ROS_INFO("Parameters:");
  ROS_INFO("  - Model path: %s", model_path_.c_str());
  ROS_INFO("  - Hover duration: %.2f s", hover_duration_);
  ROS_INFO("  - Hover thrust: %.3f (normalized [0.0-1.0])", hover_thrust_);
  ROS_INFO("  - Mode stabilization delay: %.2f s (wait for body_rate mode)", mode_stabilization_delay_);
  ROS_INFO("  - Action update period: %.3f s (%.1f Hz)", action_update_period_, 1.0/action_update_period_);
  ROS_INFO("  - Control send period: %.3f s (%.1f Hz)", control_send_period_, 1.0/control_send_period_);
  ROS_INFO("  - Target point (ENU): [%.2f, %.2f, %.2f]", target_x_, target_y_, target_z_);
  
  // Initialize neural network policy (REQUIRED)
  ROS_INFO("  - Initializing neural network policy...");
  policy_ = std::make_unique<TFLitePolicyInference50Hz>(model_path_);
  if (!policy_->is_initialized()) {
    ROS_FATAL("Failed to initialize neural network policy! Model path: %s", model_path_.c_str());
    ROS_FATAL("Neural network control is REQUIRED. Exiting...");
    ros::shutdown();
    return;
  }
  ROS_INFO("  ✅ Neural network policy initialized successfully");
  ROS_INFO("===================================================");
  
  // Publishers
  attitude_pub_ = nh_.advertise<mavros_msgs::AttitudeTarget>(
      "/mavros/setpoint_raw/attitude", 10);
  state_cmd_pub_ = nh_.advertise<std_msgs::Int32>(
      "/state/command_drone_" + std::to_string(drone_id_), 10);
  
  // Subscribers
  state_sub_ = nh_.subscribe<std_msgs::Int32>(
      "/state/state_drone_" + std::to_string(drone_id_), 10,
      &HoverTestNode50Hz::state_callback, this);
  local_pos_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>(
      "/mavros/local_position/pose", 10,
      &HoverTestNode50Hz::local_pos_callback, this);
  local_vel_sub_ = nh_.subscribe<geometry_msgs::TwistStamped>(
      "/mavros/local_position/velocity_local", 10,
      &HoverTestNode50Hz::local_vel_callback, this);
  
  // Two-timer architecture to avoid accumulated timing errors:
  // 1. Action update timer (50Hz): Neural network inference
  // 2. Control send timer (100Hz): High-frequency command transmission
  
  // Action update timer (50Hz for neural network inference)
  action_update_timer_ = nh_.createTimer(
      ros::Duration(action_update_period_),
      &HoverTestNode50Hz::action_update_callback, this);
  
  // Control send timer (100Hz for sending control commands)
  control_send_timer_ = nh_.createTimer(
      ros::Duration(control_send_period_),
      &HoverTestNode50Hz::control_send_callback, this);
}

void HoverTestNode50Hz::state_callback(const std_msgs::Int32::ConstPtr& msg)
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

  if (waiting_hover_ && !hover_started_ && !hover_completed_ &&
      (ros::Time::now() - hover_detect_time_).toSec() > 2.0) {
    ROS_INFO("Commanding state machine to TRAJ state for hover control");
    send_state_command(static_cast<int>(FsmState::TRAJ));
    hover_command_sent_ = true;
    hover_start_time_ = ros::Time::now();
  }

  // TRAJ detected - capture current position but WAIT before starting neural control
  if (state == FsmState::TRAJ && waiting_hover_ && !hover_started_) {
    hover_started_ = true;
    waiting_hover_ = false;
    hover_command_sent_ = false;
    neural_control_ready_ = false;  // Reset - will be set to true after stabilization delay
    hover_start_time_ = ros::Time::now();
    step_counter_ = 0;  // Reset step counter
    
    // Capture current position for hover
    hover_x_ = current_x_;
    hover_y_ = current_y_;
    hover_z_ = current_z_;
    hover_yaw_ = 0.0;  // Face north
    
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
    ROS_INFO("   Start pos (ENU): [%.2f, %.2f, %.2f] | Target (ENU): [%.2f, %.2f, %.2f]",
             hover_x_, hover_y_, hover_z_, target_x_, target_y_, target_z_);
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

void HoverTestNode50Hz::local_pos_callback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
  // Position (ENU frame)
  current_x_ = msg->pose.position.x;
  current_y_ = msg->pose.position.y;
  current_z_ = msg->pose.position.z;
  
  // Attitude (quaternion to Euler angles)
  double w = msg->pose.orientation.w;
  double x = msg->pose.orientation.x;
  double y = msg->pose.orientation.y;
  double z = msg->pose.orientation.z;
  
  // Roll (x-axis rotation)
  double sinr_cosp = 2.0 * (w * x + y * z);
  double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
  current_roll_ = std::atan2(sinr_cosp, cosr_cosp);
  
  // Pitch (y-axis rotation)
  double sinp = 2.0 * (w * y - z * x);
  if (std::abs(sinp) >= 1.0)
    current_pitch_ = std::copysign(M_PI / 2.0, sinp);
  else
    current_pitch_ = std::asin(sinp);
  
  // Yaw (z-axis rotation)
  double siny_cosp = 2.0 * (w * z + x * y);
  double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  current_yaw_ = std::atan2(siny_cosp, cosy_cosp);
  
  pose_ready_ = true;
}

void HoverTestNode50Hz::local_vel_callback(const geometry_msgs::TwistStamped::ConstPtr& msg)
{
  // Velocity in local frame (ENU)
  // Note: MAVROS local_position/velocity_local is in ENU frame
  // But we need body-frame velocity for neural network
  // For now, store ENU velocity and convert in get_observation()
  current_vx_ = msg->twist.linear.x;
  current_vy_ = msg->twist.linear.y;
  current_vz_ = msg->twist.linear.z;
}

// Action update callback (50Hz): Neural network inference
void HoverTestNode50Hz::action_update_callback(const ros::TimerEvent& event)
{
  if (!pose_ready_) {
    ROS_INFO_THROTTLE(5.0, "Waiting for pose data...");
    return;
  }
  
  // Only run in TRAJ state
  if (current_state_ == FsmState::TRAJ && hover_started_ && !hover_completed_) {
    // Calculate elapsed time
    double elapsed = (ros::Time::now() - hover_start_time_).toSec();
    
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
      if (!pose_ready_) {
        ROS_WARN_THROTTLE(1.0, "Waiting for pose data...");
        return;
      }
      
      ROS_INFO("✅ Mode stabilized (%.2f s) - Initializing neural control...", elapsed);
      
      // Initialize buffer with zero observation vector
      std::vector<float> initial_obs_for_reset(OBS_DIM, 0.0f);
      
      // Hovering action (normalized): [thrust in [-1,1], omega_x=0, omega_y=0, omega_z=0]
      float hover_thrust_normalized = 2.0f * static_cast<float>(hover_thrust_) - 1.0f;
      std::vector<float> hovering_action = {hover_thrust_normalized, 0.0f, 0.0f, 0.0f};
      
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
      
      // Print first action details
      double elapsed_print = (ros::Time::now() - hover_start_time_).toSec();
      
      ROS_INFO("");
      ROS_INFO("========== STEP 0 (t=%.3fs) ==========", elapsed_print);
      ROS_INFO("[RAW OBS] v_body=[%.6f, %.6f, %.6f], g_body=[%.6f, %.6f, %.6f], target_distance_body=[%.6f, %.6f, %.6f]",
               obs_raw[0], obs_raw[1], obs_raw[2],
               obs_raw[3], obs_raw[4], obs_raw[5],
               obs_raw[6], obs_raw[7], obs_raw[8]);
      ROS_INFO("[NORM OBS] v_body=[%.6f, %.6f, %.6f], g_body=[%.6f, %.6f, %.6f], target_distance_body=[%.6f, %.6f, %.6f]",
               obs_normalized[0], obs_normalized[1], obs_normalized[2],
               obs_normalized[3], obs_normalized[4], obs_normalized[5],
               obs_normalized[6], obs_normalized[7], obs_normalized[8]);
      
      float thrust_raw = first_action[0];
      float omega_x_norm = first_action[1];
      float omega_y_norm = first_action[2];
      float omega_z_norm = first_action[3];
      
      ROS_INFO("[NN RAW OUTPUT] thrust_raw=%.6f, omega_x=%.6f, omega_y=%.6f, omega_z=%.6f",
               thrust_raw, omega_x_norm, omega_y_norm, omega_z_norm);
      
      constexpr float OMEGA_MAX_X = 0.5f;
      constexpr float OMEGA_MAX_Y = 0.5f;
      constexpr float OMEGA_MAX_Z = 0.5f;
      float roll_rate = omega_x_norm * OMEGA_MAX_X;
      float pitch_rate = omega_y_norm * OMEGA_MAX_Y;
      float yaw_rate = omega_z_norm * OMEGA_MAX_Z;
      float thrust_normalized = (thrust_raw + 1.0f) * 0.5f;
      
      ROS_INFO("[DENORM OUTPUT] thrust=[0-1]:%.6f, roll_rate=%.6f rad/s, pitch_rate=%.6f rad/s, yaw_rate=%.6f rad/s",
               thrust_normalized, roll_rate, pitch_rate, yaw_rate);
      ROS_INFO("=====================================");
      
      neural_control_ready_ = true;
      ROS_INFO("🚀 Neural control active! (effective duration: %.1f s)", hover_duration_ - elapsed);
      return;
    }
    
    // Neural control is ready - update action from neural network
    if (neural_control_ready_) {
      if (!pose_ready_) {
        ROS_WARN_THROTTLE(1.0, "Pose data not ready!");
        return;
      }
      
      update_neural_action();
      
      // Log current status (throttled)
      ROS_INFO_THROTTLE(2.0,
                        "Neural control | pos(ENU)=[%.2f,%.2f,%.2f] vel=[%.2f,%.2f,%.2f] | elapsed: %.1f/%.1f s",
                        current_x_, current_y_, current_z_,
                        current_vx_, current_vy_, current_vz_,
                        elapsed, hover_duration_);
    }
  }
}

// Control send callback (100Hz): High-frequency command transmission
void HoverTestNode50Hz::control_send_callback(const ros::TimerEvent& event)
{
  if (!pose_ready_) {
    return;
  }
  
  // Only run in TRAJ state
  if (current_state_ == FsmState::TRAJ && hover_started_ && !hover_completed_) {
    // Publish current action at high frequency
    publish_current_action();
  }
}

// Get observation vector for neural network (9D)
// Based on HoverEnv observation space - all in body frame (NED/FRD)
// Observation composition:
// 1. 机体系速度 (3) - quad velocity in body frame (FRD)
// 2. 机体系重力方向 (3) - gravity direction in body frame (FRD)
// 3. 机体系到目标点距离 (3) - distance to target point in body frame (FRD)
// 
// Coordinate System:
// - Neural network trained with NED world frame (North-East-Down) and FRD body frame (Forward-Right-Down)
// - MAVROS provides ENU world frame (East-North-Up) and FLU body frame (Forward-Left-Up)
// - This function converts ENU -> NED -> FRD body frame for neural network input
std::vector<float> HoverTestNode50Hz::get_observation()
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
  
  // 3. Body-frame distance to target (target_distance_body_frd = R^T * (target - current)_ned)
  // Vector from current position to target position
  float target_distance_ned[3] = {
    static_cast<float>(target_ned_x - pos_ned_x),
    static_cast<float>(target_ned_y - pos_ned_y),
    static_cast<float>(target_ned_z - pos_ned_z)
  };
  obs[6] = R[0][0] * target_distance_ned[0] + R[1][0] * target_distance_ned[1] + R[2][0] * target_distance_ned[2];
  obs[7] = R[0][1] * target_distance_ned[0] + R[1][1] * target_distance_ned[1] + R[2][1] * target_distance_ned[2];
  obs[8] = R[0][2] * target_distance_ned[0] + R[1][2] * target_distance_ned[1] + R[2][2] * target_distance_ned[2];
  
  return obs;
}

// Update neural network action (called at 50Hz)
void HoverTestNode50Hz::update_neural_action()
{
  // Increment step counter
  step_counter_++;
  
  // Get current observation (9D) - RAW (before normalization)
  std::vector<float> obs_raw = get_observation();
  
  // Make a copy for normalization
  std::vector<float> obs_normalized = obs_raw;
  
  // Normalize observation to [-1, 1] range (CRITICAL: must match training!)
  normalize_observation(obs_normalized);
  
  // Get action from neural network
  // Note: With separate 50Hz timer, we don't need internal action repeat anymore
  std::vector<float> action = policy_->get_action(obs_normalized);
  
  // Update current action (thread-safe)
  {
    std::lock_guard<std::mutex> lock(action_mutex_);
    current_action_ = action;
  }
  
  // Calculate elapsed time
  double elapsed = (ros::Time::now() - hover_start_time_).toSec();
  
  // ==================== Full print: step number, raw observation, network output, denormalized output ====================
  
  // 1. Step number
  ROS_INFO("");  // Empty line separator
  ROS_INFO("========== STEP %d (t=%.3fs) ==========", step_counter_, elapsed);
  
  // 2. Raw observation (not normalized)
  ROS_INFO("[RAW OBS] v_body=[%.6f, %.6f, %.6f], g_body=[%.6f, %.6f, %.6f], target_distance_body=[%.6f, %.6f, %.6f]",
           obs_raw[0], obs_raw[1], obs_raw[2],    // v_body
           obs_raw[3], obs_raw[4], obs_raw[5],    // g_body
           obs_raw[6], obs_raw[7], obs_raw[8]);   // target_distance_body
  
  // 3. Normalized observation (input to neural network)
  ROS_INFO("[NORM OBS] v_body=[%.6f, %.6f, %.6f], g_body=[%.6f, %.6f, %.6f], target_distance_body=[%.6f, %.6f, %.6f]",
           obs_normalized[0], obs_normalized[1], obs_normalized[2],    // v_body
           obs_normalized[3], obs_normalized[4], obs_normalized[5],    // g_body
           obs_normalized[6], obs_normalized[7], obs_normalized[8]);   // target_distance_body
  
  // 4. Network raw output (tanh output in [-1, 1] range)
  float thrust_raw = action[0];
  float omega_x_norm = action[1];
  float omega_y_norm = action[2];
  float omega_z_norm = action[3];
  
  ROS_INFO("[NN RAW OUTPUT] thrust_raw=%.6f, omega_x=%.6f, omega_y=%.6f, omega_z=%.6f",
           thrust_raw, omega_x_norm, omega_y_norm, omega_z_norm);
  
  // 5. Denormalized output (converted to physical quantities)
  constexpr float OMEGA_MAX_X = 0.5f;  // rad/s
  constexpr float OMEGA_MAX_Y = 0.5f;  // rad/s
  constexpr float OMEGA_MAX_Z = 0.5f;  // rad/s
  float roll_rate = omega_x_norm * OMEGA_MAX_X;
  float pitch_rate = omega_y_norm * OMEGA_MAX_Y;
  float yaw_rate = omega_z_norm * OMEGA_MAX_Z;
  float thrust_normalized = (thrust_raw + 1.0f) * 0.5f;  // Map [-1,1] to [0,1]
  
  ROS_INFO("[DENORM OUTPUT] thrust=[0-1]:%.6f, roll_rate=%.6f rad/s, pitch_rate=%.6f rad/s, yaw_rate=%.6f rad/s",
           thrust_normalized, roll_rate, pitch_rate, yaw_rate);
  
  ROS_INFO("=====================================");
}

// Publish current action (called at 100Hz)
void HoverTestNode50Hz::publish_current_action()
{
  // Read current action (thread-safe)
  std::vector<float> action;
  {
    std::lock_guard<std::mutex> lock(action_mutex_);
    action = current_action_;
  }
  
  // Create AttitudeTarget message for MAVROS
  mavros_msgs::AttitudeTarget msg;
  
  // Action output: [thrust, omega_x, omega_y, omega_z]
  // All in range [-1, 1] from tanh activation
  float thrust_raw = action[0];
  float omega_x_norm = action[1];
  float omega_y_norm = action[2];
  float omega_z_norm = action[3];
  
  // Denormalize angular rates: [-1, 1] -> [-omega_max, omega_max]
  constexpr float OMEGA_MAX_X = 0.5f;  // Roll rate max [rad/s]
  constexpr float OMEGA_MAX_Y = 0.5f;  // Pitch rate max [rad/s]
  constexpr float OMEGA_MAX_Z = 0.5f;  // Yaw rate max [rad/s]
  
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
  msg.body_rate.x = roll_rate_frd;      // Roll rate: no change (Forward axis)
  msg.body_rate.y = -pitch_rate_frd;    // Pitch rate: flip sign (Right->Left)
  msg.body_rate.z = -yaw_rate_frd;      // Yaw rate: flip sign (Down->Up)
  
  // Thrust mapping: [-1, 1] -> [0, 1]
  float thrust_normalized = (thrust_raw + 1.0f) * 0.5f;
  msg.thrust = thrust_normalized;
  
  // Use body rate control mode (ignore attitude, use body rates and thrust)
  msg.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ATTITUDE;
  
  // Header
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "base_link";
  
  // Publish
  attitude_pub_.publish(msg);
}

void HoverTestNode50Hz::send_state_command(int state)
{
  std_msgs::Int32 msg;
  msg.data = state;
  state_cmd_pub_.publish(msg);
  
  ROS_INFO("Sent state command: %d", state);
}

#include "track_test/target_publisher_node.hpp"
#include <cmath>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

TargetPublisherNode::TargetPublisherNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
  : nh_(nh)
  , nh_private_(nh_private)
  , trajectory_started_(false)
  , drone_id_(0)
  , current_state_(FsmState::INIT)
  , in_traj_state_(false)
  , dshape_current_arc_length_(0.0)
{
  // Declare and get parameters
  nh_private_.param("drone_id", drone_id_, 0);
  
  // Get trajectory type
  nh_private_.param("trajectory_type", trajectory_type_str_, std::string("circle"));
  if (trajectory_type_str_ == "circle") {
    trajectory_type_ = TrajectoryType::CIRCLE;
  } else if (trajectory_type_str_ == "figure8") {
    trajectory_type_ = TrajectoryType::FIGURE8;
  } else if (trajectory_type_str_ == "d_shape") {
    trajectory_type_ = TrajectoryType::D_SHAPE;
  } else {
    ROS_WARN("Unknown trajectory_type '%s', defaulting to 'circle'", trajectory_type_str_.c_str());
    trajectory_type_ = TrajectoryType::CIRCLE;
    trajectory_type_str_ = "circle";
  }
  
  // Get trajectory parameters (new unified parameters)
  nh_private_.param("trajectory_size", trajectory_size_, 2.0);
  nh_private_.param("trajectory_duration", trajectory_duration_, 20.0);
  nh_private_.param("trajectory_times", trajectory_times_, 1);
  nh_private_.param("trajectory_center_x", trajectory_center_x_, 0.0);
  nh_private_.param("trajectory_center_y", trajectory_center_y_, 0.0);
  nh_private_.param("trajectory_center_z", trajectory_center_z_, 1.2);
  
  // Legacy parameters for backward compatibility
  nh_private_.param("circle_radius", circle_radius_, trajectory_size_);
  nh_private_.param("circle_duration", circle_duration_, trajectory_duration_);
  nh_private_.param("circle_times", circle_times_, trajectory_times_);
  nh_private_.param("circle_center_x", circle_center_x_, trajectory_center_x_);
  nh_private_.param("circle_center_y", circle_center_y_, trajectory_center_y_);
  nh_private_.param("circle_center_z", circle_center_z_, trajectory_center_z_);
  
  // Use legacy parameters if trajectory parameters not explicitly set
  if (!nh_private_.hasParam("trajectory_size")) {
    trajectory_size_ = circle_radius_;
  }
  if (!nh_private_.hasParam("trajectory_duration")) {
    trajectory_duration_ = circle_duration_;
  }
  if (!nh_private_.hasParam("trajectory_times")) {
    trajectory_times_ = circle_times_;
  }
  if (!nh_private_.hasParam("trajectory_center_x")) {
    trajectory_center_x_ = circle_center_x_;
  }
  if (!nh_private_.hasParam("trajectory_center_y")) {
    trajectory_center_y_ = circle_center_y_;
  }
  if (!nh_private_.hasParam("trajectory_center_z")) {
    trajectory_center_z_ = circle_center_z_;
  }
  
  nh_private_.param("circle_init_phase", circle_init_phase_, 0.0);
  nh_private_.param("ramp_up_time", ramp_up_time_, 3.0);
  nh_private_.param("ramp_down_time", ramp_down_time_, 3.0);
  nh_private_.param("stationary_time", stationary_time_, 3.0);
  
  // Velocity-based parameters for D-shape and Figure-8 trajectories
  nh_private_.param("max_linear_velocity", max_linear_velocity_, 1.5);
  nh_private_.param("linear_acceleration", linear_acceleration_, 0.4);
  
  // Auto-calculate trajectory parameters for D-shape
  if (trajectory_type_ == TrajectoryType::D_SHAPE) {
    // Pre-calculate arc lengths of each Bezier segment
    calculate_dshape_segment_lengths();
    
    // Use actual total arc length for trajectory planning
    double path_length = dshape_total_arc_length_;
    
    // Calculate ramp times from acceleration
    ramp_up_time_ = max_linear_velocity_ / linear_acceleration_;
    ramp_down_time_ = max_linear_velocity_ / linear_acceleration_;
    
    // Calculate distances during acceleration phases
    double accel_distance = 0.5 * linear_acceleration_ * ramp_up_time_ * ramp_up_time_;
    double decel_distance = 0.5 * linear_acceleration_ * ramp_down_time_ * ramp_down_time_;
    
    // Remaining distance at constant velocity
    double const_distance = path_length - accel_distance - decel_distance;
    
    if (const_distance < 0) {
      // If trajectory too short to reach max velocity, adjust
      ROS_WARN("Trajectory too short to reach max velocity, adjusting parameters");
      double t_total = 2.0 * std::sqrt(path_length / linear_acceleration_);
      ramp_up_time_ = t_total / 2.0;
      ramp_down_time_ = t_total / 2.0;
      trajectory_duration_ = t_total;
    } else {
      // Calculate constant velocity time
      double const_time = const_distance / max_linear_velocity_;
      trajectory_duration_ = ramp_up_time_ + const_time + ramp_down_time_;
    }
    
    ROS_INFO("[D-Shape Arc-Length] total_length=%.2fm, duration=%.2fs, ramp_time=%.2fs, max_vel=%.2fm/s",
             path_length, trajectory_duration_, ramp_up_time_, max_linear_velocity_);
    ROS_INFO("[D-Shape Segments] lengths=[%.2f, %.2f, %.2f, %.2f]m",
             dshape_segment_lengths_[0], dshape_segment_lengths_[1], 
             dshape_segment_lengths_[2], dshape_segment_lengths_[3]);
  }
  
  // Auto-calculate trajectory parameters for Figure-8
  if (trajectory_type_ == TrajectoryType::FIGURE8) {
    // Estimate Figure-8 path length (approximate)
    // For a lemniscate: approximate arc length ≈ 5.244 * a (where a is the scale parameter)
    double path_length = 5.244 * trajectory_size_;
    
    // Calculate ramp times from acceleration
    ramp_up_time_ = max_linear_velocity_ / linear_acceleration_;
    ramp_down_time_ = max_linear_velocity_ / linear_acceleration_;
    
    // Calculate distances during acceleration phases
    double accel_distance = 0.5 * linear_acceleration_ * ramp_up_time_ * ramp_up_time_;
    double decel_distance = 0.5 * linear_acceleration_ * ramp_down_time_ * ramp_down_time_;
    
    // Remaining distance at constant velocity
    double const_distance = path_length - accel_distance - decel_distance;
    
    if (const_distance < 0) {
      // If trajectory too short to reach max velocity, adjust
      ROS_WARN("Trajectory too short to reach max velocity, adjusting parameters");
      double t_total = 2.0 * std::sqrt(path_length / linear_acceleration_);
      ramp_up_time_ = t_total / 2.0;
      ramp_down_time_ = t_total / 2.0;
      trajectory_duration_ = t_total;
    } else {
      // Calculate constant velocity time
      double const_time = const_distance / max_linear_velocity_;
      trajectory_duration_ = ramp_up_time_ + const_time + ramp_down_time_;
    }
    
    ROS_INFO("[Figure-8 Velocity-Based] path_length=%.2fm, duration=%.2fs, ramp_time=%.2fs, max_vel=%.2fm/s",
             path_length, trajectory_duration_, ramp_up_time_, max_linear_velocity_);
  }
  
  nh_private_.param("timer_period", timer_period_, 0.02);
  nh_private_.param("max_speed", max_speed_, -1.0);
  use_max_speed_ = (max_speed_ > 0.0);
  
  // Get use_sim_time parameter (ROS standard parameter)
  nh_.param("use_sim_time", use_sim_time_, false);
  
  // Parameter validation
  if (trajectory_times_ < 1) {
    ROS_WARN("trajectory_times must be >= 1, setting to 1");
    trajectory_times_ = 1;
  }
  if (circle_times_ < 1) {
    circle_times_ = 1;
  }
  
  // Calculate trajectory parameters
  effective_duration_ = calculate_effective_duration();
  max_angular_vel_ = 2.0 * M_PI / effective_duration_;
  angular_acceleration_ = max_angular_vel_ / ramp_up_time_;
  
  double theta_ramp_up = 0.5 * max_angular_vel_ * ramp_up_time_;
  double theta_ramp_down = 0.5 * max_angular_vel_ * ramp_down_time_;
  double theta_ramps_total = theta_ramp_up + theta_ramp_down;
  
  // Total angular displacement required for N cycles
  double theta_required = trajectory_times_ * 2.0 * M_PI;
  // Angular displacement required for constant velocity phase
  double theta_constant = theta_required - theta_ramps_total;
  // Time required for constant velocity phase
  total_constant_duration_ = theta_constant / max_angular_vel_;
  
  ROS_INFO("=== Target Publisher Node ===");
  ROS_INFO("Trajectory type: %s", trajectory_type_str_.c_str());
  ROS_INFO("Trajectory size: %.2f m", trajectory_size_);
  ROS_INFO("Trajectory center: [%.2f, %.2f, %.2f] m (ENU)", 
              trajectory_center_x_, trajectory_center_y_, trajectory_center_z_);
  ROS_INFO("Initial stationary time: %.2f s", stationary_time_);
  ROS_INFO("Number of cycles: %d", trajectory_times_);
  ROS_INFO("Motion duration: %.2f s", 
              ramp_up_time_ + total_constant_duration_ + ramp_down_time_);
  ROS_INFO("Total duration: %.2f s", 
              stationary_time_ + ramp_up_time_ + total_constant_duration_ + ramp_down_time_);
  ROS_INFO("Publish frequency: %.0f Hz", 1.0/timer_period_);
  ROS_INFO("Clock mode: %s", 
              use_sim_time_ ? "SIM_TIME (ROS clock)" : "SYSTEM_TIME (steady_clock)");
  ROS_INFO("Drone ID: %d", drone_id_);
  ROS_INFO("State topic: /state/state_drone_%d", drone_id_);
  ROS_INFO("Target publishing will start when entering TRAJ state");
  
  // Create publishers
  position_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/target/position", 10);
  velocity_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("/target/velocity", 10);
  
  // Create state subscriber (listen to state machine state)
  std::string state_topic = "/state/state_drone_" + std::to_string(drone_id_);
  state_sub_ = nh_.subscribe<std_msgs::Int32>(
    state_topic,
    10,
    &TargetPublisherNode::state_callback,
    this);
  
  ROS_INFO("Subscribed to state topic: %s", state_topic.c_str());
  
  // Create timer
  timer_ = nh_.createTimer(ros::Duration(timer_period_), 
                          &TargetPublisherNode::timer_callback, 
                          this);
  
  // Note: Do not initialize start_time_ in constructor
  // Initialize in timer_callback after entering TRAJ state to ensure clock synchronization
  // This avoids time errors when use_sim_time=true but clock topic hasn't been subscribed yet
  trajectory_started_ = false;  // Initialize as false, wait for TRAJ state entry
  
  ROS_INFO("Target trajectory generation will start when entering TRAJ state");
}

double TargetPublisherNode::calculate_effective_duration()
{
  if (!use_max_speed_) {
    return trajectory_duration_;
  }
  
  // Estimate path length based on trajectory type
  double path_length = 0.0;
  if (trajectory_type_ == TrajectoryType::CIRCLE) {
    path_length = 2.0 * M_PI * trajectory_size_;  // Circumference
  } else if (trajectory_type_ == TrajectoryType::FIGURE8) {
    path_length = 4.0 * M_PI * trajectory_size_ * 0.7;  // Approximate length of figure-8
  } else if (trajectory_type_ == TrajectoryType::D_SHAPE) {
    // D-shape with Bezier curves (4 segments)
    // Approximate arc length for Bezier-based D-shape
    double a = trajectory_size_;
    path_length = 5.5 * a;  // Empirical approximation for 4-segment Bezier D-shape
  }
  
  double min_duration = path_length / max_speed_;
  
  if (min_duration > trajectory_duration_) {
    return min_duration;
  }
  
  return trajectory_duration_;
}

void TargetPublisherNode::state_callback(const std_msgs::Int32::ConstPtr& msg)
{
  auto state = static_cast<FsmState>(msg->data);
  current_state_ = state;
  
  // Detect entry into TRAJ state
  if (state == FsmState::TRAJ && !in_traj_state_) {
    in_traj_state_ = true;
    ROS_INFO("✅ Entered TRAJ state - Target trajectory publishing will start");
    // Don't initialize time here, initialize in timer_callback to ensure clock synchronization
  }
  
  // Detect exit from TRAJ state
  if (in_traj_state_ && state != FsmState::TRAJ) {
    in_traj_state_ = false;
    trajectory_started_ = false;  // Reset, so it re-initializes next time entering TRAJ state
    ROS_INFO("Left TRAJ state - Target trajectory publishing stopped");
  }
}

double TargetPublisherNode::calculate_theta_at_time(double t)
{
  double theta = 0.0;
  double omega_max = max_angular_vel_;
  double alpha = angular_acceleration_;
  double alpha_down = omega_max / ramp_down_time_; 
  double t_up = ramp_up_time_;
  double t_const = total_constant_duration_;  
  double t_down = ramp_down_time_;
  
  // Phase 1: Acceleration
  if (t <= t_up) {
    theta = 0.5 * alpha * t * t;
  }
  // Phase 2: Constant velocity
  else if (t <= t_up + t_const) {
    double theta_at_t_up = 0.5 * alpha * t_up * t_up;
    double dt = t - t_up;
    theta = theta_at_t_up + omega_max * dt;
  }
  // Phase 3: Deceleration
  else if (t <= t_up + t_const + t_down) {
    double theta_at_t_up = 0.5 * alpha * t_up * t_up;
    double theta_at_start_down = theta_at_t_up + omega_max * t_const;
    
    double t_start_down = t_up + t_const;
    double dt = t - t_start_down;
    theta = theta_at_start_down + omega_max * dt - 0.5 * alpha_down * dt * dt;
  }
  // Phase 4: Maintain final position
  else {
    double theta_at_t_up = 0.5 * alpha * t_up * t_up;
    double theta_at_start_down = theta_at_t_up + omega_max * t_const;
    theta = theta_at_start_down + omega_max * t_down - 0.5 * alpha_down * t_down * t_down;
  }
  
  return theta;
}

double TargetPublisherNode::calculate_angular_velocity_at_time(double t)
{
  double omega_max = max_angular_vel_;
  double alpha = angular_acceleration_;
  double alpha_down = omega_max / ramp_down_time_;
  double t_up = ramp_up_time_;
  double t_const = total_constant_duration_;
  double t_down = ramp_down_time_;
  double t_start_down = t_up + t_const;
  
  double current_omega = 0.0;

  if (t <= t_up) {
    // Acceleration phase
    current_omega = alpha * t;
  }
  else if (t <= t_start_down) {
    // Constant velocity phase
    current_omega = omega_max;
  }
  else if (t <= t_start_down + t_down) {
    // Deceleration phase
    double dt_down = t - t_start_down;
    current_omega = omega_max - alpha_down * dt_down;
    current_omega = std::max(0.0, current_omega);
  }
  else {
    // Motion complete
    current_omega = 0.0;
  }
  
  return current_omega;
}

void TargetPublisherNode::timer_callback(const ros::TimerEvent& event)
{
  // When not in TRAJ state, publish initial stationary position (avoid deadlock)
  if (!in_traj_state_) {
    // Publish initial stationary position to allow track_test_node to detect target and enter TRAJ state
    // Position should be at trajectory starting point (1m in front of drone at trajectory_center)
    double x, y, z;
    
    if (trajectory_type_ == TrajectoryType::CIRCLE) {
      // Circle: use circle-specific center and init_phase
      x = circle_center_x_ + circle_radius_ * std::cos(circle_init_phase_);
      y = circle_center_y_ + circle_radius_ * std::sin(circle_init_phase_);
      z = circle_center_z_;
    } else if (trajectory_type_ == TrajectoryType::FIGURE8) {
      // Figure-8: offset +1m from trajectory_center
      x = trajectory_center_x_ + 1.0;
      y = trajectory_center_y_;
      z = trajectory_center_z_;
    } else if (trajectory_type_ == TrajectoryType::D_SHAPE) {
      // D-shape: offset +1m from trajectory_center
      x = trajectory_center_x_ + 1.0;
      y = trajectory_center_y_;
      z = trajectory_center_z_;
    }
    
    // Zero velocity
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    
    // Publish position and velocity
    publish_target_position(x, y, z);
    publish_target_velocity(vx, vy, vz);
    
    // Periodic log
    ROS_INFO_THROTTLE(2.0, "[WAITING] Publishing initial stationary target at [%.2f, %.2f, %.2f] (waiting for TRAJ state)",
                         x, y, z);
    return;
  }
  
  // Initialize start time on first callback after entering TRAJ state (ensure clock synchronization)
  if (!trajectory_started_) {
    if (use_sim_time_) {
      // SITL mode: Use ROS clock (simulation time)
      start_time_ = ros::Time::now();
      ROS_INFO("Using ROS clock (sim_time) for trajectory timing");
      ROS_INFO("start_time_ = %.9f seconds", start_time_.toSec());
    } else {
      // Onboard mode: Use system clock (steady_clock)
      start_time_system_ = std::chrono::steady_clock::now();
      ROS_INFO("Using system clock (steady_clock) for trajectory timing");
    }
    trajectory_started_ = true;
    ROS_INFO("✅ Target trajectory generation started - starting timer (stationary for %.1fs, then circular motion)", stationary_time_);
    return;  // First callback only initializes, doesn't generate trajectory
  }
  
  double elapsed = 0.0;
  
  if (use_sim_time_) {
    // SITL mode: Use ROS clock (simulation time)
    elapsed = (ros::Time::now() - start_time_).toSec();
    
    // Handle negative time due to clock jitter
    if (elapsed < -0.01) {
      elapsed = 0.0;
    }
    elapsed = std::max(0.0, elapsed);
  } else {
    // Onboard mode: Use system clock (steady_clock)
    auto current_time_system = std::chrono::steady_clock::now();
    auto elapsed_duration = std::chrono::duration_cast<std::chrono::duration<double>>(
      current_time_system - start_time_system_);
    elapsed = elapsed_duration.count();
  }
  
  // Generate trajectory based on type
  if (trajectory_type_ == TrajectoryType::CIRCLE) {
  generate_circular_trajectory(elapsed);
  } else if (trajectory_type_ == TrajectoryType::FIGURE8) {
    generate_figure8_trajectory(elapsed);
  } else if (trajectory_type_ == TrajectoryType::D_SHAPE) {
    generate_dshape_trajectory(elapsed);
  }
}

void TargetPublisherNode::generate_circular_trajectory(double t)
{
  double x, y, z, vx, vy, vz;
  
  // Use circle-specific center (may be different from trajectory_center to position circle correctly)
  double circle_center_x = circle_center_x_;
  double circle_center_y = circle_center_y_;
  double circle_center_z = circle_center_z_;
  double radius = circle_radius_;
  
  // Phase 1: Stationary period - stay at starting point on circle
  if (t < stationary_time_) {
    // Initial position: starting point on circle (init_phase determines position on circle)
    x = circle_center_x + radius * std::cos(circle_init_phase_);
    y = circle_center_y + radius * std::sin(circle_init_phase_);
    z = circle_center_z;
    
    // Zero velocity during stationary period
    vx = 0.0;
    vy = 0.0;
    vz = 0.0;
    
    // Publish position and velocity
    publish_target_position(x, y, z);
    publish_target_velocity(vx, vy, vz);
    
    // Debug log
    ROS_INFO_THROTTLE(2.0, "[STATIONARY] t=%.1f/%.1fs | position=[%.2f, %.2f, %.2f] | velocity=0.00 m/s",
                         t, stationary_time_, x, y, z);
    return;
  }
  
  // Phase 2: Circular motion - calculate motion time offset by stationary period
  double motion_time = t - stationary_time_;
  
  // Calculate angular position at motion time
  double theta = calculate_theta_at_time(motion_time);
  double current_cycles = theta / (2.0 * M_PI);  // Current progress in cycles
  
  // Check if trajectory is complete (either by cycles or by time)
  double total_motion_time = ramp_up_time_ + total_constant_duration_ + ramp_down_time_;
  bool trajectory_complete = (current_cycles >= trajectory_times_) || (motion_time >= total_motion_time);
  
  if (trajectory_complete) {
    // Hold at final position with zero velocity
    double final_theta = trajectory_times_ * 2.0 * M_PI + circle_init_phase_;
    x = circle_center_x + radius * std::cos(final_theta);
    y = circle_center_y + radius * std::sin(final_theta);
    z = circle_center_z;
    vx = 0.0;
    vy = 0.0;
    vz = 0.0;
    
    publish_target_position(x, y, z);
    publish_target_velocity(vx, vy, vz);
    
    ROS_INFO_THROTTLE(2.0, "[CIRCLE-COMPLETE] t=%.1fs | cycles=%.2f/%d | position=[%.2f, %.2f, %.2f] | HOLDING",
                         t, current_cycles, trajectory_times_, x, y, z);
    return;
  }
  
  // Calculate angular velocity at motion time
  double current_omega = calculate_angular_velocity_at_time(motion_time);
  double theta_with_phase = theta + circle_init_phase_;
  
  // Position on circle
  x = circle_center_x + radius * std::cos(theta_with_phase);
  y = circle_center_y + radius * std::sin(theta_with_phase);
  z = circle_center_z;
  
  // Linear velocity in tangential direction
  double v_linear = current_omega * radius;
  vx = -v_linear * std::sin(theta_with_phase);
  vy =  v_linear * std::cos(theta_with_phase);
  vz = 0.0;
  
  // Publish position and velocity
  publish_target_position(x, y, z);
  publish_target_velocity(vx, vy, vz);
  
  // Debug log
  double t_up = ramp_up_time_;
  double t_const = total_constant_duration_;
  std::string phase;
  
  if (motion_time <= t_up) {
    phase = "ACCELERATING";
  } else if (motion_time <= t_up + t_const) {
    phase = "CONSTANT";
  } else {
    phase = "DECELERATING";
  }
  
  ROS_INFO_THROTTLE(2.0, "[CIRCLE-%s] t=%.1fs (motion: %.1fs) | cycle=%.2f/%d | position=[%.2f, %.2f, %.2f] | velocity=%.2f m/s",
                       phase.c_str(), t, motion_time, current_cycles, trajectory_times_, 
                       x, y, z, v_linear);
}

void TargetPublisherNode::publish_target_position(double x, double y, double z)
{
  geometry_msgs::PointStamped msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "map";  // Or "world", "odom", etc., depending on actual coordinate system
  msg.point.x = x;
  msg.point.y = y;
  msg.point.z = z;
  
  position_pub_.publish(msg);
}

void TargetPublisherNode::publish_target_velocity(double vx, double vy, double vz)
{
  geometry_msgs::TwistStamped msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "map";
  msg.twist.linear.x = vx;
  msg.twist.linear.y = vy;
  msg.twist.linear.z = vz;
  msg.twist.angular.x = 0.0;
  msg.twist.angular.y = 0.0;
  msg.twist.angular.z = 0.0;
  
  velocity_pub_.publish(msg);
}

double TargetPublisherNode::calculate_normalized_parameter_at_time(double t)
{
  // This calculates a normalized parameter from 0 to 1 over the trajectory
  // Similar to calculate_theta_at_time but returns normalized value
  double param = 0.0;
  double param_max = 1.0;  // Maximum parameter value (1 complete cycle)
  double alpha = param_max / (ramp_up_time_ * ramp_up_time_);  // Acceleration coefficient
  double alpha_down = param_max / (ramp_down_time_ * ramp_down_time_);  // Deceleration coefficient
  double param_vel_max = param_max / ramp_up_time_;  // Maximum parameter velocity during ramp up
  
  // Recalculate for multi-cycle trajectories
  double total_param_required = trajectory_times_;
  param_max = total_param_required / (ramp_up_time_ + total_constant_duration_ + ramp_down_time_) * ramp_up_time_;
  alpha = param_max / ramp_up_time_;
  param_vel_max = alpha * ramp_up_time_;
  
  double t_up = ramp_up_time_;
  double t_const = total_constant_duration_;  
  double t_down = ramp_down_time_;
  
  // Phase 1: Acceleration
  if (t <= t_up) {
    param = 0.5 * alpha * t * t / t_up;
  }
  // Phase 2: Constant velocity
  else if (t <= t_up + t_const) {
    double param_at_t_up = 0.5 * alpha * t_up;
    double dt = t - t_up;
    param = param_at_t_up + param_vel_max * dt;
  }
  // Phase 3: Deceleration
  else if (t <= t_up + t_const + t_down) {
    double param_at_t_up = 0.5 * alpha * t_up;
    double param_at_start_down = param_at_t_up + param_vel_max * t_const;
    
    double t_start_down = t_up + t_const;
    double dt = t - t_start_down;
    double alpha_down_actual = param_vel_max / t_down;
    param = param_at_start_down + param_vel_max * dt - 0.5 * alpha_down_actual * dt * dt;
  }
  // Phase 4: Maintain final position
  else {
    param = total_param_required;
  }
  
  return param;
}

double TargetPublisherNode::calculate_parameter_velocity_at_time(double t)
{
  // Calculate the rate of change of the normalized parameter
  double param_max = 1.0;
  double alpha = param_max / ramp_up_time_;
  double param_vel_max = alpha * ramp_up_time_;
  
  double t_up = ramp_up_time_;
  double t_const = total_constant_duration_;
  double t_down = ramp_down_time_;
  double t_start_down = t_up + t_const;
  
  double current_param_vel = 0.0;

  if (t <= t_up) {
    // Acceleration phase
    current_param_vel = alpha * t;
  }
  else if (t <= t_start_down) {
    // Constant velocity phase
    current_param_vel = param_vel_max;
  }
  else if (t <= t_start_down + t_down) {
    // Deceleration phase
    double dt_down = t - t_start_down;
    double alpha_down_actual = param_vel_max / t_down;
    current_param_vel = param_vel_max - alpha_down_actual * dt_down;
    current_param_vel = std::max(0.0, current_param_vel);
  }
  else {
    // Motion complete
    current_param_vel = 0.0;
  }
  
  return current_param_vel;
}

void TargetPublisherNode::generate_figure8_trajectory(double t)
{
  double x, y, z, vx, vy, vz;
  
  // Phase 1: Stationary period - stay at initial position (1m in front of center)
  if (t < stationary_time_) {
    // Initial position: 1m in front of trajectory center (along +x direction)
    x = trajectory_center_x_ + 1.0;
    y = trajectory_center_y_;
    z = trajectory_center_z_;
    
    // Zero velocity during stationary period
    vx = 0.0;
    vy = 0.0;
    vz = 0.0;
    
    // Publish position and velocity
    publish_target_position(x, y, z);
    publish_target_velocity(vx, vy, vz);
    
    // Debug log
    ROS_INFO_THROTTLE(2.0, "[STATIONARY] t=%.1f/%.1fs | position=[%.2f, %.2f, %.2f] | velocity=0.00 m/s",
                         t, stationary_time_, x, y, z);
    return;
  }
  
  // Phase 2: Figure-8 motion
  double motion_time = t - stationary_time_;
  
  // Calculate normalized parameter (0 to trajectory_times_)
  double param = calculate_normalized_parameter_at_time(motion_time);
  double param_vel = calculate_parameter_velocity_at_time(motion_time);
  
  // Check if trajectory is complete (only by time, not param, to allow smooth deceleration)
  double total_motion_time = ramp_up_time_ + total_constant_duration_ + ramp_down_time_;
  bool trajectory_complete = (motion_time >= total_motion_time);
  
  if (trajectory_complete) {
    // Hold at final position (should be back at start) with zero velocity
    double final_theta = trajectory_times_ * 2.0 * M_PI;
    double a = trajectory_size_;
    double x_local = a * std::sin(final_theta) * std::cos(final_theta) + 1.0;
    double y_local = a * std::sin(final_theta);
    
    x = trajectory_center_x_ + x_local;
    y = trajectory_center_y_ + y_local;
    z = trajectory_center_z_;
    vx = 0.0;
    vy = 0.0;
    vz = 0.0;
    
    publish_target_position(x, y, z);
    publish_target_velocity(vx, vy, vz);
    
    ROS_INFO_THROTTLE(2.0, "[FIGURE8-COMPLETE] t=%.1fs | cycle=%.2f/%d | position=[%.2f, %.2f, %.2f] | HOLDING",
                         t, param, trajectory_times_, x, y, z);
    return;
  }
  
  // Convert parameter to angle for figure-8 (one cycle = 2*pi)
  double theta = param * 2.0 * M_PI;
  double theta_dot = param_vel * 2.0 * M_PI;
  
  // Figure-8 parametric equations (Lemniscate) with offset to start at 1m ahead
  // Original: x(t) = a * sin(theta) * cos(theta), y(t) = a * sin(theta)
  // At theta=0: x=0, y=0 (center)
  // We want starting point at [1, 0] relative to trajectory_center
  // Solution: Add offset of [1, 0] to position
  
  double a = trajectory_size_;  // Scale parameter
  
  // Position with offset to start 1m ahead of center
  double x_local = a * std::sin(theta) * std::cos(theta) + 1.0;  // +1.0 offset in x
  double y_local = a * std::sin(theta);
  
  x = trajectory_center_x_ + x_local;
  y = trajectory_center_y_ + y_local;
  z = trajectory_center_z_;
  
  // Velocity (derivatives of position, offset doesn't affect velocity)
  // dx/dt = a * cos(2*theta) * theta_dot
  // dy/dt = a * cos(theta) * theta_dot
  double vx_local = a * std::cos(2.0 * theta) * theta_dot;
  double vy_local = a * std::cos(theta) * theta_dot;
  
  // Normalize velocity to ensure it doesn't exceed max_linear_velocity (consistent with D-shape)
  double tangent_magnitude = std::sqrt(vx_local * vx_local + vy_local * vy_local);
  if (tangent_magnitude > 1e-6) {
    // Calculate current physical velocity based on acceleration profile
    double current_velocity = 0.0;
    if (motion_time <= ramp_up_time_) {
      current_velocity = linear_acceleration_ * motion_time;
    } else if (motion_time <= ramp_up_time_ + total_constant_duration_) {
      current_velocity = max_linear_velocity_;
    } else if (motion_time <= ramp_up_time_ + total_constant_duration_ + ramp_down_time_) {
      double t_in_decel = motion_time - ramp_up_time_ - total_constant_duration_;
      current_velocity = max_linear_velocity_ - linear_acceleration_ * t_in_decel;
      current_velocity = std::max(0.0, current_velocity);
    }
    // Normalize tangent vector and multiply by actual physical velocity
    vx = (vx_local / tangent_magnitude) * current_velocity;
    vy = (vy_local / tangent_magnitude) * current_velocity;
  } else {
    vx = 0.0;
    vy = 0.0;
  }
  vz = 0.0;
  
  // Publish position and velocity
  publish_target_position(x, y, z);
  publish_target_velocity(vx, vy, vz);
  
  // Debug log
  double v_linear = std::sqrt(vx*vx + vy*vy);
  double t_up = ramp_up_time_;
  double t_const = total_constant_duration_;
  std::string phase;
  
  if (motion_time <= t_up) {
    phase = "ACCELERATING";
  } else if (motion_time <= t_up + t_const) {
    phase = "CONSTANT";
  } else if (motion_time <= total_motion_time) {
    phase = "DECELERATING";
  } else {
    phase = "COMPLETE";
  }
  
  ROS_INFO_THROTTLE(2.0, "[FIGURE8-%s] t=%.1fs (motion: %.1fs) | cycle=%.2f/%d | position=[%.2f, %.2f, %.2f] | velocity=%.2f m/s",
                       phase.c_str(), t, motion_time, param, trajectory_times_, 
                       x, y, z, v_linear);
}

void TargetPublisherNode::generate_dshape_trajectory(double t)
{
  double x, y, z, vx, vy, vz;
  
  // Phase 1: Stationary period - stay at initial position (1m in front of center)
  if (t < stationary_time_) {
    // Initial position: 1m in front of trajectory center (along +x direction)
    x = trajectory_center_x_ + 1.0;
    y = trajectory_center_y_;
    z = trajectory_center_z_;
    
    // Zero velocity during stationary period
    vx = 0.0;
    vy = 0.0;
    vz = 0.0;
    
    // Publish position and velocity
    publish_target_position(x, y, z);
    publish_target_velocity(vx, vy, vz);
    
    // Debug log
    ROS_INFO_THROTTLE(2.0, "[STATIONARY] t=%.1f/%.1fs | position=[%.2f, %.2f, %.2f] | velocity=0.00 m/s",
                         t, stationary_time_, x, y, z);
    return;
  }
  
  // Phase 2: D-shape with arc-length based tracking
  // NEW APPROACH: Calculate target arc length based on velocity profile,
  // then find position on curve at that arc length
  double motion_time = t - stationary_time_;
  
  // Calculate current linear velocity based on acceleration profile
  double current_velocity = 0.0;
  double t_up = ramp_up_time_;
  double t_const = total_constant_duration_;
  double t_down = ramp_down_time_;
  std::string phase;
  
  if (motion_time <= t_up) {
    // Acceleration phase
    current_velocity = linear_acceleration_ * motion_time;
    phase = "ACCELERATING";
  } else if (motion_time <= t_up + t_const) {
    // Constant velocity phase
    current_velocity = max_linear_velocity_;
    phase = "CONSTANT";
  } else if (motion_time <= t_up + t_const + t_down) {
    // Deceleration phase
    double t_in_decel = motion_time - t_up - t_const;
    current_velocity = max_linear_velocity_ - linear_acceleration_ * t_in_decel;
    phase = "DECELERATING";
  } else {
    // Complete
    current_velocity = 0.0;
    phase = "COMPLETE";
  }
  
  // Calculate target arc length (cumulative distance traveled)
  double target_arc_length = 0.0;
  if (motion_time <= t_up) {
    // s = 0.5 * a * t²
    target_arc_length = 0.5 * linear_acceleration_ * motion_time * motion_time;
  } else if (motion_time <= t_up + t_const) {
    double s_accel = 0.5 * linear_acceleration_ * t_up * t_up;
    double dt = motion_time - t_up;
    target_arc_length = s_accel + max_linear_velocity_ * dt;
  } else if (motion_time <= t_up + t_const + t_down) {
    double s_accel = 0.5 * linear_acceleration_ * t_up * t_up;
    double s_const = max_linear_velocity_ * t_const;
    double t_in_decel = motion_time - t_up - t_const;
    target_arc_length = s_accel + s_const + 
                        max_linear_velocity_ * t_in_decel - 
                        0.5 * linear_acceleration_ * t_in_decel * t_in_decel;
  } else {
    target_arc_length = dshape_total_arc_length_ * trajectory_times_;
  }
  
  // Handle multiple cycles
  int current_cycle = static_cast<int>(target_arc_length / dshape_total_arc_length_);
  double arc_length_in_cycle = target_arc_length - current_cycle * dshape_total_arc_length_;
  
  // Check if trajectory is complete
  if (current_cycle >= trajectory_times_ || motion_time >= t_up + t_const + t_down) {
    // Hold at final position (back at start point)
    x = trajectory_center_x_ + 1.0;
    y = trajectory_center_y_;
    z = trajectory_center_z_;
    vx = 0.0;
    vy = 0.0;
    vz = 0.0;
    
    publish_target_position(x, y, z);
    publish_target_velocity(vx, vy, vz);
    
    ROS_INFO_THROTTLE(2.0, "[DSHAPE-COMPLETE] t=%.1fs | cycle=%d/%d | arc_length=%.2f/%.2fm | HOLDING",
                         t, current_cycle + 1, trajectory_times_, target_arc_length, 
                         dshape_total_arc_length_ * trajectory_times_);
    return;
  }
  
  // Find which segment we're in based on arc length
  double a = trajectory_size_ / 2.0;
  double offset_x = 1.0;
  double straight_x = offset_x - 2.0 * a;
  
  int segment = 0;
  double arc_length_in_segment = arc_length_in_cycle;
  for (int i = 0; i < 4; ++i) {
    if (arc_length_in_cycle <= dshape_cumulative_lengths_[i + 1]) {
      segment = i;
      arc_length_in_segment = arc_length_in_cycle - dshape_cumulative_lengths_[i];
      break;
    }
  }
  
  // Get control points for current segment
  double P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y;
  
  if (segment == 0) {
    P0x = offset_x;              P0y = 0.0;
    P1x = offset_x;              P1y = a * 0.55;
    P2x = offset_x - a * 0.2;    P2y = a;
    P3x = offset_x - a;          P3y = a;
  } else if (segment == 1) {
    P0x = offset_x - a;          P0y = a;
    P1x = offset_x - a * 1.8;    P1y = a;
    P2x = straight_x;            P2y = a;
    P3x = straight_x;            P3y = a * 0.5;
  } else if (segment == 2) {
    P0x = straight_x;            P0y = a * 0.5;
    P1x = straight_x;            P1y = 0.0;
    P2x = straight_x;            P2y = -a * 0.5;
    P3x = straight_x;            P3y = -a;
  } else {
    P0x = straight_x;            P0y = -a;
    P1x = offset_x - a * 0.5;    P1y = -a;
    P2x = offset_x;              P2y = -a * 0.5;
    P3x = offset_x;              P3y = 0.0;
  }
  
  // Find parameter t on this segment that corresponds to the target arc length
  double t_local = find_bezier_parameter_from_arc_length(arc_length_in_segment, 
                                                          P0x, P0y, P1x, P1y, 
                                                          P2x, P2y, P3x, P3y);
  
  // Calculate position using Bezier formula
  double mt = 1.0 - t_local;
  double t2 = t_local * t_local;
  double mt2 = mt * mt;
  
  double x_local = mt * mt * mt * P0x + 3.0 * mt2 * t_local * P1x + 
                   3.0 * mt * t2 * P2x + t2 * t_local * P3x;
  double y_local = mt * mt * mt * P0y + 3.0 * mt2 * t_local * P1y + 
                   3.0 * mt * t2 * P2y + t2 * t_local * P3y;
  
  x = trajectory_center_x_ + x_local;
  y = trajectory_center_y_ + y_local;
  z = trajectory_center_z_;
  
  // Calculate velocity direction from Bezier tangent
  double vx_local = 3.0 * (mt2 * (P1x - P0x) + 
                           2.0 * mt * t_local * (P2x - P1x) + 
                           t2 * (P3x - P2x));
  double vy_local = 3.0 * (mt2 * (P1y - P0y) + 
                           2.0 * mt * t_local * (P2y - P1y) + 
                           t2 * (P3y - P2y));
  
  // Normalize tangent and multiply by current velocity
  double tangent_magnitude = std::sqrt(vx_local * vx_local + vy_local * vy_local);
  if (tangent_magnitude > 1e-6) {
    vx = (vx_local / tangent_magnitude) * current_velocity;
    vy = (vy_local / tangent_magnitude) * current_velocity;
  } else {
    vx = 0.0;
    vy = 0.0;
  }
  vz = 0.0;
  
  // Publish
  publish_target_position(x, y, z);
  publish_target_velocity(vx, vy, vz);
  
  // Debug log
  double v_linear = std::sqrt(vx*vx + vy*vy);
  ROS_INFO_THROTTLE(2.0, "[DSHAPE-%s] t=%.1fs | seg=%d | arc=%.2f/%.2fm | pos=[%.2f,%.2f,%.2f] | vel=%.2fm/s",
                       phase.c_str(), t, segment, arc_length_in_cycle, dshape_total_arc_length_,
                       x, y, z, v_linear);
}

double TargetPublisherNode::calculate_bezier_arc_length(double P0x, double P0y, double P1x, double P1y,
                                                          double P2x, double P2y, double P3x, double P3y,
                                                          double t_end)
{
  // Calculate arc length of cubic Bezier curve using numerical integration
  const int num_samples = 100;
  double arc_length = 0.0;
  
  double prev_x = P0x;
  double prev_y = P0y;
  
  int end_sample = static_cast<int>(t_end * num_samples);
  
  for (int i = 1; i <= end_sample; ++i) {
    double t = static_cast<double>(i) / num_samples;
    double mt = 1.0 - t;
    double t2 = t * t;
    double mt2 = mt * mt;
    
    // Calculate position at parameter t
    double x = mt * mt * mt * P0x + 3.0 * mt2 * t * P1x + 
               3.0 * mt * t2 * P2x + t2 * t * P3x;
    double y = mt * mt * mt * P0y + 3.0 * mt2 * t * P1y + 
               3.0 * mt * t2 * P2y + t2 * t * P3y;
    
    // Accumulate distance
    double dx = x - prev_x;
    double dy = y - prev_y;
    arc_length += std::sqrt(dx * dx + dy * dy);
    
    prev_x = x;
    prev_y = y;
  }
  
  return arc_length;
}

double TargetPublisherNode::find_bezier_parameter_from_arc_length(double target_length,
                                                                   double P0x, double P0y, double P1x, double P1y,
                                                                   double P2x, double P2y, double P3x, double P3y)
{
  // Use binary search to find parameter t that gives target arc length
  if (target_length <= 0.0) {
    return 0.0;
  }
  
  // Get total length
  double total_length = calculate_bezier_arc_length(P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y, 1.0);
  
  if (target_length >= total_length) {
    return 1.0;
  }
  
  // Binary search
  double t_low = 0.0;
  double t_high = 1.0;
  const double tolerance = 0.001;  // Arc length tolerance in meters
  const int max_iterations = 20;
  
  for (int iter = 0; iter < max_iterations; ++iter) {
    double t_mid = (t_low + t_high) / 2.0;
    double arc_at_mid = calculate_bezier_arc_length(P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y, t_mid);
    
    if (std::abs(arc_at_mid - target_length) < tolerance) {
      return t_mid;
    }
    
    if (arc_at_mid < target_length) {
      t_low = t_mid;
    } else {
      t_high = t_mid;
    }
  }
  
  return (t_low + t_high) / 2.0;
}

void TargetPublisherNode::calculate_dshape_segment_lengths()
{
  // Pre-calculate arc lengths for each Bezier segment
  double a = trajectory_size_ / 2.0;
  double offset_x = 1.0;
  double straight_x = offset_x - 2.0 * a;
  
  // Segment 0
  double P0x = offset_x;              double P0y = 0.0;
  double P1x = offset_x;              double P1y = a * 0.55;
  double P2x = offset_x - a * 0.2;    double P2y = a;
  double P3x = offset_x - a;          double P3y = a;
  dshape_segment_lengths_[0] = calculate_bezier_arc_length(P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y);
  
  // Segment 1
  P0x = offset_x - a;          P0y = a;
  P1x = offset_x - a * 1.8;    P1y = a;
  P2x = straight_x;            P2y = a;
  P3x = straight_x;            P3y = a * 0.5;
  dshape_segment_lengths_[1] = calculate_bezier_arc_length(P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y);
  
  // Segment 2
  P0x = straight_x;            P0y = a * 0.5;
  P1x = straight_x;            P1y = 0.0;
  P2x = straight_x;            P2y = -a * 0.5;
  P3x = straight_x;            P3y = -a;
  dshape_segment_lengths_[2] = calculate_bezier_arc_length(P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y);
  
  // Segment 3
  P0x = straight_x;            P0y = -a;
  P1x = offset_x - a * 0.5;    P1y = -a;
  P2x = offset_x;              P2y = -a * 0.5;
  P3x = offset_x;              P3y = 0.0;
  dshape_segment_lengths_[3] = calculate_bezier_arc_length(P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y);
  
  // Calculate cumulative lengths
  dshape_cumulative_lengths_[0] = 0.0;
  dshape_total_arc_length_ = 0.0;
  for (int i = 0; i < 4; ++i) {
    dshape_total_arc_length_ += dshape_segment_lengths_[i];
    dshape_cumulative_lengths_[i + 1] = dshape_total_arc_length_;
  }
}

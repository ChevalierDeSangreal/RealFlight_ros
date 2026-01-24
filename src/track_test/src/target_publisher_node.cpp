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
{
  // Declare and get parameters
  nh_private_.param("drone_id", drone_id_, 0);
  nh_private_.param("circle_radius", circle_radius_, 2.0);
  nh_private_.param("circle_duration", circle_duration_, 20.0);
  nh_private_.param("circle_init_phase", circle_init_phase_, 0.0);
  nh_private_.param("circle_times", circle_times_, 1);
  nh_private_.param("ramp_up_time", ramp_up_time_, 3.0);
  nh_private_.param("ramp_down_time", ramp_down_time_, 3.0);
  nh_private_.param("stationary_time", stationary_time_, 3.0);
  
  nh_private_.param("circle_center_x", circle_center_x_, 0.0);
  nh_private_.param("circle_center_y", circle_center_y_, 0.0);
  nh_private_.param("circle_center_z", circle_center_z_, 1.2);  // ENU: positive z is up
  
  nh_private_.param("timer_period", timer_period_, 0.02);
  nh_private_.param("max_speed", max_speed_, -1.0);
  use_max_speed_ = (max_speed_ > 0.0);
  
  // Get use_sim_time parameter (ROS standard parameter)
  nh_.param("use_sim_time", use_sim_time_, false);
  
  // Parameter validation
  if (circle_times_ < 1) {
    ROS_WARN("circle_times must be >= 1, setting to 1");
    circle_times_ = 1;
  }
  
  // Calculate trajectory parameters
  effective_duration_ = calculate_effective_duration();
  max_angular_vel_ = 2.0 * M_PI / effective_duration_;
  angular_acceleration_ = max_angular_vel_ / ramp_up_time_;
  
  double theta_ramp_up = 0.5 * max_angular_vel_ * ramp_up_time_;
  double theta_ramp_down = 0.5 * max_angular_vel_ * ramp_down_time_;
  double theta_ramps_total = theta_ramp_up + theta_ramp_down;
  
  // Total angular displacement required for N circles
  double theta_required = circle_times_ * 2.0 * M_PI;
  // Angular displacement required for constant velocity phase
  double theta_constant = theta_required - theta_ramps_total;
  // Time required for constant velocity phase
  total_constant_duration_ = theta_constant / max_angular_vel_;
  
  ROS_INFO("=== Target Publisher Node ===");
  ROS_INFO("Circle radius: %.2f m", circle_radius_);
  ROS_INFO("Circle center: [%.2f, %.2f, %.2f] m (ENU)", 
              circle_center_x_, circle_center_y_, circle_center_z_);
  ROS_INFO("Initial stationary time: %.2f s", stationary_time_);
  ROS_INFO("Number of circles: %d", circle_times_);
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
    return circle_duration_;
  }
  
  double circumference = 2.0 * M_PI * circle_radius_;
  double min_duration = circumference / max_speed_;
  
  if (min_duration > circle_duration_) {
    return min_duration;
  }
  
  return circle_duration_;
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
    double theta_initial = circle_init_phase_;
    double x = circle_center_x_ + circle_radius_ * std::cos(theta_initial);
    double y = circle_center_y_ + circle_radius_ * std::sin(theta_initial);
    double z = circle_center_z_;
    
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
  
  generate_circular_trajectory(elapsed);
}

void TargetPublisherNode::generate_circular_trajectory(double t)
{
  double x, y, z, vx, vy, vz;
  
  // Phase 1: Stationary period - stay at initial position
  if (t < stationary_time_) {
    // Calculate initial position (starting point on circle)
    double theta_initial = circle_init_phase_;
    x = circle_center_x_ + circle_radius_ * std::cos(theta_initial);
    y = circle_center_y_ + circle_radius_ * std::sin(theta_initial);
    z = circle_center_z_;
    
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
  
  // Calculate angular velocity at motion time
  double current_omega = calculate_angular_velocity_at_time(motion_time);
  
  // Calculate angular position at motion time
  double theta = calculate_theta_at_time(motion_time);
  double theta_with_phase = theta + circle_init_phase_;
  
  // Position on circle
  x = circle_center_x_ + circle_radius_ * std::cos(theta_with_phase);
  y = circle_center_y_ + circle_radius_ * std::sin(theta_with_phase);
  z = circle_center_z_;
  
  // Linear velocity in tangential direction
  double v_linear = current_omega * circle_radius_;
  vx = -v_linear * std::sin(theta_with_phase);
  vy =  v_linear * std::cos(theta_with_phase);
  vz = 0.0;
  
  // Publish position and velocity
  publish_target_position(x, y, z);
  publish_target_velocity(vx, vy, vz);
  
  // Debug log
  double total_motion_time = ramp_up_time_ + total_constant_duration_ + ramp_down_time_;
  double t_up = ramp_up_time_;
  double t_const = total_constant_duration_;
  std::string phase;
  double current_circle = 0.0;
  
  if (motion_time <= t_up) {
    phase = "ACCELERATING";
    current_circle = 0.0;
  } else if (motion_time <= t_up + t_const) {
    phase = "CONSTANT";
    double elapsed_const = motion_time - t_up;
    current_circle = elapsed_const / effective_duration_;
  } else if (motion_time <= total_motion_time) {
    phase = "DECELERATING";
    current_circle = circle_times_;
  } else {
    phase = "COMPLETE";
    current_circle = circle_times_;
  }
  
  ROS_INFO_THROTTLE(2.0, "[%s] t=%.1fs (motion: %.1fs) | circle=%.2f/%d | position=[%.2f, %.2f, %.2f] | velocity=%.2f m/s",
                       phase.c_str(), t, motion_time, current_circle, circle_times_, 
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


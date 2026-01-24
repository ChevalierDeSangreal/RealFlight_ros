#include "track_test/target_generator.hpp"
#include <cmath>

TargetGenerator::TargetGenerator(
  ros::NodeHandle* nh,
  bool use_target_topic,
  const std::string& position_topic,
  const std::string& velocity_topic)
  : nh_(nh)
  , use_target_topic_(use_target_topic)
  , position_topic_(position_topic)
  , velocity_topic_(velocity_topic)
  , target_x_(0.0)
  , target_y_(0.0)
  , target_z_(0.0)
  , target_vx_(0.0)
  , target_vy_(0.0)
  , target_vz_(0.0)
  , target_ready_(false)
{
  if (use_target_topic_) {
    // Topic mode: create subscribers
    ROS_INFO("TargetGenerator: Topic mode - Subscribing to target position topic: %s", 
             position_topic_.c_str());
    
    position_sub_ = nh_->subscribe<geometry_msgs::PointStamped>(
      position_topic_,
      10,
      &TargetGenerator::position_callback, this);
    
    // Velocity topic is optional
    ROS_INFO("TargetGenerator: Subscribing to target velocity topic (optional): %s", 
             velocity_topic_.c_str());
    
    velocity_sub_ = nh_->subscribe<geometry_msgs::TwistStamped>(
      velocity_topic_,
      10,
      &TargetGenerator::velocity_callback, this);
  } else {
    // Static mode: no subscribers created
    ROS_INFO("TargetGenerator: Static mode - Fixed target will be generated on initialization");
  }
}

void TargetGenerator::initialize_static_target(
  double drone_x, 
  double drone_y, 
  double drone_z, 
  double drone_yaw,
  double offset_distance)
{
  if (use_target_topic_) {
    ROS_WARN("TargetGenerator: initialize_static_target call ignored in topic mode");
    return;
  }

  std::lock_guard<std::mutex> lock(target_mutex_);
  
  // Calculate target position in front of the drone
  // In ENU frame, yaw=0 points East (+x), yaw=π/2 points North (+y)
  // Target position = current position + offset * [cos(yaw), sin(yaw), 0]
  target_x_ = drone_x + offset_distance * std::cos(drone_yaw);
  target_y_ = drone_y + offset_distance * std::sin(drone_yaw);
  target_z_ = drone_z;  // Keep same altitude
  
  // Static target, zero velocity
  target_vx_ = 0.0;
  target_vy_ = 0.0;
  target_vz_ = 0.0;
  
  target_ready_ = true;
  
  ROS_INFO("TargetGenerator: Static target initialized");
  ROS_INFO("  Drone position (ENU): [%.2f, %.2f, %.2f], yaw=%.2f rad", 
           drone_x, drone_y, drone_z, drone_yaw);
  ROS_INFO("  Target position (ENU): [%.2f, %.2f, %.2f] (%.2f m ahead)", 
           target_x_, target_y_, target_z_, offset_distance);
}

void TargetGenerator::get_target_position(double& target_x, double& target_y, double& target_z) const
{
  std::lock_guard<std::mutex> lock(target_mutex_);
  target_x = target_x_;
  target_y = target_y_;
  target_z = target_z_;
}

void TargetGenerator::get_target_velocity(double& target_vx, double& target_vy, double& target_vz) const
{
  std::lock_guard<std::mutex> lock(target_mutex_);
  target_vx = target_vx_;
  target_vy = target_vy_;
  target_vz = target_vz_;
}

bool TargetGenerator::is_target_ready() const
{
  std::lock_guard<std::mutex> lock(target_mutex_);
  return target_ready_;
}

void TargetGenerator::reset()
{
  std::lock_guard<std::mutex> lock(target_mutex_);
  target_x_ = 0.0;
  target_y_ = 0.0;
  target_z_ = 0.0;
  target_vx_ = 0.0;
  target_vy_ = 0.0;
  target_vz_ = 0.0;
  target_ready_ = false;
  
  ROS_INFO("TargetGenerator: Reset");
}

// ROS1 topic callbacks (话题模式)
void TargetGenerator::position_callback(const geometry_msgs::PointStamped::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(target_mutex_);
  target_x_ = msg->point.x;
  target_y_ = msg->point.y;
  target_z_ = msg->point.z;
  target_ready_ = true;
  
  ROS_DEBUG("TargetGenerator: Received target position [%.2f, %.2f, %.2f]",
            target_x_, target_y_, target_z_);
}

void TargetGenerator::velocity_callback(const geometry_msgs::TwistStamped::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(target_mutex_);
  target_vx_ = msg->twist.linear.x;
  target_vy_ = msg->twist.linear.y;
  target_vz_ = msg->twist.linear.z;
  
  ROS_DEBUG("TargetGenerator: Received target velocity [%.2f, %.2f, %.2f]",
            target_vx_, target_vy_, target_vz_);
}




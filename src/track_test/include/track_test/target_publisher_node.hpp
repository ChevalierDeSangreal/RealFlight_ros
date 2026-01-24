#ifndef TARGET_PUBLISHER_NODE_HPP_
#define TARGET_PUBLISHER_NODE_HPP_

#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <std_msgs/Int32.h>

#include <chrono>
#include <string>

/**
 * @brief Target object position publisher node - publishes circular motion trajectory target position
 * 
 * This node generates circular motion target trajectory and publishes target position and velocity,
 * for track_test_node to subscribe.
 */
class TargetPublisherNode
{
public:
  explicit TargetPublisherNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private);

private:
  void timer_callback(const ros::TimerEvent& event);
  void state_callback(const std_msgs::Int32::ConstPtr& msg);
  
  /**
   * @brief Generate circular motion trajectory
   * @param t Current time (seconds)
   */
  void generate_circular_trajectory(double t);
  
  /**
   * @brief Publish target position
   */
  void publish_target_position(double x, double y, double z);
  
  /**
   * @brief Publish target velocity
   */
  void publish_target_velocity(double vx, double vy, double vz);
  
  /**
   * @brief Calculate angular position at specified time
   */
  double calculate_theta_at_time(double t);
  
  /**
   * @brief Calculate angular velocity at specified time
   */
  double calculate_angular_velocity_at_time(double t);
  
  /**
   * @brief Calculate effective duration (considering speed limit)
   */
  double calculate_effective_duration();

  // Node handles
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // Circular motion parameters
  double circle_radius_;              // Circular trajectory radius [m]
  double circle_duration_;            // Single circle duration [s]
  double circle_init_phase_;          // Initial phase angle [rad]
  int circle_times_;                  // Number of circles
  double ramp_up_time_;               // Acceleration time [s]
  double ramp_down_time_;             // Deceleration time [s]
  double stationary_time_;            // Initial stationary time [s]
  
  // Circle center position (ENU coordinate system for ROS1/MAVROS)
  double circle_center_x_;            // Circle center East position [m]
  double circle_center_y_;            // Circle center North position [m]
  double circle_center_z_;            // Circle center Up position [m]
  
  // Control parameters
  double timer_period_;               // Control period [s]
  double max_speed_;                  // Maximum linear speed limit [m/s] (-1 = unlimited)
  bool use_max_speed_;
  
  // Calculated parameters
  double effective_duration_;         // Actual duration considering speed limit [s]
  double total_constant_duration_;    // Total constant velocity phase time [s]
  double max_angular_vel_;            // Maximum angular velocity [rad/s]
  double angular_acceleration_;       // Angular acceleration [rad/s²]
  
  // State machine state enum (consistent with track_test_node)
  enum class FsmState {
    INIT = 0,
    ARMING = 1,
    TAKEOFF = 2,
    GOTO = 3,
    HOVER = 4,
    TRAJ = 5,
    END_TRAJ = 6,
    LAND = 7,
    DONE = 8
  };
  
  // Runtime state
  bool trajectory_started_;
  ros::Time start_time_;  // ROS time start time (for use_sim_time mode)
  std::chrono::steady_clock::time_point start_time_system_;  // System clock start time (for onboard mode)
  bool use_sim_time_;  // Whether to use simulation time
  
  // State machine interaction
  int drone_id_;                    // Drone ID, used to determine state topic
  FsmState current_state_;          // Current state machine state
  bool in_traj_state_;              // Whether entered TRAJ state
  
  // ROS1 interfaces
  ros::Publisher position_pub_;
  ros::Publisher velocity_pub_;
  ros::Subscriber state_sub_;  // State subscription
  ros::Timer timer_;
};

#endif  // TARGET_PUBLISHER_NODE_HPP_


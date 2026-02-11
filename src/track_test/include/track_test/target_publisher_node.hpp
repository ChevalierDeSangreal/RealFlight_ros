#ifndef TARGET_PUBLISHER_NODE_HPP_
#define TARGET_PUBLISHER_NODE_HPP_

#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <std_msgs/Int32.h>

#include <chrono>
#include <string>

/**
 * @brief Target object position publisher node - publishes trajectory target position
 * 
 * This node generates trajectory (circle, figure-8, D-shape) and publishes target position and velocity,
 * for track_test_node to subscribe.
 */
class TargetPublisherNode
{
public:
  explicit TargetPublisherNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private);

private:
  // Trajectory type enum
  enum class TrajectoryType {
    CIRCLE,      // 圆形轨迹
    FIGURE8,     // 八字轨迹
    D_SHAPE      // D型轨迹
  };

  void timer_callback(const ros::TimerEvent& event);
  void state_callback(const std_msgs::Int32::ConstPtr& msg);
  
  /**
   * @brief Generate circular motion trajectory
   * @param t Current time (seconds)
   */
  void generate_circular_trajectory(double t);
  
  /**
   * @brief Generate figure-8 motion trajectory
   * @param t Current time (seconds)
   */
  void generate_figure8_trajectory(double t);
  
  /**
   * @brief Generate D-shape motion trajectory
   * @param t Current time (seconds)
   */
  void generate_dshape_trajectory(double t);
  
  /**
   * @brief Publish target position
   */
  void publish_target_position(double x, double y, double z);
  
  /**
   * @brief Publish target velocity
   */
  void publish_target_velocity(double vx, double vy, double vz);
  
  /**
   * @brief Calculate normalized path parameter at specified time (0 to 1)
   */
  double calculate_normalized_parameter_at_time(double t);
  
  /**
   * @brief Calculate parameter velocity at specified time
   */
  double calculate_parameter_velocity_at_time(double t);
  
  /**
   * @brief Calculate angular position at specified time (for circular trajectory)
   */
  double calculate_theta_at_time(double t);
  
  /**
   * @brief Calculate angular velocity at specified time (for circular trajectory)
   */
  double calculate_angular_velocity_at_time(double t);
  
  /**
   * @brief Calculate effective duration (considering speed limit)
   */
  double calculate_effective_duration();
  
  /**
   * @brief Calculate arc length of a cubic Bezier curve using numerical integration
   * @param P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y Control points
   * @param t_end End parameter (0 to t_end), default 1.0
   * @return Arc length [m]
   */
  double calculate_bezier_arc_length(double P0x, double P0y, double P1x, double P1y,
                                     double P2x, double P2y, double P3x, double P3y,
                                     double t_end = 1.0);
  
  /**
   * @brief Find parameter t on Bezier curve given target arc length from start
   * @param target_length Target arc length from start of curve
   * @param P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y Control points
   * @return Parameter t (0 to 1)
   */
  double find_bezier_parameter_from_arc_length(double target_length,
                                                double P0x, double P0y, double P1x, double P1y,
                                                double P2x, double P2y, double P3x, double P3y);
  
  /**
   * @brief Pre-calculate D-shape segment arc lengths
   */
  void calculate_dshape_segment_lengths();

  // Node handles
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // Trajectory type
  TrajectoryType trajectory_type_;    // Trajectory type (CIRCLE, FIGURE8, D_SHAPE)
  std::string trajectory_type_str_;   // Trajectory type string (for parsing)

  // Trajectory motion parameters
  double trajectory_size_;            // Trajectory size [m] (radius for circle, scale for figure-8/D-shape)
  double trajectory_duration_;        // Single trajectory cycle duration [s] (auto-calculated for D-shape)
  double circle_init_phase_;          // Initial phase angle [rad] (for circle)
  int trajectory_times_;              // Number of trajectory cycles
  double ramp_up_time_;               // Acceleration time [s] (auto-calculated for D-shape)
  double ramp_down_time_;             // Deceleration time [s] (auto-calculated for D-shape)
  double stationary_time_;            // Initial stationary time [s]
  
  // Velocity-based parameters for D-shape trajectory
  double max_linear_velocity_;        // Maximum linear velocity [m/s] (for D-shape)
  double linear_acceleration_;        // Linear acceleration/deceleration [m/s²] (for D-shape)
  
  // Trajectory center position (ENU coordinate system for ROS1/MAVROS)
  double trajectory_center_x_;        // Trajectory center East position [m]
  double trajectory_center_y_;        // Trajectory center North position [m]
  double trajectory_center_z_;        // Trajectory center Up position [m]
  
  // Legacy circular motion parameters (for backward compatibility)
  double circle_radius_;              // Circular trajectory radius [m]
  double circle_duration_;            // Single circle duration [s]
  int circle_times_;                  // Number of circles
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
  
  // D-shape arc-length based tracking (NEW: replaces time-based ratios)
  double dshape_total_arc_length_;     // Total arc length of D-shape trajectory [m]
  double dshape_segment_lengths_[4];   // Arc length of each Bezier segment [m]
  double dshape_cumulative_lengths_[5]; // Cumulative arc lengths [0, L0, L0+L1, ..., total]
  double dshape_current_arc_length_;   // Current traveled arc length [m] (for arc-length tracking)
  
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


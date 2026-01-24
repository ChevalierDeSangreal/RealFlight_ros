#ifndef TRACK_TEST_NODE_HPP_
#define TRACK_TEST_NODE_HPP_

#include <ros/ros.h>
#include <std_msgs/Int32.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/State.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include "offboard_state_machine/utils.hpp"
#include "track_test/tflite_policy.hpp"
#include "track_test/target_generator.hpp"

#include <chrono>
#include <string>
#include <vector>
#include <mutex>
#include <memory>

class TrackTestNode
{
public:
  explicit TrackTestNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private, int drone_id);

private:
  // Timer callbacks (two-timer architecture)
  void action_update_callback(const ros::TimerEvent& event);  // 50Hz: Neural network inference
  void control_send_callback(const ros::TimerEvent& event);   // 100Hz: High-frequency command transmission
  
  // ROS1 subscriber callbacks
  void state_callback(const std_msgs::Int32::ConstPtr& msg);
  void local_pos_callback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void local_vel_callback(const geometry_msgs::TwistStamped::ConstPtr& msg);
  
  // Control functions
  void update_neural_action();       // Update action from neural network
  void publish_current_action();     // Publish current action at high frequency
  void send_state_command(int state);
  
  std::vector<float> get_observation();
  
  // Neural network dimensions (matching TrackEnvVer6)
  static constexpr int OBS_DIM = 9;  // 机体系速度(3) + 机体系重力(3) + 机体系目标位置(3)  

  // Node handles
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // Basic parameters
  int drone_id_;
  double hover_duration_;             // Duration to hover [s]
  double hover_thrust_;               // Hover thrust [0.0-1.0]
  double mode_stabilization_delay_;   // Delay before starting neural control to ensure mode switch [s]
  
  // Control parameters (two-timer architecture)
  double action_update_period_;       // Action update period [s] (50Hz for NN inference)
  double control_send_period_;        // Control send period [s] (100Hz for command transmission)
  
  // State management
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
  
  FsmState current_state_;
  bool waiting_hover_;
  bool hover_command_sent_;
  bool hover_started_;
  bool hover_completed_;
  bool neural_control_ready_;         // True when mode stabilization delay has passed
  ros::Time hover_detect_time_;
  ros::Time hover_start_time_;
  
  // Hover position (captured when entering TRAJ state, ENU frame)
  double hover_x_;
  double hover_y_;
  double hover_z_;
  double hover_yaw_;
  
  // Current drone position (ENU frame)
  double current_x_;
  double current_y_;
  double current_z_;
  bool odom_ready_;
  
  // Current drone state (for neural network observation, ENU frame)
  double current_vx_;
  double current_vy_;
  double current_vz_;
  double current_roll_;
  double current_pitch_;
  double current_yaw_;
  bool local_position_ready_;
  bool attitude_ready_;
  
  // Target generator (管理目标位置生成或订阅)
  std::unique_ptr<TargetGenerator> target_generator_;
  bool use_target_topic_;          // 是否使用ROS1话题订阅目标
  std::string target_position_topic_;  // 目标位置话题名称
  std::string target_velocity_topic_;  // 目标速度话题名称
  double target_offset_distance_;  // 静态目标距离无人机的距离[m]
  
  // Target state (目标位置/轨迹) - from target_generator
  double target_x_;
  double target_y_;
  double target_z_;
  double target_vx_;
  double target_vy_;
  double target_vz_;
  
  // Neural network policy (REQUIRED)
  std::unique_ptr<TFLitePolicyInference> policy_;
  std::string model_path_;
  
  // Current action storage (shared between timers)
  std::vector<float> current_action_;  // [thrust, omega_x, omega_y, omega_z]
  std::mutex action_mutex_;            // Protect current_action_ from concurrent access
  
  // Step counter for debugging
  int step_counter_;                   // Track number of neural network inference steps
  
  // ROS interfaces
  ros::Publisher attitude_pub_;
  ros::Publisher state_cmd_pub_;
  ros::Subscriber state_sub_;
  ros::Subscriber local_pos_sub_;
  ros::Subscriber local_vel_sub_;
  
  // Two-timer architecture
  ros::Timer action_update_timer_;  // 50Hz: Neural network inference
  ros::Timer control_send_timer_;   // 100Hz: Command transmission
};

#endif  // TRACK_TEST_NODE_HPP_


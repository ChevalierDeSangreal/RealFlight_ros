#ifndef TARGET_GENERATOR_HPP_
#define TARGET_GENERATOR_HPP_

#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <memory>
#include <mutex>

/**
 * @brief 目标生成器类 - 负责生成或订阅目标位置
 * 
 * 两种模式:
 * 1. 静态模式 (use_target_topic=false): 在初始化时生成固定位置的静止目标
 * 2. 话题模式 (use_target_topic=true): 订阅ROS1话题获取实时目标位置
 */
class TargetGenerator
{
public:
  /**
   * @brief 构造函数
   * @param nh ROS1节点句柄(用于创建订阅者)
   * @param use_target_topic 是否使用ROS1话题订阅目标
   * @param position_topic 目标位置话题名称
   * @param velocity_topic 目标速度话题名称(可选)
   */
  explicit TargetGenerator(
    ros::NodeHandle* nh,
    bool use_target_topic = false,
    const std::string& position_topic = "/target/position",
    const std::string& velocity_topic = "/target/velocity");

  /**
   * @brief 初始化静态目标(在无人机正前方offset_distance米处)
   * @param drone_x 无人机当前x位置
   * @param drone_y 无人机当前y位置
   * @param drone_z 无人机当前z位置
   * @param drone_yaw 无人机当前偏航角(用于确定"正前方")
   * @param offset_distance 目标距离无人机的距离[m]
   */
  void initialize_static_target(
    double drone_x, 
    double drone_y, 
    double drone_z, 
    double drone_yaw,
    double offset_distance = 1.0);

  /**
   * @brief 获取当前目标位置
   * @param target_x 输出目标x坐标
   * @param target_y 输出目标y坐标
   * @param target_z 输出目标z坐标
   */
  void get_target_position(double& target_x, double& target_y, double& target_z) const;

  /**
   * @brief 获取当前目标速度
   * @param target_vx 输出目标x速度
   * @param target_vy 输出目标y速度
   * @param target_vz 输出目标z速度
   */
  void get_target_velocity(double& target_vx, double& target_vy, double& target_vz) const;

  /**
   * @brief 检查目标数据是否就绪
   * @return true表示目标数据已初始化或已接收到话题数据
   */
  bool is_target_ready() const;

  /**
   * @brief 重置目标生成器(清除所有状态)
   */
  void reset();

private:
  // ROS1 topic callbacks
  void position_callback(const geometry_msgs::PointStamped::ConstPtr& msg);
  void velocity_callback(const geometry_msgs::TwistStamped::ConstPtr& msg);

  // ROS1 node handle pointer (not owned)
  ros::NodeHandle* nh_;

  // Configuration
  bool use_target_topic_;      // 是否使用话题模式
  std::string position_topic_; // 位置话题名称
  std::string velocity_topic_;  // 速度话题名称

  // Target state (thread-safe)
  mutable std::mutex target_mutex_;
  double target_x_;
  double target_y_;
  double target_z_;
  double target_vx_;
  double target_vy_;
  double target_vz_;
  bool target_ready_;

  // ROS1 subscribers (only used in topic mode)
  ros::Subscriber position_sub_;
  ros::Subscriber velocity_sub_;
};

#endif  // TARGET_GENERATOR_HPP_




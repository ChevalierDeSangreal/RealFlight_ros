#include "hover_test/hover_test_node.hpp"
#include <ros/ros.h>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "hover_test_node");
  
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  
  // Read the drone_id parameter
  int drone_id;
  nh_private.param("drone_id", drone_id, 0);
  
  ROS_INFO("Starting Hover Test Node for drone %d", drone_id);
  
  // Create hover test node
  HoverTestNode hover_test(nh, nh_private, drone_id);
  
  ros::spin();
  
  return 0;
}


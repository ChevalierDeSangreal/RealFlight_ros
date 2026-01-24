#include "track_test/target_publisher_node.hpp"
#include <ros/ros.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "target_publisher_node");
  
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  
  TargetPublisherNode node(nh, nh_private);
  
  ros::spin();
  
  return 0;
}


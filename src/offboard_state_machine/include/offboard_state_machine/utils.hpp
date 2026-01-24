#ifndef OFFBOARD_STATE_MACHINE_UTILS_HPP_
#define OFFBOARD_STATE_MACHINE_UTILS_HPP_

#include <ros/ros.h>
#include <cmath>

namespace offboard_utils {

// Wrap angle to [-pi, pi]
inline float wrap_pi(float x) {
  while (x >  M_PI) x -= 2.f * M_PI;
  while (x < -M_PI) x += 2.f * M_PI;
  return x;
}

// Convert ENU yaw to NED yaw (if needed for coordinate transformations)
// Note: In RealFlight_ros we use ENU throughout, but keep this for compatibility
inline float enu_to_ned_yaw(float yaw_enu) {
  float y = yaw_enu + static_cast<float>(M_PI_2);
  return wrap_pi(y);
}

// Convert NED yaw to ENU yaw (if needed for coordinate transformations)
inline float ned_to_enu_yaw(float yaw_ned) {
  float y = yaw_ned - static_cast<float>(M_PI_2);
  return wrap_pi(y);
}

}  // namespace offboard_utils

#endif  // OFFBOARD_STATE_MACHINE_UTILS_HPP_


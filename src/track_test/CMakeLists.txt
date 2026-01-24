cmake_minimum_required(VERSION 3.0.2)
project(track_test)

## C++17
add_compile_options(-std=c++17)

## Find catkin and dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  mavros_msgs
  tf2
  tf2_ros
  offboard_state_machine
)

## TensorFlow Lite (REQUIRED)
# Using local installation in third_party directory
set(TFLITE_ROOT "$ENV{HOME}/wangzimo/third_party/tensorflow_lite" CACHE PATH "TensorFlow Lite root directory")
set(TFLITE_INCLUDE_DIR "${TFLITE_ROOT}/include" CACHE PATH "TensorFlow Lite include directory")
set(TFLITE_LIB_DIR "${TFLITE_ROOT}/lib" CACHE PATH "TensorFlow Lite library directory")

# Check if TFLite is properly installed (REQUIRED - fail if not found)
if(NOT EXISTS ${TFLITE_INCLUDE_DIR}/tensorflow/lite)
    message(FATAL_ERROR "TensorFlow Lite not found! Neural network control is REQUIRED.\n"
                        "  Expected include directory: ${TFLITE_INCLUDE_DIR}/tensorflow/lite\n"
                        "  Please compile TensorFlow Lite first:\n"
                        "    cd ~/wangzimo && ./build_tflite_minimal.sh")
endif()

# 支持静态库 (.a) 或共享库 (.so)
if(EXISTS ${TFLITE_LIB_DIR}/libtensorflow-lite.a)
    set(TFLITE_LIB_FILE "${TFLITE_LIB_DIR}/libtensorflow-lite.a")
    set(TFLITE_LIB_TYPE "static")
elseif(EXISTS ${TFLITE_LIB_DIR}/libtensorflow-lite.so)
    set(TFLITE_LIB_FILE "${TFLITE_LIB_DIR}/libtensorflow-lite.so")
    set(TFLITE_LIB_TYPE "shared")
else()
    message(FATAL_ERROR "TensorFlow Lite library not found! Neural network control is REQUIRED.\n"
                        "  Expected library: ${TFLITE_LIB_DIR}/libtensorflow-lite.a or .so\n"
                        "  Please compile TensorFlow Lite first:\n"
                        "    cd ~/wangzimo && ./build_tflite_minimal.sh")
endif()

message(STATUS "TensorFlow Lite found (REQUIRED):")
message(STATUS "  - Include: ${TFLITE_INCLUDE_DIR}")
message(STATUS "  - Library: ${TFLITE_LIB_FILE} (${TFLITE_LIB_TYPE})")

# Define TFLite libraries list
set(TFLITE_LIBS ${TFLITE_LIB_FILE})

# 如果是共享库，添加运行时库路径
if(TFLITE_LIB_TYPE STREQUAL "shared")
    set(TFLITE_LIBS ${TFLITE_LIBS} ${CMAKE_DL_LIBS} pthread m)
endif()

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS 
    roscpp 
    std_msgs 
    geometry_msgs
    mavros_msgs 
    tf2 
    tf2_ros
    offboard_state_machine
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${TFLITE_INCLUDE_DIR}
  $ENV{HOME}/wangzimo/Downloads/tensorflow
  $ENV{HOME}/.cache/bazel/_bazel_core/c7257a5dbf4f297278dae03eac548f56/external/flatbuffers/include
)

## Declare a C++ executable
add_executable(track_test_node
  src/track_test_node.cpp
  src/track_test_main.cpp
  src/target_generator.cpp
)

## Link libraries
target_link_libraries(track_test_node
  ${catkin_LIBRARIES}
  ${TFLITE_LIBS}
)

## TensorFlow Lite is always enabled (required)
target_compile_definitions(track_test_node PRIVATE TFLITE_ENABLED)

## Target publisher node (for publishing target object circular trajectory)
add_executable(target_publisher_node
  src/target_publisher_node.cpp
  src/target_publisher_main.cpp
)

## Link libraries for target_publisher_node
target_link_libraries(target_publisher_node
  ${catkin_LIBRARIES}
)

#############
## Install ##
#############

## Mark executables for installation
install(TARGETS track_test_node target_publisher_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark headers for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)

## Mark launch files for installation
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Mark config files for installation
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)


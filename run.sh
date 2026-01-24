#!/bin/bash
# Quick start script for RealFlight_ros with PX4 SITL

echo "========================================="
echo "RealFlight_ros Quick Start"
echo "========================================="

# Check if workspace is built
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -f "$SCRIPT_DIR/devel/setup.bash" ]; then
    echo "ERROR: Workspace not built!"
    echo "Please run ./build.sh first"
    exit 1
fi

echo ""
echo "This script will guide you through running RealFlight_ros"
echo ""
echo "Prerequisites:"
echo "1. PX4 SITL must be running in another terminal"
echo "   cd ~/PX4-Autopilot"
echo "   make px4_sitl gazebo"
echo ""
echo "Press Enter to continue..."
read

# Source workspace
source "$SCRIPT_DIR/devel/setup.bash"

echo ""
echo "Select test mode:"
echo "1) Basic offboard control (takeoff, hover, land)"
echo "2) Hover test (constant thrust hovering)"
echo "3) Track test (circular trajectory)"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Launching basic offboard control..."
        roslaunch realflight_ros sitl_single_drone.launch
        ;;
    2)
        echo "Launching hover test..."
        roslaunch realflight_ros sitl_hover_test.launch
        ;;
    3)
        echo "Launching track test..."
        roslaunch realflight_ros sitl_track_test.launch
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac


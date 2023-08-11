#!/bin/bash
source ./install/local_setup.bash
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 launch autoware_launch logging_simulator.launch.xml map_path:=/home/orin/autoware_map/sample-map-rosbag vehicle_model:=sample_vehicle sensor_model:=sample_sensor_kit

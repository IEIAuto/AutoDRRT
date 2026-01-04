rm -rf /autodrrt/logs/*
export ROS_LOG_DIR=/autodrrt/logs/
export ROS_HOME=/autodrrt
source /autodrrt/install/setup.bash
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 launch autoware_launch logging_simulator.launch.xml map_path:=/autoware_map/sample-map-rosbag vehicle_model:=sample_vehicle sensor_model:=sample_sensor_kit | tee 1.txt


rm -rf /home/orin/autodrrt/logs/*
export ROS_LOG_DIR=/home/orin/autodrrt/logs/
export ROS_HOME=/home/orin/autodrrt
source /home/orin/disk/autodrrt/install/setup.bash
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 launch autoware_launch logging_simulator.launch.xml map_path:=/home/orin/disk/autoware_map/sample-map-rosbag vehicle_model:=sample_vehicle sensor_model:=sample_sensor_kit | tee 1.txt


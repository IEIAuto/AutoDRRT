ros2 launch data_preprocess_launch data_preprocess_launch.xml
ros2 run data_format_dump data_dumper
export ROS_DOMAIN_ID=2
ros2 bag play rosbag2_2024_01_30-01_39_58 -r 0.2
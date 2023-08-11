#!/bin/bash
source ./install/local_setup.bash
ros2 bag play /home/orin/autoware_map/sample-rosbag/sample.db3 -r 0.4 -s sqlite3


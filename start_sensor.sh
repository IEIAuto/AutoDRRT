#!/bin/bash

# 定义Docker容器名称
# CONTAINER_NAME="autoware_humble"

# carla_vehicle_can
gnome-terminal --title="CARLA_VEHICLE_CAN" -- bash -c "
    echo 'Opening carla_vehicle_can terminal...';
    docker exec -it autoware_humble bash -c '
        cd Kunyi_ws/ &&
        source install/setup.bash &&
        ros2 launch carla_vehicle_can carla_vehicle_can.launch.xml
    ';
    exec bash
"

# 等待1秒
sleep 5

# gnss
gnome-terminal --title="CGI430_CAN_DRIVER" -- bash -c "
    echo 'Opening cgi430_can_driver terminal...';
    docker exec -it autoware_humble bash -c '
        cd Kunyi_ws/ &&
        source install/setup.bash &&
        ros2 launch cgi430_can_driver cgi430_can_driver_node.launch.xml
    ';
    exec bash
"

sleep 5
#lidar
gnome-terminal --title="ROBOSENSE_LIDAR" -- bash -c "
    echo 'Opening robosense lidar terminal...';
    docker exec -it autoware_humble bash -c '
        cd Kunyi_ws/ &&
        source install/setup.bash &&
        ros2 launch rslidar_sdk start.py
    ';
    exec bash
"


sleep 5
#vehicle_converse
gnome-terminal --title="VEHICLE_CONVERSE" -- bash -c "
    echo 'Opening vehicle_converse terminal...';
    docker exec -it autoware_humble bash -c '
        cd disk/Kunyi_ws/ &&
        source install/setup.bash &&
        ros2 launch ackermann_control ackermann_control.launch.xml
    ';
    exec bash
"

sleep 5
#localization
gnome-terminal --title="LOCALIZATION" -- bash -c "
    echo 'Opening localization terminal...';
    docker exec -it autoware_humble bash -c '
        cd disk/autodrrt_v2.0 &&
        source install/setup.bash &&
        ros2 launch localization_gnss localization_gnss.launch.xml
    ';
    exec bash
"






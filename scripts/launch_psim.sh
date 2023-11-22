#!/bin/bash

source ./vars/vehicle.env

ros2 launch autoware_launch planning_simulator.launch.xml map_path:=$MAP_PATH vehicle_model:=$VEHICLE_MODEL sensor_model:=$SENSOR_MODEL

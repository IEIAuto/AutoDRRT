#!/bin/bash

source ./vars/vehicle.env
export VEHICLE_ID=GCP02

ros2 launch autoware_launch logging_simulator.launch.xml map_path:=$MAP_PATH vehicle_model:=$VEHICLE_MODEL sensor_model:=$SENSOR_MODEL

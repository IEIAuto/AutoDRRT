#!/bin/bash

### Check LiDAR connection
source ./lidar_connection.sh

### Check CAN connection
source ./can_config.sh

### Set environment
source ./vars/vehicle.env

### Echo environment variables
echo "Launch autoware."
echo "  VEHICLE_ID=$VEHICLE_ID"
echo "  VEHICLE_MODEL=$VEHICLE_MODEL"
echo "  SENSOR_MODEL=$SENSOR_MODEL"
echo "  MAP_PATH=$MAP_PATH"

# Launch autoware
ros2 launch autoware_launch autoware.launch.xml vehicle_model:=$VEHICLE_MODEL sensor_model:=$SENSOR_MODEL map_path:=$MAP_PATH

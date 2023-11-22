#!/bin/bash

### USB NIC connectivity check and VEHICLE_ID identification
# Declare specified USB NIC ID
declare -A NIC_DEV=(
    ["enxc436c0eaa070"]="GCP02"
    ["enx68e1dc12f7f8"]="GCP03"
)

# USB NIC connectivity check
for DEV in `find /sys/devices -name net | grep -v virtual`
do
  DEV_NAME=("$(ls --quoting-style=shell $DEV)")
  if [ -n "${NIC_DEV[$DEV_NAME]}" ]; then
    # Export VEHICLE_ID
    export VEHICLE_ID="${NIC_DEV[$DEV_NAME]}"
  fi
done

if [ -z $VEHICLE_ID ]; then
  echo "Error: Specified USB NIC is not connected."
  exit 1
fi

echo "USB NIC connectivity OK."
echo "VEHICLE_ID=$VEHICLE_ID"

### Lidar connection check
/bin/ping 192.168.1.201 -w 1 -c 1 >> /dev/null
if [ $? == 0 ]; then
  echo "Velodyne connectivity OK."
else
  echo "Error: Please check network settings and vehicle power is ON."
  exit 1
fi

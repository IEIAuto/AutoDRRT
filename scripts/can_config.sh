#!/bin/bash

### PCAN-USB connection check
for USB_DEV in `lsusb`
do
  if [ $USB_DEV = "PCAN-USB" ]; then
    PCAN_USB_CONNECTION=true
  fi
done

if [ "${PCAN_USB_CONNECTION}" ]; then
  echo "PCAN-USB is connected."
else
  echo "Error: PCAN-USB is not connected."
  exit 1
fi

### can0 configuration check
for IF_DEV in `ifconfig`
do
  if [ $IF_DEV = "can0:" ]; then
    CAN_IF_EXIST=true
  fi
done

# If can0 does not exist, configure can0
if [ "${CAN_IF_EXIST}" ]; then
  echo "can0 already configured."
else
  set -e
  echo "Configuring CAN interface..."
  sudo ip link set can0 type can bitrate 500000
  sudo ip link set can0 txqueuelen 500000
  sudo ip link set can0 up
  set +e
  echo "can0 configuring done."
fi

# CAN data reception check
echo "Check candump..."
if [ -z "`candump can0 -n 200`" ]; then
  echo "Error: Please check the CAN cable connection."
  exit 1
else
  echo "CAN Interface configuration done."
fi

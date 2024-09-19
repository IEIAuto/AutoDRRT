# Instructions for using the AWSIM

AWSIM is a simulator for Autoware development and testing, initially developed by TIER IV and still actively maintained. AWSIM Labs is a fork of AWSIM, developed under the Autoware Foundation, providing additional features and lighter resource usage. Take AWSIM v1.2.3 as an example.

## 1 Steps

### 1.1 Localhost settings

[Download AWSIM v1.2.3](https://github.com/tier4/AWSIM/releases/download/v1.2.3/AWSIM_v1.2.3.zip). To apply required localhost settings please add the following lines to `~/.bashrc` file

```
if [ ! -e /tmp/cycloneDDS_configured ]; then
    sudo sysctl -w net.core.rmem_max=2147483647
    sudo ip link set lo multicast on
    touch /tmp/cycloneDDS_configured
fi

export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

Install the library

```
sudo apt install libvulkan1
```

### 1.2 Launching AutoDRRT

[Download map files (pcd, osm)](https://github.com/tier4/AWSIM/releases/download/v1.1.0/nishishinjuku_autoware_map.zip) and unzip them.

Configure the environment. Install dependent ROS packages and build the workspace

```
cd <AutoDRRT_DIR>
source ./setup.bash
ros2 launch autoware_launch e2e_simulator.launch.xml vehicle_model:=sample_vehicle sensor_model:=awsim_sensor_kit map_path:=<your mapfile location>
```

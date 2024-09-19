# Instructions for using the Carla

Each version is much the same, take 0.9.14 as an example.

## 1 Steps

### 1.1 Get the [carla release](https://github.com/carla-simulator/carla/releases)

```
tar -zxvf CARLA_0.9.14.tar.gz
cd CARLA_0.9.14
./CarlaUE4.sh
```

Set manual mode, you can control the vehicle by keypad.

```
# cd <Your_DIR>/CARLA_0.9.14/PythonAPI/examples
python manual_control.py
```

Introduce several APIs

```
# cd <Your_DIR>/CARLA_0.9.14/PythonAPI/examples
# add perpson and car
python generate_traffic.py -w 10 -n 10
# change weather
python dynamic_weather.py -s 5

```

### 1.2 Change launch files

```
# pull the required code
cd <Your_DIR>
mkdir op_carla && cd op_carla/
git clone https://github.com/hatem-darweesh/op_bridge.git -b ros2
git clone https://github.com/hatem-darweesh/op_agent.git -b ros2
git clone https://github.com/hatem-darweesh/scenario_runner.git
```

[Download maps](https://drive.google.com/drive/folders/1Or0CMS08AW8XvJtzzR8TfhqdY9MMUBpS) and rename them.

| original name | revised name |
| --- | --- |
| Town01.pcd | pointcloud_map.pcd |
| Town01.osm | lanelet2_map.osm |

Set Carla IP in file `op_carla/op_bridge/op_scripts/run_exploration_mode_ros2.sh`

```
# local
export SIMULATOR_LOCAL_HOST="localhost"
# other computer.
# Change the IP based on the actual IP address
export SIMULATOR_LOCAL_HOST="10.128.5.25"
```

Add environment variable in file `op_carla/op_agent/start_ros2.sh`

```
# line 15
source <AutoDRRT_DIR>/install/setup.bash
```

In the same file, change the launch file path of AutoDRRT based on the actual installation path, and change the map_path based on the Town01 path.

```
# line 18
# NOTE: autoware.carla.launch.xml is modified based on autoware.launch.xml
ros2 launch \
  <AutoDRRT_DIR>\src\autodrrtscene\launcher\autoware_launch\autoware_launch\launch\autoware.carla.launch.xml \
  map_path:=${YouPath}/autoware_universe/autoware/src/${map_name} \
  vehicle_model:=sample_vehicle \
  sensor_model:=sample_sensor_kit
```

Modify file `autoware.carla.launch.xml`

```
# line 11, add a line to convert the Carla pointcloud to the pointcloud used by AutoDRRT
<arg name="launch_carla_interface" default="true" description="convert carla sensor data to autoware suitable format"/>
# line 19, disable the original sensor drivers
<arg name="launch_sensing_driver" default="false" description="launch sensing driver"/>
# line 26, use sim time
<arg name="use_sim_time" default="true" description="use_sim_time"/>
# Add belowed code
<!-- CARLA -->
<group if="$(var launch_carla_interface)">
    <node pkg="carla_pointcloud" exec="carla_pointcloud_node" name="carla_pointcloud_interface" output="screen"/>
</group>
```

Modify file `sensor_kit_calibration.yaml` refering to the following

```
sensor_kit_base_link:
  camera0/camera_link:
    x: 0.7
    y: 0.0
    z: 0.0
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  camera1/camera_link:
    x: 0.0
    y: 0.0
    z: 0.0
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  camera2/camera_link:
    x: 0.0
    y: 0.0
    z: 0.0
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  camera3/camera_link:
    x: 0.0
    y: 0.0
    z: 0.0
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  camera4/camera_link:
    x: 0.0
    y: 0.0
    z: 0.0
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  camera5/camera_link:
    x: 0.0
    y: 0.0
    z: 0.0
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  traffic_light_right_camera/camera_link:
    x: 0.0
    y: 0.0
    z: 0.0
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  traffic_light_left_camera/camera_link:
    x: 0.0
    y: 0.0
    z: 0.0
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  velodyne_top_base_link:
    x: 0.0
    y: 0.0
    z: 0.8
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  velodyne_left_base_link:
    x: -0.5
    y: 0.0
    z: 0.8
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  velodyne_right_base_link:
    x: 0.5
    y: 0.0
    z: 0.8
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  gnss_link:
    x: 0.0
    y: 0.0
    z: 0.8
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  tamagawa/imu_link:
    x: 0.0
    y: 0.0
    z: 0.8
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
```

Modify file `sensors_calibration.yaml` refering to the following

```
base_link:
  sensor_kit_base_link:
    x: 0.0
    y: 0.0
    z: 1.6
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  velodyne_rear_base_link:
    x: 0.0
    y: 0.0
    z: 0.0
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
```

Setting environment variables. If you use Bash, you can modify `.bashrc` refering to following

```
export CARLA_ROOT=<Carla_DIR>
export SCENARIO_RUNNER_ROOT=<Your_DIR>/op_carla/scenario_runner
export LEADERBOARD_ROOT=<Your_DIR>/op_carla/op_bridge
export TEAM_CODE_ROOT=<Your_DIR>/op_carla/op_agent
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/util
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
```

## 2 Run simulator

```
# rebuild packages
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select autoware_launch sample_sensor_kit_description
# install some app
pip install py-trees networkx tabulate transforms3d
sudo apt-get install ros-humble-sensor-msgs-py
# run carla
cd $CARLA_ROOT
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600
```

Open another terminal, set carla-ros2 bridge

```
cd <Your_DIR>/op_carla/op_bridge/op_scripts
source <AutoDRRT_DIR>/install/setup.bash
./run_exploration_mode_ros2.sh
```
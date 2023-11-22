# aip_launch

## Directory Structure

```bash
.
├── aip_x1_launch # AI.Pilot X1 package
│   └── ... # some directories and files
├── aip_x2_launch
│   └── ...
├── aip_xx1_launch
│   └── ...
├── ... # other AI.Pilot packages...
└── common_sensor_launch # common sensor driver package
    └── ... # some directories and files
```

## Description for `aip_*_launch`

### Overview

These are packages which contain some launch files for launching AI.Pilot sensors.

When you create your own sensor kit packages, please refer to `aip_*_launch` as a sample.

### Directory Structure (ex. aip_x1)

```bash
.
├── aip_x1_launch
│   ├── CMakeLists.txt
│   ├── data
│   │   └── ... # some parameters
│   ├── launch
│   │   ├── camera.launch.xml
│   │   ├── gnss.launch.xml
│   │   ├── imu.launch.xml
│   │   ├── lidar.launch.xml
│   │   ├── sensing.launch.xml
│   │   └── ... # other launch files
│   └── package.xml
└── ... # other AI.Pilot packages...
```

### Description for launch files

#### sensing.launch.xml

- `sensing.launch.xml` is called from `sensing_launch/sensing.launch.xml` in `autoware_launcher`.

```xml
# https://github.com/tier4/autoware_launcher/blob/7e07eaa954bc04d252cda6b65bb7eae050ebdfb2/sensing_launch/package.xml

  <let name="sensor_launch_pkg" value="$(find-pkg-share $(var sensor_model)_launch)"/> # ex. sensor_model = aip_x1
...
  <include file="$(var sensor_launch_pkg)/launch/sensing.launch.xml">
...
```

- sensor drivers such as LiDAR driver can be described as follows.

```xml
...
<!-- LiDAR Driver -->
<include file="$(find-pkg-share aip_xxx_launch)/launch/lidar.launch.xml">
    <arg name="launch_driver" value="$(var launch_driver)" />
    <arg name="vehicle_mirror_param_file" value="$(var vehicle_mirror_param_file)" />
</include>

<!-- Camera Driver -->
<include file="$(find-pkg-share aip_xxx_launch)/launch/camera.launch.xml">
    <arg name="launch_driver" value="$(var launch_driver)" />
</include>

<!-- IMU Driver -->
<include file="$(find-pkg-share aip_xxx_launch)/launch/imu.launch.xml">
    <arg name="launch_driver" value="$(var launch_driver)" />
</include>

<!-- GNSS Driver -->
<include file="$(find-pkg-share aip_xxx_launch)/launch/gnss.launch.xml">
    <arg name="launch_driver" value="$(var launch_driver)" />
</include>
...
```

- Other necessary sensor drivers also can be allowed to describe, but please follow above examples.

#### lidar.launch.xml

In localization, to refer to LiDAR data,

- The LiDAR output name is specified as `/sensing/lidar/top/rectified/pointcloud`.
- The LiDAR driver container name is specified as `/sensing/lidar/top/pointcloud_preprocessor/velodyne_node_container`.

These are referred from `util.launch.xml` in `localization_launch`.

```xml
# https://github.com/tier4/autoware_launcher/blob/1ccc97034f84f67d8ff000a308b58ffa9be58091/localization_launch/launch/util/util.launch.xml

...
  <arg name="input_sensor_points_topic" default="/sensing/lidar/top/rectified/pointcloud" description="input topic name for raw pointcloud"/>
...
  <arg name="container" default="/sensing/lidar/top/pointcloud_preprocessor/velodyne_node_container"  description="container name"/>
...
```

#### imu.launch.xml

The IMU output name is specified as `/sensing/imu/imu_data` to refer from `gyro_odometer` package.

```xml
# https://github.com/tier4/autoware.iv/blob/05a0bec8350c867c4e7bce3fd2f3e63bc8c9168e/localization/twist_estimator/gyro_odometer/launch/gyro_odometer.launch.xml

...
  <arg name="input_imu_topic" default="/sensing/imu/imu_data" description="input imu topic name" />
...
```

#### gnss.launch.xml

The GNSS output name is specified as `/sensing/gnss/pose_with_covariance` to refer from `pose_initializer` package.

```xml
# https://github.com/tier4/autoware.iv/blob/883951fb3eae8722fd896218d6798b5e19cebc5c/localization/util/pose_initializer/launch/pose_initializer.launch.xml

...
  <remap from="gnss_pose_cov" to="/sensing/gnss/pose_with_covariance" />
...
```

#### camera.launch.xml

There are no constraints now.

## Description for common_sensor_launch

This is the package that contains some sensor driver launch files for reasons such as performing autoware's own processing.

### Directory Structure

```bash
.
└── common_sensor_launch
    ├── CMakeLists.txt
    └── launch
    │   ├── velodyne_node_container.launch.py
    │   ├── velodyne_VLP16.launch.xml
    │   └── ...
    └── package.xml
```

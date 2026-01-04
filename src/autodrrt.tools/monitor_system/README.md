# ROS2 Monitor System

A comprehensive monitoring system based on ROS2 for monitoring ROS2 node states, system resources, and topic status.

## 📋 Overview

This project provides three independent monitoring modules for real-time monitoring of ROS2 system health:

- **node_state_monitor**: Monitors the running status of ROS2 nodes
- **system_monitor**: Monitors system resources (CPU, GPU, memory, network, etc.)
- **topic_state_monitor**: Monitors the status and frequency of ROS2 topics

All monitoring results are published through ROS2 diagnostic messages (diagnostic_msgs), making it easy to integrate into existing monitoring and diagnostic systems.
## 🚀 Quick Start

### Building the Project
```
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

## 📦 Module Description

### 1. Node State Monitor

Monitors the running status of ROS2 nodes and checks if nodes are running normally.

#### Features

- Detects online/offline status of ROS2 nodes
- Publishes diagnostic information about node status
- Configurable list of monitored nodes

#### Usage

```bash
# Launch with default configuration
ros2 launch node_state_monitor node_monitor.launch.xml

# Launch with custom configuration file
ros2 launch node_state_monitor node_monitor.launch.xml param_path:=/path/to/config.yaml
```

#### Configuration Parameters

Configuration file location: `src/node_state_monitor/config/node_monitor_params.yaml`

| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `vx_threshold` | double | 0.1 | Velocity threshold [m/s] |
| `wz_threshold` | double | 0.02 | Angular velocity threshold [rad/s] |

#### Output

- `/diagnostics`: `diagnostic_msgs/DiagnosticArray` - Diagnostic information about node status

---

### 2. System Monitor

Monitors system hardware resources including CPU, GPU, memory, and network status.

#### Features

- **CPU Monitoring**: Load, temperature, frequency
- **GPU Monitoring**: Load, temperature, frequency
- **Memory Monitoring**: Usage rate, temperature, read/write speed
- **Network Monitoring**: Upload/download rate

#### Usage

```bash
# Use default network interface (wlan0)
ros2 launch system_monitor system_monitor.launch.xml

# Specify network interface
ros2 launch system_monitor system_monitor.launch.xml network_card_id:=eth0
```

#### Parameters

| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `network_card_id` | string | wlan0 | Network interface name |

#### Output

- `/diagnostics`: `diagnostic_msgs/DiagnosticArray` - Diagnostic information about system resources

---

### 3. Topic State Monitor

Monitors the status of ROS2 topics and detects issues such as timeouts and frequency abnormalities.

#### Features

- Monitors topic reception status
- Detects topic frequency abnormalities (too low)
- Detects topic timeouts
- Supports Transform (TF) topic monitoring
- Configurable QoS policies

#### Topic Status Types

| Topic Status | Diagnostic Status | Description |
|-------------|-------------------|-------------|
| `OK` | OK | Topic is normal with no abnormalities |
| `NotReceived` | ERROR | Topic has not been received yet |
| `WarnRate` | WARN | Topic frequency has dropped |
| `ErrorRate` | ERROR | Topic frequency has significantly dropped |
| `Timeout` | ERROR | Topic subscription stopped for more than specified time |

#### Usage

```bash
# Monitor regular topic
ros2 launch topic_state_monitor topic_state_monitor.launch.xml \
  node_name_suffix:=camera \
  topic:=/camera/image_raw \
  topic_type:=sensor_msgs/msg/Image \
  diag_name:=camera_monitor \
  warn_rate:=10.0 \
  error_rate:=5.0 \
  timeout:=1.0

# Monitor TF topic
ros2 launch topic_state_monitor topic_state_monitor_tf.launch.xml \
  node_name_suffix:=base_link \
  frame_id:=map \
  child_frame_id:=base_link \
  diag_name:=tf_monitor \
  warn_rate:=10.0 \
  error_rate:=5.0 \
  timeout:=1.0
```

#### Parameters

**Node Parameters**

| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `topic` | string | - | Name of target topic to monitor |
| `topic_type` | string | - | Topic type (used for non-TF topics) |
| `frame_id` | string | - | TF parent frame ID (used for TF topics) |
| `child_frame_id` | string | - | TF child frame ID (used for TF topics) |
| `transient_local` | bool | false | QoS policy: Transient Local |
| `best_effort` | bool | false | QoS policy: Best Effort |
| `diag_name` | string | - | Name used for diagnostics |
| `update_rate` | double | 10.0 | Timer callback period [Hz] |

**Core Parameters**

| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `warn_rate` | double | 0.5 | Warning frequency threshold [Hz] |
| `error_rate` | double | 0.1 | Error frequency threshold [Hz] |
| `timeout` | double | 1.0 | Timeout period [s] |
| `window_size` | int | 10 | Window size for calculating frequency |

#### Output

- `/diagnostics`: `diagnostic_msgs/DiagnosticArray` - Diagnostic information about topic status

For detailed documentation, please refer to: `src/topic_state_monitor/README.md`


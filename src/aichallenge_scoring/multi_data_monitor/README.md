# Multi Data Monitor

This is an rviz2 plugin to monitor topics. [See here for package details.](./multi_data_monitor/README.md)

![multi-data-monitor-with-autoware](https://github.com/tier4/multi_data_monitor/assets/43976882/d35b1173-493b-4617-a371-02aeb360e95c)

## Installation

```bash
mkdir -p multi_data_monitor_ws/src
cd multi_data_monitor_ws/src
git clone https://github.com/tier4/multi_data_monitor.git
git clone https://github.com/tier4/generic_type_utility.git
cd ..
colcon build --symlink-install
```

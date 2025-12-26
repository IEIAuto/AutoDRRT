## Heterogeneous Computing Scheduling Overview

This repository provides **real-time CPU scheduling + GPU/DLA scheduling enhancements** for Autodrrt. It focuses on improving **latency, determinism and stability**, while keeping autodrrt modules non-intrusive and engineering-friendly.

Core ideas:

- Linux Real-Time Scheduling (SCHED_RR)
- ROS2 Executor level thread control
- CPU core affinity binding
- GPU multi-model priority scheduling
- GPU + DLA parallel scheduling
- CPU ↔ GPU async pipeline decoupling

---

## Background & Motivation

The scheduling framework is based on the autoware pipeline.

We focus on improving the end-to-end latency of the following chain:

```
e2e → lidar_top → map_based_prediction → planning → vehicle_cmd_gate
```

## Key Modules Under Scheduling

We selectively enable scheduling for CPU/GPU heavy components:

- `ndt_scan_matcher`
- `ekf_localizer`
- `pointcloud_container`
- `planning_container`
- `control_container`

The principle is:
> apply scheduling only where it brings clear benefit, while keeping the rest unchanged.


## CPU Scheduling Design

### Design Principles

1️⃣ Each critical node runs in a **dedicated executor thread**  
2️⃣ Threads are **pinned to specific CPU cores** based on resource analysis  
3️⃣ Linux **real-time policy SCHED_RR** is applied  

This ensures:
- lower latency
- reduced jitter
- predictable runtime behavior
- less CPU contention between ROS processes



## Node-Level Enhancements

### ndt_scan_matcher Scheduling

In `ndt_scan_matcher_node.cpp`:

- Create a custom executor thread
- Bind to selected CPU cores
- Apply `SCHED_RR` real-time policy

Result: more stable scan matching and reduced spikes.



### ekf_localizer Scheduling

In `ekf_localizer_node.cpp`:

- CPU affinity binding
- Real-time scheduling enabled
- Lower computation delay


### pointcloud_container Scheduling

Point cloud preprocessing is highly CPU-intensive.  
If left unscheduled, ROS resource competition can cause:

- CPU contention
- reduced efficiency
- occasional stutter

However, since this is a **component container with no main()**, we:

- created `custom_pointcloud_container`
- added a standalone main node
- switched launching strategy from “component loading” to “scheduled node”

📂 Location:
```
autodrrt.scene/custom_pointcloud_container
```

👉 **Launch file modification**
The main idea is to change the container start in this launch.py to node start

```markdown
TODO: update pointcloud_container.launch.py example here
```


### control_container Scheduling

Similar to pointcloud:

- high CPU usage
- originally launched as a component
- converted to schedulable standalone process

📂 Package:
```
autodrrt.scene/custom_control_container
```

👉 **Launch changes**
```
autodrrt.core/launch/tier4_control_launch/launch/control_launch.py
```

```markdown
TODO: add launch details here
```


### planning_container Scheduling

Same strategy:

- created `custom_planning_container`
- replaced component launch mode

Modified files:

```
autodrrt.core/launch/tier4_planning_launch/launch/scenario_planning/lane_driver/
  behavior_planning_launch.xml
  motion_planning.launch.xml
```

```markdown
TODO: add launch details here
```


## GPU Scheduling Framework

Further optimization:

- CenterPoint encoder → GPU
- CenterPoint backbone → DLA0
- YOLOX → DLA1

Final:
- CenterPoint ~21 ms total
- ≈ 30% latency reduction


## gpu_utils Package

📂 Location

```
autodrrt.application/perception/gpu_utils
```

✔ priority scheduling via CUDA stream priority  
✔ model capability detection (GPU / DLA)  
✔ async CPU/GPU pipeline support  


## How to Run 

Example placeholders:
To use this dependency library, you need to include gpu_stream_manager.hpp in lidarcenterpoint and yolox, and then change the stream entry to gpu_stream_manager's stream. 
For example, for lidar_centerpoint, I need to put cudaStreamCreate(&stream_); in the constructor of centerpoint_trt.cpp. Replaced with stream_ = GPUStreamManagher::instance().high();


## Summary

This scheduling framework provides:

- lower latency
- lower jitter
- deterministic execution
- engineering-ready integration
- no intrusive modification to autodrrt nodes

It has been validated on real autodrrt workloads.

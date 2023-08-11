# Usage Instructions

## Demo Run Instructions


1.  Grant container access to display GUI: Open a terminal (Terminal A) and enter `xhost +` and press Enter.

2.  Enter the container: Run `docker exec --it autodrrt  /bin/bash` and press Enter.

3.  Navigate to the Autoware directory: Run `cd /home/orin/autoware/`.

4.  Open another terminal (Terminal B) and follow the previous two steps.

5.  In Terminal A, run `./start_demo.sh` and wait for the rviz2 interface to appear, displaying the map.
    ![avatar](../imgs/rviz.jpeg)

6.  In Terminal B, run `./logging_play.sh` to start the demo.
    ![avatar](../imgs/demo.jpeg)

## Adding Custom Algorithms

This framework is developed based on Autoware.universe. The process of inserting custom functionality into the framework is the same as modifying Autoware.universe.

Follow these steps:

1.  Place your custom code in the code directory. It is recommended to place it in the corresponding location within the `src` directory.
    ![avatar](../imgs/add_src.png)

2.  Configure the input-output mapping for the algorithm:
    ![avatar](../imgs/remap.png)

    In the startup configuration file of the algorithm, configure the input-output relationships to ensure proper functionality.

3.  Build: Execute the colcon build command in the autoware directory to compile your custom algorithm.

4.  Run `source install/setup.bash`.

## Distributed Tools Usage Instructions

> The distributed tools allow you to easily transform a single-machine application into a multi-machine application. Follow these steps:

1.  Open the configuration file: The configuration file path is `{workspace}/autoware/install/launch/share/launch/param/node_config.param.yaml`.

2.  Configure the file content as follows:
    ![avatar](../imgs/node_param.png)

    > In the configuration file, specify the nodes/containers that will start on the corresponding device nodes. Nodes not specified will start on the current device. The device username is fixed as `root`, and the IP address can be changed to the actual IP in use.

3.  Execute the launch operation as usual.

4.  Shutdown: Simply pressing Ctrl+C may not completely close processes on other devices. Run the `killtask.sh` script in the autoware directory to ensure all processes on the devices are completely closed.

## Fault Tolerance Tools Usage Instructions

1.  Open the configuration file, which is identical to the distributed functionality configuration file.

2.  Modify the `redundant_info` field in the configuration file.

3.  Execute the launch operation as usual.
    ![avatar](../imgs/node_param2.png)

    > Note: This feature requires the fault-tolerant communication feature, which is enabled by default. After enabling it, applications that rely on the ros2 package in Hermes (including x86 and Orin) can communicate with the existing framework. Applications that rely on other ros2 packages may be affected in terms of communication.

## Acceleration Toolkit Usage Instructions

### Overview

This toolkit is located in `/home/orin/acc_tools` and consists of two parts: the General Acceleration Tool and the Low-Latency Function Library. They can be used together or separately as needed.

### General Acceleration Tool

1.  Run environment optimization using `./GeneralAccelerator --opt-env=true`. This needs to be executed **outside the container**. It is already performed by default.

2.  You can also accelerate inference of models using `./GeneralAccelerator --model-path=/path_to_model`. This will generate an optimized TRT model in the model directory. This step requires **an NVIDIA GPU device and is executed within the container**.

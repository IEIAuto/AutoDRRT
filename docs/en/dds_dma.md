# DDS_Opt Usage Instructions

The following example introduces the entire process of using DDS and DMA for cross-SOC big data communication. By using this method, the latency of cross-SOC big data communication can be effectively reduced.


## DMA Settings
1. Use PCIE to connect two jetson orin devices.
2. Refresh BSP and turn on DMA shared data settings to enable large data communication at the OS level.
3. Start two devices to test communication.

## DDS_Opt Data Communication Principles
Users can pay attention to the development of ROS2 upper-layer applications. By calling the API, users can implement DMA for data communication. Taking the ROS2 publish-subscribe mechanism as an example, the working principle is as follows:

1.  The publishing node writes the data into the memory
2.  After the data is written, the publishing node sends a write success signal, which is sent to the subscribing node in the form of a topic through DDS
3.  After the subscribing node receives the signal from the DDS end, it reads the data in the memory for use by the upper-level ROS2 application

> Note: The RP end is the publishing node, and the EP end is the subscribing node.


## DDS_Opt Data Communication Example
The working directory is `src/autodrrt.core/dma_transfer`.
Use `send.cpp` and `recv.cpp` to call related APIs to implement DDS & DMA functions. Users can modify the type according to their own data type. The data type in the example is `sensor_msgs/Image`.

Execute the startup command and use ros2 to complete the transmission of big data. 
`ros2 run dma_transfer demo_node2`


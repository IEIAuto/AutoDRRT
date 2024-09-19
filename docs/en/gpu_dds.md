# IO_Opt Sharing Usage Instructions

Based on GPU shared data to achieve data communication between nodes, by using this method, the communication delay between data can be effectively reduced.

## IO_Opt Data Communication Principles
For common large data types, such as image data, the communication between two ROS2 nodes consumes a lot of time and computing resources. To alleviate this situation, we move the data of the two ROS2 nodes to the GPU and use data sharing between GPUs to complete the data transmission between the two nodes, thereby reducing communication latency.

## IO_Opt Data Communication Example
The working directory is `src/autodrrt.core/io_opt`.
Execute the startup command and use ros2 to complete the transmission of big data. 
`ros2 launch composition composition_demo.launch.py`

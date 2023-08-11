This ROS package is simple converter from .bag file that contains "velodyne_packets" topic to .pcap file.
.pcap file is useful because it can be used to display LiDAR pointclouds using VeloView or VeloVision.

Once rosbag is finished playing, press Ctrl+C to stop the program and generate .pcap file.

Usage: rosrun bag_to_pcap bag_to_pcap.py (topic_name)
For example, rosrun bag_to_pcap bag_to_pcap.py velodyne_packets

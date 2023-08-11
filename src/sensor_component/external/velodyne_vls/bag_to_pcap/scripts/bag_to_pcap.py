#!/usr/bin/env python
import sys
import rospy
from velodyne_msgs.msg import VelodyneScan

import signal
import binascii

#Global header for pcap 2.4
global_header =  'd4c3b2a10200040000000000000000000000010001000000'         # 24 bytes = 48 characters.
packet_header = '0a98945a25250f00e0040000e0040000'      # 16 bytes = 32 characters.
udp_header = 'ffffffffffff607688343ba70800450004d200004000ff11b4a9c0a801c9ffffffff0940094004be0000'     # 42 bytes = 84 characters.

udp_packets = ''

def signal_handler(sig, frame):
    global global_header
    global udp_packets

    print "saving to velodyne_packets.pcap file"
    byte_string = global_header + udp_packets
    # print byte_string

    bytes = binascii.a2b_hex(byte_string)
    # print binascii.hexlify(bytes)
    filename = 'velodyne_packets.pcap'
    bitout = open(filename, 'wb')
    bitout.write(bytes)
    bitout.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def callback(ros_data):
    global packet_header
    global udp_packets
    global udp_header

    num_packets = len(ros_data.packets)
    print num_packets
    for i in range(0, num_packets):         # keep adding packet header, udp header and data
        # print binascii.hexlify(ros_data.packets[i].data)
        udp_packets = udp_packets + packet_header + udp_header + binascii.hexlify(ros_data.packets[i].data)

def listener(listening_topic):
    print(listening_topic), "is being subscribed."
    rospy.init_node('bag_to_pcap', anonymous=True)
    rospy.Subscriber(listening_topic, VelodyneScan, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "usage: rosrun bag_to_pcap bag_to_pcap.py subscribing_topic"
    listening_topic = sys.argv[1]
    listener(listening_topic)

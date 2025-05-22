#!/bin/bash
# array up:0
# array down:1
# echo $1

# if (($1==0))
# then 
#     ros2 topic pub --once /planning/mission_planning/goal geometry_msgs/msg/PoseStamped "{
#     header:
#     {stamp:
#         {sec: 1794,
#         nanosec: 59959899},
#     frame_id: map},
#     pose:
#     {position:
#         {x: 81677.5546875,
#         y: 50297.84765625,
#         z: 0.0},
#     orientation:
#         {x: 0.0,
#         y: 0.0,
#         z: 0.7688108399054929,
#         w: 0.6394762641754661}}
#     }
#     "
# else
    # ros2 topic pub --once /planning/mission_planning/goal geometry_msgs/msg/PoseStamped "{
    #     header:
    #     {stamp:
    #         {sec: 3794,
    #         nanosec: 99959899},
    #     frame_id: map},
    #     pose:
    #     {position:
    #         {x: 81680.296875,
    #         y: 50336.66015625,
    #         z: 0.0},
    #     orientation:
    #         {x: 0.0,
    #         y: 0.0,
    #         z: -0.6188001134135206,
    #         w: 0.7855484833155838}}
    #     }
    #     "
# fi

ros2 topic pub /autoware/engage autoware_auto_vehicle_msgs/msg/Engage '{engage: True}' -1

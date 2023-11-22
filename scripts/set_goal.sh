#!/bin/bash

ros2 topic pub -1 /planning/mission_planning/goal geometry_msgs/msg/PoseStamped 'header:
  stamp: now
  frame_id: "map"
pose:
  position:
    x: 3702.17919921875
    y: 73743.1953125
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.8537888223989135
    w: 0.5206194836410528'

#!/bin/bash

PubDummyObject() {
  ros2 topic pub -1 /simulation/dummy_perception_publisher/object_info dummy_perception_publisher/msg/Object "header:
  stamp: now
  frame_id: "map"
initial_state:
  pose_covariance:
    pose:
      position:
        x: $1
        y: $2
        z: $3
      orientation:
        x: $4
        y: $5
        z: $6
        w: $7
classification:
  label: 0
shape:
  dimensions:
    x: $8
    y: $9
    z: ${10}"
}

# First Wall
PubDummyObject 3796.7 73759.2 0.0 0.0 0.0 -0.9721 0.2347 0.5 1.5 1.0

# Second Wall
PubDummyObject 3786.4 73756.9 0.0 0.0 0.0 -0.9721 0.2347 0.5 1.5 1.0

# Third Wall
PubDummyObject 3778.7 73750.1 0.0 0.0 0.0 -0.9721 0.2347 0.5 2.0 1.0

# Fourth Wall
PubDummyObject 3758.35 73741.8 0.0 0.0 0.0 -0.9721 0.2347 0.5 2.5 1.0

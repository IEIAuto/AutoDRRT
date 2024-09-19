cmake_minimum_required(VERSION 3.14)
project(autoware_perception_msgs)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectClassification.msg"
  "msg/PredictedObject.msg"
  "msg/PredictedObjectKinematics.msg"
  "msg/PredictedObjects.msg"
  "msg/PredictedPath.msg"
  "msg/Shape.msg"
  "msg/TrafficSignalElement.msg"
  "msg/TrafficSignal.msg"
  "msg/TrafficSignalArray.msg"

  DEPENDENCIES
    std_msgs
    geometry_msgs
    builtin_interfaces
    unique_identifier_msgs
)

ament_auto_package()

cmake_minimum_required(VERSION 3.14)
project(autoware_perception_msgs)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrafficLight.msg"
  "msg/TrafficLightArray.msg"
  "msg/TrafficLightElement.msg"
  "msg/TrafficSignal.msg"
  "msg/TrafficSignalArray.msg"
  DEPENDENCIES
    builtin_interfaces
)

ament_auto_package()

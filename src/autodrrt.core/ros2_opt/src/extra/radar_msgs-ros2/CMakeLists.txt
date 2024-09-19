cmake_minimum_required(VERSION 3.5)
project(radar_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(unique_identifier_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RadarReturn.msg"
  "msg/RadarScan.msg"
  "msg/RadarTrack.msg"
  "msg/RadarTracks.msg"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs unique_identifier_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()

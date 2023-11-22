cmake_minimum_required(VERSION 3.14)
project(aichallenge_scoring_msgs)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Score.msg"
  "msg/Result.msg"
)

ament_auto_package()

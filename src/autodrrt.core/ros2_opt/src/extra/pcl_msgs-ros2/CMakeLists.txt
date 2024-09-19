cmake_minimum_required(VERSION 3.5)

project(pcl_msgs)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# declare the message files to generate code for
set(msg_files
  "msg/ModelCoefficients.msg"
  "msg/PointIndices.msg"
  "msg/PolygonMesh.msg"
  "msg/Vertices.msg"
)

# declare the service files to generate code for
set(srv_files
  "srv/UpdateFilename.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
   ${msg_files}
   ${srv_files}
   DEPENDENCIES sensor_msgs std_msgs
   ADD_LINTER_TESTS
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()

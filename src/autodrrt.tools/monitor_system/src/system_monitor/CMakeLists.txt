cmake_minimum_required(VERSION 3.8)
project(system_monitor)

####################################################################
# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
#############################

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find  ROS2 dependencies############################################
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(diagnostic_msgs REQUIRED)
###############################

# include head file##################################################
include_directories(
  include
)

add_library(system_monitor SHARED
  src/system_monitor.cpp
)

set(lib_dependencies
  rclcpp
  std_msgs
  diagnostic_updater
  diagnostic_msgs
)
ament_target_dependencies(system_monitor ${lib_dependencies})
#############################################################

# build exe ##########################################################
add_executable(system_monitor_node src/system_monitor_node.cpp)
ament_target_dependencies(system_monitor_node ${lib_dependencies})
target_link_libraries(system_monitor_node system_monitor)
###############################################################

# install ############################################################
install(TARGETS
  system_monitor_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}   #程序运行的时候调用install中的路径
)

install(TARGETS
  system_monitor
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
###########################################

ament_package()

cmake_minimum_required(VERSION 3.8)
project(aion_interface)

####################################################################
# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
#####################################################################

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find ROS2 dependencies ############################################
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(autoware_auto_control_msgs REQUIRED)
find_package(autoware_auto_vehicle_msgs REQUIRED)
#####################################################################

#include head file #####################################################
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(aion_interface SHARED
  src/aion_interface.cpp
)

set(lib_dependencies
  rclcpp
  std_msgs
  autoware_auto_control_msgs
  autoware_auto_vehicle_msgs
)
ament_target_dependencies(aion_interface ${lib_dependencies})
target_link_libraries(aion_interface ${PROJECT_SOURCE_DIR}/lib/libcontrolcan.so)
##########################################################################

# build exe ##############################################################
add_executable(aion_interface_node src/aion_interface_node.cpp)
ament_target_dependencies(aion_interface_node ${lib_dependencies})
target_link_libraries(aion_interface_node aion_interface)
###########################################################################

# install ##################################################################
install(TARGETS
  aion_interface_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}   #程序运行的时候调用install中的路径
)

install(TARGETS
  aion_interface
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

################################################################################
ament_package()

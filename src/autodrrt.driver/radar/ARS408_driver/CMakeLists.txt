cmake_minimum_required(VERSION 3.8)
project(ARS408_driver)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(radar_msgs REQUIRED)

#########################################################################
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(ARS408_CAN_core SHARED
  src/ARS408_CAN_core.cpp
)
set(lib_dependencies
  rclcpp
  radar_msgs
  std_msgs
)
ament_target_dependencies(ARS408_CAN_core ${lib_dependencies})
target_link_libraries(ARS408_CAN_core Eigen3::Eigen ${PROJECT_SOURCE_DIR}/lib/libcontrolcan.so)
#################################################################################

add_executable(ARS408_CAN_node src/ARS408_CAN_node.cpp)
ament_target_dependencies(ARS408_CAN_node ${lib_dependencies})
target_link_libraries(ARS408_CAN_node ARS408_CAN_core)
##################################################################################

install(TARGETS 
  ARS408_CAN_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}  #程序运行的时候调用install中的路径
) 

##添加
install(TARGETS
  ARS408_CAN_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
launch
DESTINATION share/${PROJECT_NAME})


ament_package()

cmake_minimum_required(VERSION 3.8)
project(multi_lidar_calibration)

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
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED )
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  include/multi_lidar_calibration
)

set(CMAKE_AUTOMOC ON)

add_library(multi_lidar_calibration SHARED
  src/multi_lidar_calibration.cpp
)

set(lib_dependencies
    rclcpp
    sensor_msgs
    std_msgs
    geometry_msgs
    pcl_ros  
    pcl_conversions
    tf2
    tf2_ros
    tf2_geometry_msgs
)
ament_target_dependencies(multi_lidar_calibration ${lib_dependencies})    
target_link_libraries(multi_lidar_calibration Eigen3::Eigen ${PCL_LIBRARIES} ${OpenCV_INCLUDE_DIRS})

add_executable(multi_lidar_calibration_node src/multi_lidar_calibration_node.cpp)
ament_target_dependencies(multi_lidar_calibration_node ${lib_dependencies})
target_link_libraries(multi_lidar_calibration_node multi_lidar_calibration)

install(TARGETS
  multi_lidar_calibration_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS
  multi_lidar_calibration_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME})


ament_package()

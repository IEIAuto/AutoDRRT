cmake_minimum_required(VERSION 3.8)
project(cgi610_driver)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fPIC)
  message("fpic add")
endif()

#find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(serial REQUIRED)

#################################################
find_package(serial REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${SERIAL_INCLUDE_DIRS}
)
message("SERIAL_INCLUDE_DIRS",${SERIAL_INCLUDE_DIRS})
add_library(cgi610_driver_core SHARED
  src/cgi610_driver_core.cpp
)

set(lib_dependencies
  geometry_msgs
  std_msgs
  rclcpp
  sensor_msgs
  diagnostic_msgs
  tf2_ros
  serial
)
ament_target_dependencies(cgi610_driver_core ${lib_dependencies})
target_link_libraries(cgi610_driver_core Eigen3::Eigen serial)
###########################################################################

add_executable(cgi610_driver_node src/cgi610_driver_node.cpp)
ament_target_dependencies(cgi610_driver_node ${lib_dependencies})
target_link_libraries(cgi610_driver_node cgi610_driver_core)
###########################################################################
install(TARGETS 
  cgi610_driver_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}  #程序运行的时候调用install中的路径
)

install(TARGETS
  cgi610_driver_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
launch
DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

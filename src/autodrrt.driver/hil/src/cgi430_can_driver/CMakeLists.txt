cmake_minimum_required(VERSION 3.8)
project(cgi430_can_driver)

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

# find  ROS2 dependencies############################################
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
#####################################################################
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GeographicLib REQUIRED geographiclib)
# find_package(GeographicLib REQUIRED PATHS /usr NO_DEFAULT_PATH)

# include head file##################################################
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)
include_directories(${GeographicLib_INCLUDE_DIRS})
link_directories(${GeographicLib_LIBRARY_DIRS})

add_library(cgi430_can_driver SHARED
  src/cgi430_can_driver_core.cpp
)

set(lib_dependencies
  rclcpp
  std_msgs
  sensor_msgs
)
ament_target_dependencies(cgi430_can_driver ${lib_dependencies})
target_link_libraries(cgi430_can_driver ${PROJECT_SOURCE_DIR}/lib/libcontrolcan.so ${GeographicLib_LIBRARIES} Eigen3::Eigen)
#####################################################################

# build exe #########################################################
add_executable(cgi430_can_driver_node src/cgi430_can_driver_node.cpp)
ament_target_dependencies(cgi430_can_driver_node ${lib_dependencies})
target_link_libraries(cgi430_can_driver_node cgi430_can_driver)
###########################################################################

# install ############################################################
install(TARGETS
  cgi430_can_driver_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}   #程序运行的时候调用install中的路径
)

install(TARGETS
  cgi430_can_driver
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

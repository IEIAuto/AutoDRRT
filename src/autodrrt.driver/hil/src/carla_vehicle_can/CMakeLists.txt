cmake_minimum_required(VERSION 3.8)
project(carla_vehicle_can)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find  ROS2 dependencies############################################
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tier4_vehicle_msgs REQUIRED)
find_package(autoware_auto_vehicle_msgs REQUIRED)
#####################################################################

# include head file##################################################
include_directories(
  include
)

add_executable(carla_vehicle_can 
  src/agv.c
  src/vehicle_interface.cpp
  src/agv_codec.cpp
  )
target_include_directories(carla_vehicle_can PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(carla_vehicle_can PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  carla_vehicle_can
  "rclcpp"
  "std_msgs"
  "tier4_vehicle_msgs"  
  "autoware_auto_vehicle_msgs"
  "tier4_control_msgs"
)
target_link_libraries(carla_vehicle_can ${PROJECT_SOURCE_DIR}/lib/libcontrolcan.so)

install(TARGETS carla_vehicle_can
  DESTINATION lib/${PROJECT_NAME})

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

# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()

cmake_minimum_required(VERSION 3.5)
project(velodyne_driver)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# This driver uses Boost threads
find_package(Boost REQUIRED COMPONENTS thread)

# libpcap provides no pkg-config or find_package module:
set(libpcap_LIBRARIES -lpcap)

include_directories(${Boost_INCLUDE_DIR})
include_directories(include)

# compile the driver and input library
add_subdirectory(src/lib)
# add_subdirectory(src/driver)

ament_auto_add_library(velodyne_driver SHARED
  src/driver/driver.cc
  src/driver/driver.h
  src/driver/nodelet.cc
  # src/driver/driver.cpp
)
target_link_libraries(velodyne_driver velodyne_input)

rclcpp_components_register_node(velodyne_driver
  PLUGIN "velodyne_driver::VelodyneDriver"
  EXECUTABLE velodyne_driver_node
)

ament_auto_package(
  INSTALL_TO_SHARE
  launch
)

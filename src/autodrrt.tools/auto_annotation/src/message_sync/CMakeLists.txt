cmake_minimum_required(VERSION 3.8)
project(message_sync)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
# find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(rclcpp REQUIRED)
find_package(autoware_auto_perception_msgs REQUIRED)
find_package(message_filters REQUIRED)

include_directories( 
  src/
  ${message_filters_INCLUDE_DIRS}
)

link_directories(
  ${message_filters_LIBRARIES}
)

ament_auto_add_executable(message_sync_node
  src/main.cpp
)
ament_target_dependencies(message_sync_node)

# ament_auto_add_library(message_sync SHARED
#   src/main.cpp
# )

# rclcpp_components_register_node(message_sync
#   PLUGIN "MessageSync"
#   EXECUTABLE message_sync_node
# )
# ament_auto_package(INSTALL_TO_SHARE)
ament_auto_package(
  INSTALL_TO_SHARE
    launch
)
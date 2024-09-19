cmake_minimum_required(VERSION 3.14)
project(behavior_path_avoidance_module)

find_package(autoware_cmake REQUIRED)
autoware_package()
pluginlib_export_plugin_description_file(behavior_path_planner plugins.xml)
find_package( tf2_geometry_msgs REQUIRED)
# find_package( geometry_msgs REQUIRED)
# find_package( tf2_geometry_msgs REQUIRED)
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/scene.cpp
  src/debug.cpp
  src/utils.cpp
  src/manager.cpp
  src/shift_line_generator.cpp
)
target_include_directories(${PROJECT_NAME} 
PRIVATE 
src
${tf2_geometry_msgs_INCLUDE_DIRS}
)
message("tf2_geometry_msgs",${tf2_geometry_msgs_INCLUDE_DIRS})
if(BUILD_TESTING)
  ament_add_ros_isolated_gmock(test_${PROJECT_NAME}
    test/test_utils.cpp
    test/test_behavior_path_planner_node_interface.cpp
  )

  target_link_libraries(test_${PROJECT_NAME}
    ${PROJECT_NAME}
  )

  target_include_directories(test_${PROJECT_NAME} 
  PRIVATE 
  src
  ${tf2_geometry_msgs_INCLUDE_DIRS}
  )
endif()

ament_auto_package(INSTALL_TO_SHARE config)

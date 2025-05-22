cmake_minimum_required(VERSION 3.5)

project(gpudirect-dds)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
find_package(CUDA REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rcutils REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(include)

# create ament index resource which references the libraries in the binary dir
set(node_plugins "")

add_library(talker_component SHARED
  src/talker_component.cpp)
target_compile_definitions(talker_component
  PRIVATE "COMPOSITION_BUILDING_DLL")
ament_target_dependencies(talker_component
  "rclcpp"
  "rclcpp_components"
  "std_msgs"
  "CUDA")
rclcpp_components_register_nodes(talker_component "gpudirect::Talker")
set(node_plugins "${node_plugins}gpudirect::Talker;$<TARGET_FILE:talker_component>\n")

add_library(listener_component SHARED
  src/listener_component.cpp)
target_compile_definitions(listener_component
  PRIVATE "COMPOSITION_BUILDING_DLL")
ament_target_dependencies(listener_component
  "rclcpp"
  "rclcpp_components"
  "std_msgs"
  "CUDA")
rclcpp_components_register_nodes(listener_component "gpudirect::Listener")
set(node_plugins "${node_plugins}gpudirect::Listener;$<TARGET_FILE:listener_component>\n")




# since the package installs libraries without exporting them
# it needs to make sure that the library path is being exported
if(NOT WIN32)
  ament_environment_hooks(
    "${ament_cmake_package_templates_ENVIRONMENT_HOOK_LIBRARY_PATH}")
endif()


install(TARGETS
  talker_component
  listener_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)



# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

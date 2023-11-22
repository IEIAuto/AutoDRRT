cmake_minimum_required(VERSION 3.8)
project(aichallenge_scoring_result)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(aichallenge_scoring_msgs REQUIRED)
find_package(autoware_auto_vehicle_msgs REQUIRED)

add_executable(scoring_result src/scoring_result.cpp)

ament_target_dependencies(scoring_result rclcpp std_msgs geometry_msgs aichallenge_scoring_msgs autoware_auto_vehicle_msgs)


install(TARGETS
  scoring_result
  DESTINATION lib/${PROJECT_NAME})

ament_package()

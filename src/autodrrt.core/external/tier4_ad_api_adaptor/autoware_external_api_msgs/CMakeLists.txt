cmake_minimum_required(VERSION 3.14)
project(autoware_external_api_msgs)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/ResponseStatus.msg
  srv/GetVersion.srv
  DEPENDENCIES
    tier4_external_api_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()

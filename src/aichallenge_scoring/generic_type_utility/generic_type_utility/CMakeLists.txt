cmake_minimum_required(VERSION 3.14)
project(generic_type_utility)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/generic_property.cpp
  src/generic_message.cpp
  src/ros2/introspection.cpp
  src/ros2/serialization.cpp
  src/ros2/message.cpp
  src/ros2/structure.cpp
)

ament_auto_add_executable(print
  src/examples/print.cpp
)

ament_auto_add_executable(check
  src/examples/check.cpp
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  ament_add_gtest(${PROJECT_NAME}_test test/test.cpp)
  target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
endif()

ament_auto_package()

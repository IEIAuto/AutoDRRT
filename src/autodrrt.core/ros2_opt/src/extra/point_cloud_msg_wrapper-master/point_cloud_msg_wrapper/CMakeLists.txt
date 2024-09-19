# Copyright 2021 Apex.AI, Inc.
# All rights reserved.
cmake_minimum_required(VERSION 3.5)
project(point_cloud_msg_wrapper)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include/)

if(BUILD_TESTING)
    # run common linters
    find_package(ament_lint_auto)
    ament_lint_auto_find_test_dependencies()
    ament_add_gtest(${PROJECT_NAME}_test
        test/test_field_generators.cpp
        test/test_point_cloud_msg_wrapper.cpp
    )
    target_compile_options(${PROJECT_NAME}_test PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wall
        -Wconversion
        -Wsign-conversion
        -Werror
        -Wcast-align
        -Wunused
        -Wsign-conversion
        -Wdouble-promotion)

    ament_target_dependencies(${PROJECT_NAME}_test sensor_msgs geometry_msgs)
    target_include_directories(${PROJECT_NAME}_test PRIVATE include)
    target_include_directories(${PROJECT_NAME}_test SYSTEM PRIVATE ${GTEST_INCLUDE_DIRS})

endif()

ament_auto_package()

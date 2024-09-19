cmake_minimum_required(VERSION 3.5)
project(filters)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

##############################################################################
# Find dependencies
##############################################################################

find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Boost REQUIRED)

##############################################################################
# Build
##############################################################################

add_library(real_time_circular_buffer INTERFACE)
target_include_directories(real_time_circular_buffer INTERFACE
  include
  ${Boost_INCLUDE_DIRS})

# Plugins
add_library(mean SHARED src/mean.cpp)
target_link_libraries(mean
  real_time_circular_buffer)
ament_target_dependencies(mean
  "pluginlib"
  "rclcpp"
)

add_library(params SHARED src/test_params.cpp)
target_link_libraries(params
  real_time_circular_buffer)
ament_target_dependencies(params
  "pluginlib"
  "rclcpp"
)

add_library(increment SHARED src/increment.cpp)
target_link_libraries(increment
  real_time_circular_buffer)
ament_target_dependencies(increment
  "pluginlib"
  "rclcpp"
)

add_library(median SHARED src/median.cpp)
target_link_libraries(median
  real_time_circular_buffer)
ament_target_dependencies(median
  "pluginlib"
  "rclcpp"
)

add_library(transfer_function SHARED src/transfer_function.cpp)
target_link_libraries(transfer_function
  real_time_circular_buffer)
ament_target_dependencies(transfer_function
  "pluginlib"
  "rclcpp"
)

if(BUILD_TESTING)
  # TODO(hidmic): use ament_lint_common once ament_copyright complains are resolved
  find_package(ament_cmake_cppcheck REQUIRED)
  find_package(ament_cmake_cpplint REQUIRED)
  find_package(ament_cmake_uncrustify REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)

  # Enforce ROS2 style
  ament_cpplint()
  ament_cppcheck(
    LANGUAGE c++  # Force C++ for .h headers
    INCLUDE_DIRS include
  )
  ament_uncrustify()
  ament_xmllint()

  find_package(ament_cmake_gtest REQUIRED)
  # Test median filter
  ament_add_gtest(median_test test/test_median.cpp)
  target_link_libraries(median_test median)

  # Test transfer function filter
  ament_add_gtest(transfer_function_test test/test_transfer_function.cpp)
  target_link_libraries(transfer_function_test transfer_function)

  # Test mean filter
  ament_add_gtest(mean_test test/test_mean.cpp)
  target_link_libraries(mean_test mean)

  # Test params filter
  ament_add_gtest(params_test test/test_params.cpp)
  target_link_libraries(params_test params)

  # Test realtime safe buffer class
  ament_add_gtest(realtime_buffer_test test/test_realtime_circular_buffer.cpp)
  target_link_libraries(realtime_buffer_test real_time_circular_buffer)

  # Pluginlib specific testing for filter chain
  set(test_prefix "${CMAKE_CURRENT_BINARY_DIR}/pluginlib_test_prefix")
  set(test_package_name "${PROJECT_NAME}")
  set(test_plugins_xml "default_plugins.xml")
  set(test_package_xml "package.xml")
  set(test_plugin_targets mean increment median transfer_function)

  # Write the necessary files to detect a package with pluginlib plugins
  file(COPY "${test_plugins_xml}" DESTINATION "${test_prefix}/share/${test_package_name}/")
  configure_file("${test_package_xml}" "${test_prefix}/share/${test_package_name}/package.xml" COPYONLY)
  file(WRITE "${test_prefix}/share/ament_index/resource_index/${test_package_name}__pluginlib__plugin/pluginlib"
    "share/${test_package_name}/${test_plugins_xml}\n")
  file(WRITE "${test_prefix}/share/ament_index/resource_index/packages/${test_package_name}" "")

  # Copy library files to test folder so they can be loaded
  foreach(test_plugin_target IN LISTS test_plugin_targets)
    set(_resource "$<TARGET_FILE:${test_plugin_target}>")
    set(_output "${test_prefix}/lib/$<TARGET_FILE_NAME:${test_plugin_target}>")
    add_custom_command(TARGET ${test_plugin_target}
      COMMENT "Copying target ${test_plugin_target} for testing"
      DEPENDS "${_resource}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_resource}"
        "${_output}"
    )
    add_custom_target("copy_${test_plugin_target}_for_testing" DEPENDS "${test_plugin_target}")
  endforeach()

  # Test plugin loading into filter chain
  ament_add_gtest(chain_test test/test_chain.cpp
    ENV "AMENT_PREFIX_PATH=${test_prefix}")
  target_include_directories(chain_test PRIVATE include)
  foreach(test_plugin_target IN LISTS test_plugin_targets)
    add_dependencies(chain_test "copy_${test_plugin_target}_for_testing")
  endforeach()
  ament_target_dependencies(chain_test
    "rclcpp"
    "pluginlib"
  )
endif()

##############################################################################
# Install
##############################################################################

# Install libraries
install(
  TARGETS mean increment median transfer_function
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/
  DESTINATION include)

# Install pluginlib xml
pluginlib_export_plugin_description_file(filters "default_plugins.xml")
ament_export_include_directories(include)
ament_package()

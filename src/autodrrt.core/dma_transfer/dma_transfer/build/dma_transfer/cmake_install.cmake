# Install script for directory: /home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/install/dma_transfer")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node1" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node1")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node1"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/dma_transfer" TYPE EXECUTABLE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/demo_node1")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node1" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node1")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node1"
         OLD_RPATH "/home/orin/disk/autodrrt_v2.0/install/rclcpp_components/lib:/home/orin/disk/autodrrt_v2.0/install/rclcpp/lib:/home/orin/disk/autodrrt_v2.0/install/libstatistics_collector/lib:/home/orin/disk/autodrrt_v2.0/install/rcl/lib:/home/orin/disk/autodrrt_v2.0/install/rmw_implementation/lib:/home/orin/disk/autodrrt_v2.0/install/rcl_logging_spdlog/lib:/home/orin/disk/autodrrt_v2.0/install/rcl_logging_interface/lib:/home/orin/disk/autodrrt_v2.0/install/rcl_yaml_param_parser/lib:/home/orin/disk/autodrrt_v2.0/install/libyaml_vendor/lib:/home/orin/disk/autodrrt_v2.0/install/rosgraph_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/statistics_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/tracetools/lib:/home/orin/disk/autodrrt_v2.0/install/class_loader/lib:/home/orin/disk/autodrrt_v2.0/install/console_bridge_vendor/lib:/home/orin/disk/autodrrt_v2.0/install/ament_index_cpp/lib:/home/orin/disk/autodrrt_v2.0/install/composition_interfaces/lib:/home/orin/disk/autodrrt_v2.0/install/rcl_interfaces/lib:/home/orin/disk/autodrrt_v2.0/install/builtin_interfaces/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_fastrtps_c/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_fastrtps_cpp/lib:/home/orin/disk/autodrrt_v2.0/install/rmw/lib:/home/orin/disk/autodrrt_v2.0/install/fastcdr/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_introspection_cpp/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_introspection_c/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_cpp/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_c/lib:/home/orin/disk/autodrrt_v2.0/install/rcpputils/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_runtime_c/lib:/home/orin/disk/autodrrt_v2.0/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node1")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/CMakeFiles/demo_node1.dir/install-cxx-module-bmi-noconfig.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node2" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node2")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node2"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/dma_transfer" TYPE EXECUTABLE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/demo_node2")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node2" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node2")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node2"
         OLD_RPATH "/home/orin/disk/autodrrt_v2.0/install/rclcpp_components/lib:/home/orin/disk/autodrrt_v2.0/install/rclcpp/lib:/home/orin/disk/autodrrt_v2.0/install/libstatistics_collector/lib:/home/orin/disk/autodrrt_v2.0/install/rcl/lib:/home/orin/disk/autodrrt_v2.0/install/rmw_implementation/lib:/home/orin/disk/autodrrt_v2.0/install/rcl_logging_spdlog/lib:/home/orin/disk/autodrrt_v2.0/install/rcl_logging_interface/lib:/home/orin/disk/autodrrt_v2.0/install/rcl_yaml_param_parser/lib:/home/orin/disk/autodrrt_v2.0/install/libyaml_vendor/lib:/home/orin/disk/autodrrt_v2.0/install/rosgraph_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/statistics_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/tracetools/lib:/home/orin/disk/autodrrt_v2.0/install/class_loader/lib:/home/orin/disk/autodrrt_v2.0/install/console_bridge_vendor/lib:/home/orin/disk/autodrrt_v2.0/install/ament_index_cpp/lib:/home/orin/disk/autodrrt_v2.0/install/composition_interfaces/lib:/home/orin/disk/autodrrt_v2.0/install/rcl_interfaces/lib:/home/orin/disk/autodrrt_v2.0/install/builtin_interfaces/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_fastrtps_c/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_fastrtps_cpp/lib:/home/orin/disk/autodrrt_v2.0/install/rmw/lib:/home/orin/disk/autodrrt_v2.0/install/fastcdr/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_introspection_cpp/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_introspection_c/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_cpp/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_c/lib:/home/orin/disk/autodrrt_v2.0/install/rcpputils/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_runtime_c/lib:/home/orin/disk/autodrrt_v2.0/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/dma_transfer/demo_node2")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/CMakeFiles/demo_node2.dir/install-cxx-module-bmi-noconfig.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/include/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer/environment" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/install/ament_package/lib/python3.8/site-packages/ament_package/template/environment_hook/library_path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer/environment" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_environment_hooks/library_path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_transfer.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_transfer.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_transfer.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/libdma_transfer.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_transfer.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_transfer.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_transfer.so"
         OLD_RPATH "/home/orin/disk/autodrrt_v2.0/install/rclcpp_components/lib:/home/orin/disk/autodrrt_v2.0/install/cv_bridge/lib:/home/orin/disk/autodrrt_v2.0/install/nav_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/dma_customer_msg/lib:/home/orin/disk/autodrrt_v2.0/install/autoware_auto_perception_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/class_loader/lib:/home/orin/disk/autodrrt_v2.0/install/console_bridge_vendor/lib:/home/orin/disk/autodrrt_v2.0/install/composition_interfaces/lib:/home/orin/disk/autodrrt_v2.0/install/rclcpp/lib:/home/orin/disk/autodrrt_v2.0/install/libstatistics_collector/lib:/home/orin/disk/autodrrt_v2.0/install/rcl/lib:/home/orin/disk/autodrrt_v2.0/install/rmw_implementation/lib:/home/orin/disk/autodrrt_v2.0/install/ament_index_cpp/lib:/home/orin/disk/autodrrt_v2.0/install/rcl_logging_spdlog/lib:/home/orin/disk/autodrrt_v2.0/install/rcl_logging_interface/lib:/home/orin/disk/autodrrt_v2.0/install/rcl_interfaces/lib:/home/orin/disk/autodrrt_v2.0/install/rcl_yaml_param_parser/lib:/home/orin/disk/autodrrt_v2.0/install/libyaml_vendor/lib:/home/orin/disk/autodrrt_v2.0/install/rosgraph_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/statistics_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/tracetools/lib:/home/orin/disk/autodrrt_v2.0/install/sensor_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/autoware_auto_geometry_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/geometry_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/std_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/builtin_interfaces/lib:/home/orin/disk/autodrrt_v2.0/install/unique_identifier_msgs/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_fastrtps_c/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_fastrtps_cpp/lib:/home/orin/disk/autodrrt_v2.0/install/fastcdr/lib:/home/orin/disk/autodrrt_v2.0/install/rmw/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_introspection_cpp/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_introspection_c/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_cpp/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_typesupport_c/lib:/home/orin/disk/autodrrt_v2.0/install/rcpputils/lib:/home/orin/disk/autodrrt_v2.0/install/rosidl_runtime_c/lib:/home/orin/disk/autodrrt_v2.0/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_transfer.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/dma_transfer")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/dma_transfer")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer/environment" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/install/ament_cmake_core/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer/environment" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer/environment" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/install/ament_cmake_core/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer/environment" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_environment_hooks/path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_environment_hooks/local_setup.bash")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_environment_hooks/local_setup.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_environment_hooks/package.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_index/share/ament_index/resource_index/packages/dma_transfer")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/rclcpp_components" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_index/share/ament_index/resource_index/rclcpp_components/dma_transfer")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer/cmake" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_export_dependencies/ament_cmake_export_dependencies-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer/cmake" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_export_include_directories/ament_cmake_export_include_directories-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer/cmake" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_export_libraries/ament_cmake_export_libraries-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer/cmake" TYPE FILE FILES
    "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_core/dma_transferConfig.cmake"
    "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/ament_cmake_core/dma_transferConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_transfer" TYPE FILE FILES "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/orin/disk/autodrrt_v2.0/src/autodrrt.core/dma_transfer/dma_transfer/build/dma_transfer/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")

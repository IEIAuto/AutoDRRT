# Install script for directory: /home/orin/disk/demos-humble/io_opt

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/orin/disk/demos-humble/io_opt/install/io_opt")
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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt/environment" TYPE FILE FILES "/home/orin/ros2/install/ament_package/lib/python3.8/site-packages/ament_package/template/environment_hook/library_path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt/environment" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_environment_hooks/library_path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtalker_component.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtalker_component.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtalker_component.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/libtalker_component.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtalker_component.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtalker_component.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtalker_component.so"
         OLD_RPATH "/home/orin/ros2/install/rclcpp_components/lib:/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/rclcpp/lib:/home/orin/ros2/install/libstatistics_collector/lib:/home/orin/ros2/install/rcl/lib:/home/orin/ros2/install/rmw_implementation/lib:/home/orin/ros2/install/rcl_logging_spdlog/lib:/home/orin/ros2/install/rcl_logging_interface/lib:/home/orin/ros2/install/rcl_yaml_param_parser/lib:/home/orin/ros2/install/libyaml_vendor/lib:/home/orin/ros2/install/rosgraph_msgs/lib:/home/orin/ros2/install/statistics_msgs/lib:/home/orin/ros2/install/tracetools/lib:/home/orin/ros2/install/ament_index_cpp/lib:/home/orin/ros2/install/class_loader/lib:/data/source/v1.0/ros2/install/console_bridge_vendor/lib:/home/orin/ros2/install/composition_interfaces/lib:/home/orin/ros2/install/rcl_interfaces/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_c/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_cpp/lib:/home/orin/ros2/install/fastcdr/lib:/home/orin/ros2/install/rmw/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_cpp/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_c/lib:/home/orin/ros2/install/rosidl_typesupport_cpp/lib:/home/orin/ros2/install/rosidl_typesupport_c/lib:/home/orin/ros2/install/rcpputils/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtalker_component.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liblistener_component.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liblistener_component.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liblistener_component.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/liblistener_component.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liblistener_component.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liblistener_component.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liblistener_component.so"
         OLD_RPATH "/home/orin/ros2/install/rclcpp_components/lib:/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/rclcpp/lib:/home/orin/ros2/install/libstatistics_collector/lib:/home/orin/ros2/install/rcl/lib:/home/orin/ros2/install/rmw_implementation/lib:/home/orin/ros2/install/rcl_logging_spdlog/lib:/home/orin/ros2/install/rcl_logging_interface/lib:/home/orin/ros2/install/rcl_yaml_param_parser/lib:/home/orin/ros2/install/libyaml_vendor/lib:/home/orin/ros2/install/rosgraph_msgs/lib:/home/orin/ros2/install/statistics_msgs/lib:/home/orin/ros2/install/tracetools/lib:/home/orin/ros2/install/ament_index_cpp/lib:/home/orin/ros2/install/class_loader/lib:/data/source/v1.0/ros2/install/console_bridge_vendor/lib:/home/orin/ros2/install/composition_interfaces/lib:/home/orin/ros2/install/rcl_interfaces/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_c/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_cpp/lib:/home/orin/ros2/install/fastcdr/lib:/home/orin/ros2/install/rmw/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_cpp/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_c/lib:/home/orin/ros2/install/rosidl_typesupport_cpp/lib:/home/orin/ros2/install/rosidl_typesupport_c/lib:/home/orin/ros2/install/rcpputils/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liblistener_component.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt" TYPE DIRECTORY FILES "/home/orin/disk/demos-humble/io_opt/launch")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/io_opt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/io_opt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt/environment" TYPE FILE FILES "/home/orin/ros2/install/ament_cmake_core/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt/environment" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt/environment" TYPE FILE FILES "/home/orin/ros2/install/ament_cmake_core/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt/environment" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_environment_hooks/path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_environment_hooks/local_setup.bash")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_environment_hooks/local_setup.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_environment_hooks/package.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_index/share/ament_index/resource_index/packages/io_opt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/rclcpp_components" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_index/share/ament_index/resource_index/rclcpp_components/io_opt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt/cmake" TYPE FILE FILES
    "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_core/io_optConfig.cmake"
    "/home/orin/disk/demos-humble/io_opt/build/io_opt/ament_cmake_core/io_optConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/io_opt" TYPE FILE FILES "/home/orin/disk/demos-humble/io_opt/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/orin/disk/demos-humble/io_opt/build/io_opt/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")

# Install script for directory: /home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/install/dma_customer_msg")
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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/rosidl_interfaces" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_index/share/ament_index/resource_index/rosidl_interfaces/dma_customer_msg")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dma_customer_msg/dma_customer_msg" TYPE DIRECTORY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_generator_c/dma_customer_msg/" REGEX "/[^/]*\\.h$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/environment" TYPE FILE FILES "/home/orin/ros2/install/ament_package/lib/python3.8/site-packages/ament_package/template/environment_hook/library_path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/environment" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_environment_hooks/library_path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_c.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/libdma_customer_msg__rosidl_generator_c.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_c.so"
         OLD_RPATH "/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_c.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dma_customer_msg/dma_customer_msg" TYPE DIRECTORY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_typesupport_fastrtps_c/dma_customer_msg/" REGEX "/[^/]*\\.cpp$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_c.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/libdma_customer_msg__rosidl_typesupport_fastrtps_c.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_c.so"
         OLD_RPATH "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg:/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_cpp/lib:/home/orin/ros2/install/fastcdr/lib:/home/orin/ros2/install/rmw/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_c/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_c.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dma_customer_msg/dma_customer_msg" TYPE DIRECTORY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_generator_cpp/dma_customer_msg/" REGEX "/[^/]*\\.hpp$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dma_customer_msg/dma_customer_msg" TYPE DIRECTORY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_typesupport_fastrtps_cpp/dma_customer_msg/" REGEX "/[^/]*\\.cpp$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_cpp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_cpp.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/libdma_customer_msg__rosidl_typesupport_fastrtps_cpp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_cpp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_cpp.so"
         OLD_RPATH "/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_cpp/lib:/home/orin/ros2/install/fastcdr/lib:/home/orin/ros2/install/rmw/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_fastrtps_cpp.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dma_customer_msg/dma_customer_msg" TYPE DIRECTORY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_typesupport_introspection_c/dma_customer_msg/" REGEX "/[^/]*\\.h$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_c.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/libdma_customer_msg__rosidl_typesupport_introspection_c.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_c.so"
         OLD_RPATH "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg:/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_c/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_c.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_c.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/libdma_customer_msg__rosidl_typesupport_c.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_c.so"
         OLD_RPATH "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg:/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rosidl_typesupport_c/lib:/home/orin/ros2/install/rcpputils/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_c.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dma_customer_msg/dma_customer_msg" TYPE DIRECTORY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_typesupport_introspection_cpp/dma_customer_msg/" REGEX "/[^/]*\\.hpp$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_cpp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_cpp.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/libdma_customer_msg__rosidl_typesupport_introspection_cpp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_cpp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_cpp.so"
         OLD_RPATH "/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_cpp/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_c/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_introspection_cpp.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_cpp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_cpp.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/libdma_customer_msg__rosidl_typesupport_cpp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_cpp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_cpp.so"
         OLD_RPATH "/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rosidl_typesupport_cpp/lib:/home/orin/ros2/install/rosidl_typesupport_c/lib:/home/orin/ros2/install/rcpputils/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_typesupport_cpp.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/environment" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_environment_hooks/pythonpath.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/environment" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_environment_hooks/pythonpath.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg-0.0.0-py3.8.egg-info" TYPE DIRECTORY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_python/dma_customer_msg/dma_customer_msg.egg-info/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg" TYPE DIRECTORY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_generator_py/dma_customer_msg/" REGEX "/[^/]*\\.pyc$" EXCLUDE REGEX "/\\_\\_pycache\\_\\_$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(
        COMMAND
        "/usr/bin/python3.8" "-m" "compileall"
        "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/install/dma_customer_msg/lib/python3.8/site-packages/dma_customer_msg"
      )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_introspection_c.cpython-38-aarch64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_introspection_c.cpython-38-aarch64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_introspection_c.cpython-38-aarch64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg" TYPE SHARED_LIBRARY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_generator_py/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_introspection_c.cpython-38-aarch64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_introspection_c.cpython-38-aarch64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_introspection_c.cpython-38-aarch64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_introspection_c.cpython-38-aarch64-linux-gnu.so"
         OLD_RPATH "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_generator_py/dma_customer_msg:/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg:/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rmw/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_c/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_cpp/lib:/home/orin/ros2/install/fastcdr/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_cpp/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_c/lib:/home/orin/ros2/install/rosidl_typesupport_cpp/lib:/home/orin/ros2/install/rosidl_typesupport_c/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcpputils/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_introspection_c.cpython-38-aarch64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_fastrtps_c.cpython-38-aarch64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_fastrtps_c.cpython-38-aarch64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_fastrtps_c.cpython-38-aarch64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg" TYPE SHARED_LIBRARY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_generator_py/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_fastrtps_c.cpython-38-aarch64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_fastrtps_c.cpython-38-aarch64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_fastrtps_c.cpython-38-aarch64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_fastrtps_c.cpython-38-aarch64-linux-gnu.so"
         OLD_RPATH "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_generator_py/dma_customer_msg:/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg:/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rmw/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_c/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_cpp/lib:/home/orin/ros2/install/fastcdr/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_cpp/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_c/lib:/home/orin/ros2/install/rosidl_typesupport_cpp/lib:/home/orin/ros2/install/rosidl_typesupport_c/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcpputils/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_fastrtps_c.cpython-38-aarch64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_c.cpython-38-aarch64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_c.cpython-38-aarch64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_c.cpython-38-aarch64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg" TYPE SHARED_LIBRARY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_generator_py/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_c.cpython-38-aarch64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_c.cpython-38-aarch64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_c.cpython-38-aarch64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_c.cpython-38-aarch64-linux-gnu.so"
         OLD_RPATH "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_generator_py/dma_customer_msg:/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg:/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rmw/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_c/lib:/home/orin/ros2/install/rosidl_typesupport_fastrtps_cpp/lib:/home/orin/ros2/install/fastcdr/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_cpp/lib:/home/orin/ros2/install/rosidl_typesupport_introspection_c/lib:/home/orin/ros2/install/rosidl_typesupport_cpp/lib:/home/orin/ros2/install/rosidl_typesupport_c/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcpputils/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages/dma_customer_msg/dma_customer_msg_s__rosidl_typesupport_c.cpython-38-aarch64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_py.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_py.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_py.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_generator_py/dma_customer_msg/libdma_customer_msg__rosidl_generator_py.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_py.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_py.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_py.so"
         OLD_RPATH "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg:/home/orin/ros2/install/std_msgs/lib:/home/orin/ros2/install/builtin_interfaces/lib:/home/orin/ros2/install/rosidl_typesupport_c/lib:/home/orin/ros2/install/rosidl_runtime_c/lib:/home/orin/ros2/install/rcpputils/lib:/home/orin/ros2/install/rcutils/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdma_customer_msg__rosidl_generator_py.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/msg" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_adapter/dma_customer_msg/msg/Dmamessage.idl")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/msg" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/msg/Dmamessage.msg")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/dma_customer_msg")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/dma_customer_msg")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/environment" TYPE FILE FILES "/home/orin/ros2/install/ament_cmake_core/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/environment" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/environment" TYPE FILE FILES "/home/orin/ros2/install/ament_cmake_core/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/environment" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_environment_hooks/path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_environment_hooks/local_setup.bash")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_environment_hooks/local_setup.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_environment_hooks/package.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_index/share/ament_index/resource_index/packages/dma_customer_msg")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_generator_cExport.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_generator_cExport.cmake"
         "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_generator_cExport.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_generator_cExport-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_generator_cExport.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_generator_cExport.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_generator_cExport-noconfig.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_typesupport_fastrtps_cExport.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_typesupport_fastrtps_cExport.cmake"
         "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_typesupport_fastrtps_cExport.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_typesupport_fastrtps_cExport-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_typesupport_fastrtps_cExport.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_typesupport_fastrtps_cExport.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_typesupport_fastrtps_cExport-noconfig.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_generator_cppExport.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_generator_cppExport.cmake"
         "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_generator_cppExport.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_generator_cppExport-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_generator_cppExport.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_generator_cppExport.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_typesupport_fastrtps_cppExport.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_typesupport_fastrtps_cppExport.cmake"
         "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_typesupport_fastrtps_cppExport.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_typesupport_fastrtps_cppExport-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_typesupport_fastrtps_cppExport.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_typesupport_fastrtps_cppExport.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_typesupport_fastrtps_cppExport-noconfig.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_introspection_cExport.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_introspection_cExport.cmake"
         "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/dma_customer_msg__rosidl_typesupport_introspection_cExport.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_introspection_cExport-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_introspection_cExport.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/dma_customer_msg__rosidl_typesupport_introspection_cExport.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/dma_customer_msg__rosidl_typesupport_introspection_cExport-noconfig.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_cExport.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_cExport.cmake"
         "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/dma_customer_msg__rosidl_typesupport_cExport.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_cExport-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_cExport.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/dma_customer_msg__rosidl_typesupport_cExport.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/dma_customer_msg__rosidl_typesupport_cExport-noconfig.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_introspection_cppExport.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_introspection_cppExport.cmake"
         "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/dma_customer_msg__rosidl_typesupport_introspection_cppExport.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_introspection_cppExport-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_introspection_cppExport.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/dma_customer_msg__rosidl_typesupport_introspection_cppExport.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/dma_customer_msg__rosidl_typesupport_introspection_cppExport-noconfig.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_cppExport.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_cppExport.cmake"
         "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/dma_customer_msg__rosidl_typesupport_cppExport.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_cppExport-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/dma_customer_msg__rosidl_typesupport_cppExport.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/dma_customer_msg__rosidl_typesupport_cppExport.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/dma_customer_msg__rosidl_typesupport_cppExport-noconfig.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_generator_pyExport.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_generator_pyExport.cmake"
         "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_generator_pyExport.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_generator_pyExport-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake/export_dma_customer_msg__rosidl_generator_pyExport.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_generator_pyExport.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/Export/365c94749722a1408b4d75a0f0b01192/export_dma_customer_msg__rosidl_generator_pyExport-noconfig.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_cmake/rosidl_cmake-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_export_dependencies/ament_cmake_export_dependencies-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_export_include_directories/ament_cmake_export_include_directories-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_export_libraries/ament_cmake_export_libraries-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_export_targets/ament_cmake_export_targets-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_cmake/rosidl_cmake_export_typesupport_targets-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/rosidl_cmake/rosidl_cmake_export_typesupport_libraries-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg/cmake" TYPE FILE FILES
    "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_core/dma_customer_msgConfig.cmake"
    "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_core/dma_customer_msgConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dma_customer_msg" TYPE FILE FILES "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/package.xml")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/dma_customer_msg__py/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")

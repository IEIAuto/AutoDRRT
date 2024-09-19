# Copyright 2020-2022 The Autoware Foundation, Arm Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

project(tvm_vendor)

find_package(ros_environment REQUIRED)

set(ROS_VERSION $ENV{ROS_VERSION})

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(${ROS_VERSION} EQUAL 1)
  cmake_minimum_required(VERSION 2.8.3)

  # find dependencies
  find_package(catkin REQUIRED)
elseif(${ROS_VERSION} EQUAL 2)
  cmake_minimum_required(VERSION 3.5)

  if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()

  # find dependencies
  find_package(ament_cmake REQUIRED)
endif()

include(ExternalProject)

set(PATCH_FILE ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.patch)
if(ROS_VERSION EQUAL 2)
  set(PATCH_FILE ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.gpu.patch)
endif()

ExternalProject_Add(tvm-build
  GIT_REPOSITORY https://github.com/apache/tvm
  GIT_TAG da7b48f9487c9ee8eb8c6a6d7f80b59969f842c8 # v0.10.0
  GIT_SHALLOW TRUE
  BUILD_IN_SOURCE TRUE
  PATCH_COMMAND git reset --hard && git apply ${PATCH_FILE}
  COMMAND cp <SOURCE_DIR>/cmake/config.cmake <SOURCE_DIR>
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -DCMAKE_POSITION_INDEPENDENT_CODE=ON .
  INSTALL_COMMAND ""
)

# Build DMLC
ExternalProject_Add_Step(tvm-build dmlc-build
  DEPENDEES patch
  WORKING_DIRECTORY <SOURCE_DIR>/3rdparty/dmlc-core
  COMMAND ${CMAKE_COMMAND} -DCMAKE_POSITION_INDEPENDENT_CODE=ON .
  COMMAND ${CMAKE_MAKE_PROGRAM}
  BYPRODUCTS libdmlc.a
)

ExternalProject_Get_property(tvm-build SOURCE_DIR)

add_library(tvm SHARED IMPORTED
  IMPORTED_LOCATION ${SOURCE_DIR}/libtvm.so
)

add_library(tvm_runtime SHARED IMPORTED
  IMPORTED_LOCATION ${SOURCE_DIR}/libtvm_runtime.so
)

add_library(dmlc STATIC IMPORTED
  IMPORTED_LOCATION ${SOURCE_DIR}/3rdparty/dmlc-core
)

if(${ROS_VERSION} EQUAL 1)
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES
      tvm
      tvm_runtime
      dmlc
    EXPORTED_TARGETS tvm-build
  )

  add_custom_command(TARGET tvm-build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/include/tvm_vendor/tvm
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${SOURCE_DIR}/include/tvm/runtime/ ${CMAKE_CURRENT_SOURCE_DIR}/include/tvm_vendor/tvm/runtime
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CATKIN_DEVEL_PREFIX}/lib
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_DIR}/libtvm.so ${CATKIN_DEVEL_PREFIX}/lib
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_DIR}/libtvm_runtime.so ${CATKIN_DEVEL_PREFIX}/lib
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/include/tvm_vendor/dlpack
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${SOURCE_DIR}/3rdparty/dlpack/include/dlpack/dlpack.h ${CMAKE_CURRENT_SOURCE_DIR}/include/tvm_vendor/dlpack/dlpack.h
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${SOURCE_DIR}/3rdparty/dlpack/contrib/dlpack/dlpackcpp.h ${CMAKE_CURRENT_SOURCE_DIR}/include/tvm_vendor/dlpack/dlpackcpp.h
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${SOURCE_DIR}/3rdparty/dmlc-core/include/dmlc/ ${CMAKE_CURRENT_SOURCE_DIR}/include/tvm_vendor/dmlc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${SOURCE_DIR}/3rdparty/dmlc-core/libdmlc.a ${CATKIN_DEVEL_PREFIX}/lib/libdmlc.a
  )

  install(DIRECTORY ${SOURCE_DIR}/include/tvm/runtime
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/tvm
  )

  install(
    FILES
      ${SOURCE_DIR}/libtvm.so
      ${SOURCE_DIR}/libtvm_runtime.so
    DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  )

  install(
    FILES
      ${SOURCE_DIR}/3rdparty/dlpack/include/dlpack/dlpack.h
      ${SOURCE_DIR}/3rdparty/dlpack/contrib/dlpack/dlpackcpp.h
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/dlpack
  )

  install(DIRECTORY ${SOURCE_DIR}/3rdparty/dmlc-core/include/dmlc
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  )
  
  install(FILES ${SOURCE_DIR}/3rdparty/dmlc-core/libdmlc.a
    DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  )
elseif(${ROS_VERSION} EQUAL 2)
  ament_export_include_directories(include include/tvm_vendor)
  ament_export_libraries(tvm tvm_runtime)

  install(DIRECTORY ${SOURCE_DIR}/include/tvm/runtime/
    DESTINATION include/${PROJECT_NAME}/tvm/runtime
  )

  install(
    FILES
      ${SOURCE_DIR}/libtvm.so
      ${SOURCE_DIR}/libtvm_runtime.so
    DESTINATION lib
  )

  install(
    FILES
      ${SOURCE_DIR}/3rdparty/dlpack/include/dlpack/dlpack.h
      ${SOURCE_DIR}/3rdparty/dlpack/contrib/dlpack/dlpackcpp.h
    DESTINATION include/${PROJECT_NAME}/dlpack
  )

  install(DIRECTORY ${SOURCE_DIR}/3rdparty/dmlc-core/include/dmlc
    DESTINATION include/${PROJECT_NAME}
  )

  install(FILES ${SOURCE_DIR}/3rdparty/dmlc-core/libdmlc.a
    DESTINATION lib
  )

  install(
    FILES
      cmake/tvm_runtimeConfig.cmake
      cmake/tvmConfig.cmake
    DESTINATION share/${PROJECT_NAME}/cmake)

  ament_package()
endif()

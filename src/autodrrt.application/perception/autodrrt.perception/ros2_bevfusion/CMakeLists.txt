# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.5)
project(bevfusion)

find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

#LANGUAGES CXX CUDA
#set(Torch_DIR /opt/libtorch/share/cmake/Torch)
#set(pcl_DIR /)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)
set(CMAKE_BUILD_PYTHON "ON")
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

#ament_auto_find_build_dependencies()
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)
find_package(rclcpp REQUIRED)
find_package(autoware_auto_perception_msgs REQUIRED)
find_package(tier4_autoware_utils REQUIRED)

#find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(message_filters REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")
#set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3 -DENABLE_TEXT_BACKEND_STB")
set(CUDA_NVCC_FLAGS_RELEASE "-Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3 -DENABLE_TEXT_BACKEND_STB")
set(CUDA_NVCC_FLAGS_DEBUG "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_87,code=sm_87)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_$ENV{CUDASM},code=compute_$ENV{CUDASM})

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++17)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CMAKE_C_COMPILER /usr/bin/gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/g++)
  set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
endif()

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)

set(CMAKE_BUILD_TYPE "RELEASE")

set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O2")

add_compile_options(-W)
add_compile_options(-std=c++14)

set(SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 86 87)
foreach(sm ${SMS})
        set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -ccbin ${CMAKE_CXX_COMPILER}
)

if(${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")       # orin
  set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu/)
  set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu/)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(TENSORRT_INCLUDE_DIRS /root/TensorRT-8.5.2.2/include)  # your tensorrt path
  set(TENSORRT_LIBRARY_DIRS /root/TensorRT-8.5.2.2/lib)
endif()


set(CUDA_LIB_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/lib)
find_library(NVJPEG_LIBRARY nvjpeg ${CUDA_LIB_DIRS})
if(NVJPEG_LIBRARY)
  add_definitions(-D__HAVE_NVJPEG__)
  link_libraries(${NVJPEG_LIBRARY})
  message(STATUS ${NVJPEG_LIBRARY})
endif()


include_directories(
  ${Protobuf_INCLUDE_DIRS}
  $ENV{CUDA_Inc}
  $ENV{TensorRT_Inc}
  $ENV{Python_Inc}
  /usr/include/python3.8/
  ../dependencies/stb
  ../dependencies/pybind11/include
  ../libraries/cuOSD/src
  ../libraries/3DSparseConvolution/libspconv/include
  include/
  src/common
  src/
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  /home/orin/ros2/install/cv_bridge/include/
  #${TORCH_INCLUDE_DIRS}
  ${message_filters_INCLUDE_DIRS}
  /home/orin/ros2/install/message_filters/include/
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
)

link_directories(
  #$ENV{CUDA_Lib}
  /usr/local/cuda-11.4/lib64/
  $ENV{TensorRT_Lib}
  ../libraries/3DSparseConvolution/libspconv/lib/${arch}
  build
  /usr/lib/python3.8/
  $ENV{Python_Lib}
  ${cv_bridge_LIBRARIES}
  /home/orin/ros2/install/cv_bridge/lib/
  #${TORCH_LIBRARIES}
  /opt/ros/galactic/lib/
  ${PCL_LIBRARIES}
  ${message_filters_LIBRARIES}
  /home/orin/ros2/install/message_filters/lib/
  ${TENSORRT_LIBRARY_DIRS}
  ${libcv_bridge_LIBRARIES}
  libcv_bridge.so
)

file(GLOB_RECURSE CORE_FILES 
  src/bevfusion/*.cu 
  src/bevfusion/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
  src/common/threadPool.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES 
  ../libraries/cuOSD/src/*.cpp
  ../libraries/cuOSD/src/*.cu
  src/common/visualize.cu
)

####################### libbevfusion_core.so ##########################
cuda_add_library(bevfusion_core SHARED 
  ${CORE_FILES}
)

target_link_libraries(bevfusion_core
  libcudart.so 
  libcublasLt.so
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so 
)
#######################################################################


####################### bevfusion program #############################
cuda_add_executable(bevfusion 
  #demo_bevdet.cpp
  #src/main_nuscenes.cpp
  src/main_ros2.cpp
  ${VISUALIZE_FILES}
  src/bevdet/bevdet.cpp
  src/bevdet/preprocess.cu
  src/bevdet/bevpool.cu
  src/bevdet/float2half_cuda.cu
  src/bevdet/iou3d_nms.cu
  src/bevdet/postprocess.cu
  src/bevdet/data.cpp
  src/bevdet/grid_sampler.cu
  src/bevdet/cpu_jpegdecoder.cpp
  src/bevdet/nvjpegdecoder.cpp
)

#set(CMAKE_CUDA_STANDARD 14)
#set_property(TARGET bevfusion PROPERTY CUDA_STANDARD 14)

target_link_libraries(bevfusion
  bevfusion_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  ${OpenCV_LIBRARIES}
  libcv_bridge.so
  #${TORCH_LIBRARIES}
  libmessage_filters.so
  ${PCL_LIBRARIES}
  yaml-cpp
  libjpeg.so
)

#######################################################################


########################## libpybev.so ################################
if (${CMAKE_BUILD_PYTHON} STREQUAL "ON")
  cuda_add_library(pybev SHARED
    src/python.cpp
  )

  target_link_libraries(pybev
    bevfusion_core
    libnvinfer.so
    libspconv.so
    ${Protobuf_LIBRARIES}
    libnvinfer_plugin.so
    $ENV{Python_Soname}
  )
endif()
#######################################################################

ament_target_dependencies(bevfusion rclcpp std_msgs autoware_auto_perception_msgs tier4_autoware_utils sensor_msgs message_filters pcl_conversions cv_bridge ament_index_cpp) 

#ament_auto_package(
#  INSTALL_TO_SHARE
#    model
#    cfgs
#    sample0
#)

install(TARGETS bevfusion
  DESTINATION lib/${PROJECT_NAME})

ament_package()

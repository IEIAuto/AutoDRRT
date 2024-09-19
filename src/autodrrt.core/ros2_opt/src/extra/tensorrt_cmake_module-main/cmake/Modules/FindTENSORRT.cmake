# Copyright 2022 Tier IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(FindPackageHandleStandardArgs)

if(NOT DEFINED TENSORRT_ROOT)
  if(DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
  else()
    set(TENSORRT_ROOT "/usr" CACHE PATH "Folder contains NVIDIA TensorRT")
  endif()
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_NVINFER_LIBRARY nvinfer
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_NVINFER_PLUGIN_LIBRARY nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_NVPARSERS_LIBRARY nvparsers
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_NVONNXPARSER_LIBRARY nvonnxparser
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY
  ${TENSORRT_NVINFER_LIBRARY}
  ${TENSORRT_NVINFER_PLUGIN_LIBRARY}
  ${TENSORRT_NVPARSERS_LIBRARY}
  ${TENSORRT_NVONNXPARSER_LIBRARY})
message(STATUS "Found TensorRT libs at ${TENSORRT_LIBRARY}")

if(TENSORRT_INCLUDE_DIR)
  if(EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
    file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" TENSORRT_H_CONTENTS)
  else()
    file(READ "${TENSORRT_INCLUDE_DIR}/NvInfer.h" TENSORRT_H_CONTENTS)
  endif()

  string(REGEX MATCH "define NV_TENSORRT_MAJOR ([0-9]+)" _ "${TENSORRT_H_CONTENTS}")
  set(TENSORRT_VERSION_MAJOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
  string(REGEX MATCH "define NV_TENSORRT_MINOR ([0-9]+)" _ "${TENSORRT_H_CONTENTS}")
  set(TENSORRT_VERSION_MINOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
  string(REGEX MATCH "define NV_TENSORRT_PATCH ([0-9]+)" _ "${TENSORRT_H_CONTENTS}")
  set(TENSORRT_VERSION_PATCH ${CMAKE_MATCH_1} CACHE INTERNAL "")

  set(TENSORRT_VERSION
    "${TENSORRT_VERSION_MAJOR}.${TENSORRT_VERSION_MINOR}.${TENSORRT_VERSION_PATCH}"
    CACHE
    STRING
    "TensorRT version"
  )

  unset(TENSORRT_H_CONTENTS)
endif()

find_package_handle_standard_args(TENSORRT
  FOUND_VAR TENSORRT_FOUND
  REQUIRED_VARS
    TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY
  VERSION_VAR TENSORRT_VERSION
)

if(TENSORRT_FOUND)
  set(TENSORRT_LIBRARIES ${TENSORRT_LIBRARY})
  set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
endif()

mark_as_advanced(
  TENSORRT_LIBRARY
  TENSORRT_INCLUDE_DIR
  TENSORRT_VERSION
)

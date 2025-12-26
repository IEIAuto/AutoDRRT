cmake_minimum_required(VERSION 3.8)
project(comm_lib)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)


find_package(CUDA REQUIRED)
enable_language(CUDA)


if(DEFINED CUDACXX)
  set(CMAKE_CUDA_COMPILER ${CUDACXX})
else()
  find_program(CMAKE_CUDA_COMPILER
    NAMES nvcc
    PATHS
      /usr/local/cuda-12.6/bin
      /usr/local/cuda/bin
      /usr/bin
    NO_DEFAULT_PATH
  )
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found. Please set CUDACXX environment variable.")
  endif()
endif()

message(STATUS "Using CUDA compiler: ${CMAKE_CUDA_COMPILER}")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


include_directories(include)

set_source_files_properties(src/cuda_ipc_api.cu PROPERTIES LANGUAGE CUDA)

add_library(cuda_ipc_api
  src/cuda_ipc_api.cu
)


set_property(TARGET cuda_ipc_api PROPERTY CUDA_STANDARD 17)
set_property(TARGET cuda_ipc_api PROPERTY CUDA_STANDARD_REQUIRED ON)
set_property(TARGET cuda_ipc_api PROPERTY CUDA_SEPARABLE_COMPILATION ON)

ament_target_dependencies(cuda_ipc_api
  rclcpp
  std_msgs
)


target_link_libraries(cuda_ipc_api
  ${CUDA_LIBRARIES}
)

target_include_directories(cuda_ipc_api PRIVATE
  ${CUDA_INCLUDE_DIRS}
)

target_include_directories(cuda_ipc_api PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)


install(TARGETS cuda_ipc_api
  EXPORT export_cuda_ipc_api
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)



set_source_files_properties(src/example_api_usage.cu PROPERTIES LANGUAGE CUDA)

add_executable(example_api_usage
  src/example_api_usage.cu
)

set_property(TARGET example_api_usage PROPERTY CUDA_STANDARD 17)
set_property(TARGET example_api_usage PROPERTY CUDA_STANDARD_REQUIRED ON)

target_link_libraries(example_api_usage
  cuda_ipc_api
  ${CUDA_LIBRARIES}
)

ament_target_dependencies(example_api_usage
  rclcpp
  std_msgs
)

install(TARGETS example_api_usage
  DESTINATION lib/${PROJECT_NAME}
)


set_source_files_properties(src/publisher_node_example.cu PROPERTIES LANGUAGE CUDA)

add_executable(publisher_node_example
  src/publisher_node_example.cu
)

set_property(TARGET publisher_node_example PROPERTY CUDA_STANDARD 17)
set_property(TARGET publisher_node_example PROPERTY CUDA_STANDARD_REQUIRED ON)

target_link_libraries(publisher_node_example
  cuda_ipc_api
  ${CUDA_LIBRARIES}
)

ament_target_dependencies(publisher_node_example
  rclcpp
  std_msgs
)

install(TARGETS publisher_node_example
  DESTINATION lib/${PROJECT_NAME}
)


set_source_files_properties(src/subscriber_node_example.cu PROPERTIES LANGUAGE CUDA)

add_executable(subscriber_node_example
  src/subscriber_node_example.cu
)

set_property(TARGET subscriber_node_example PROPERTY CUDA_STANDARD 17)
set_property(TARGET subscriber_node_example PROPERTY CUDA_STANDARD_REQUIRED ON)

target_link_libraries(subscriber_node_example
  cuda_ipc_api
  ${CUDA_LIBRARIES}
)

ament_target_dependencies(subscriber_node_example
  rclcpp
  std_msgs
)

install(TARGETS subscriber_node_example
  DESTINATION lib/${PROJECT_NAME}
)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(cuda_ipc_api)
ament_export_dependencies(rclcpp std_msgs)

ament_package()


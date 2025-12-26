cmake_minimum_required(VERSION 3.8)
project(gpu_utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(CUDA REQUIRED)


include_directories(
  include
  ${CUDA_INCLUDE_DIRS}
)

# This is a header-only utility library.
add_library(gpu_stream_manager INTERFACE)

target_include_directories(gpu_stream_manager INTERFACE
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include>
)


target_link_libraries(gpu_stream_manager INTERFACE
${CUDA_LIBRARIES}
)


install(DIRECTORY include/
DESTINATION include
)


install(TARGETS gpu_stream_manager
EXPORT export_gpu_stream_manager
)


ament_export_targets(export_gpu_stream_manager HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(CUDA)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

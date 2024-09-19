cmake_minimum_required(VERSION 3.5)
project(tensorrt_cmake_module)

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_copyright REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)

  # Find module is intentionally mixed case, so tell linter to ignore that
  ament_lint_cmake("--filters=-convention/filename,-package/stdargs")
  ament_copyright()
  ament_xmllint()
endif()

ament_package(
  CONFIG_EXTRAS "tensorrt_cmake_module-extras.cmake"
)

install(DIRECTORY cmake
  DESTINATION share/${PROJECT_NAME})

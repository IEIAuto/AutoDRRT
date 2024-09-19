cmake_minimum_required(VERSION 3.5)
project(cudnn_cmake_module)

find_package(ament_cmake REQUIRED)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/Modules/FindCUDNN.cmake")
  file(DOWNLOAD
    https://raw.githubusercontent.com/opencv/opencv/0677f3e21cc3379e68d517dc80a8c12e5df0c608/cmake/FindCUDNN.cmake
    ${CMAKE_SOURCE_DIR}/cmake/Modules/FindCUDNN.cmake
    SHOW_PROGRESS
    TIMEOUT 60s
  )
endif()

if(BUILD_TESTING)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_copyright REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)

  # Find module is intentionally mixed case, so tell linter to ignore that
  ament_lint_cmake("--filters=-convention/filename,-package/stdargs")
  ament_copyright("--exclude cmake/Modules/FindCUDNN.cmake")
  ament_xmllint()
endif()

ament_package(
  CONFIG_EXTRAS "cudnn_cmake_module-extras.cmake"
)

install(DIRECTORY cmake
  DESTINATION share/${PROJECT_NAME})

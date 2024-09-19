cmake_minimum_required(VERSION 3.5)
project(llh_converter)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake_auto REQUIRED)

find_package(PkgConfig)
find_path(GeographicLib INCLUDE_DIR GeographicLib/config.h
  PATH_SUFFIXES GeographicLib
)
set(GeographicLib_LIBRARIES
  NAMES Geographic
)

ament_auto_find_build_dependencies()

include_directories(
  include
  ${GeographicLib_INCLUDE_DIRS}
)
link_directories(
  /usr/local/lib
)

ament_auto_add_library(llh_converter SHARED
  src/height_converter.cpp
  src/gsigeo.cpp
  src/llh_converter.cpp
  src/meridian_convergence_angle_correction.cpp
  include/llh_converter/height_converter.hpp
  include/llh_converter/gsigeo.hpp
  include/llh_converter/llh_converter.hpp
  include/llh_converter/meridian_convergence_angle_correction.hpp
)

target_link_libraries(llh_converter PUBLIC ${catkin_LIBRARIES} Geographic)

ament_auto_find_build_dependencies()

install(TARGETS llh_converter
  INCLUDES DESTINATION include
)

ament_target_dependencies(llh_converter)

install(TARGETS llh_converter
  DESTINATION lib/${PROJECT_NAME}
)

target_include_directories(llh_converter
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_export_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

ament_export_libraries(llh_converter)

install(
  DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
)

install(
  DIRECTORY data
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  llh_converter
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )

ament_auto_add_executable(llh_test test/height_converter_test.cpp)
ament_auto_add_executable(mcac_test test/meridian_convergence_angle_correction_test.cpp)

install(TARGETS llh_test
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS mcac_test
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()

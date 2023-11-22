cmake_minimum_required(VERSION 3.5)
project(velodyne_pointcloud)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Ignore PCL errors in Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wno-gnu-anonymous-struct -Wno-nested-anon-types)
endif()

set(CMAKE_BUILD_TYPE "Release")

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()


find_package(Boost COMPONENTS signals)
find_package(PCL REQUIRED COMPONENTS common)

# Resolve system dependency on yaml-cpp, which apparently does not
# provide a CMake find_package() module.
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
find_path(YAML_CPP_INCLUDE_DIR
          NAMES yaml_cpp.h
          PATHS ${YAML_CPP_INCLUDE_DIRS})
find_library(YAML_CPP_LIBRARY
             NAMES YAML_CPP
             PATHS ${YAML_CPP_LIBRARY_DIRS})

find_package(OpenCV REQUIRED)

link_directories(${YAML_CPP_LIBRARY_DIRS})

if(NOT ${YAML_CPP_VERSION} VERSION_LESS "0.5")
  add_definitions(-DHAVE_NEW_YAMLCPP)
endif(NOT ${YAML_CPP_VERSION} VERSION_LESS "0.5")

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_COMMON_INCLUDE_DIRS}
)

# add_subdirectory(src/lib)
ament_auto_add_library(velodyne_rawdata SHARED src/lib/rawdata.cc src/lib/calibration.cc)

ament_auto_add_library(cloud_nodelet SHARED
  src/conversions/convert.cc
  src/conversions/pointcloudXYZIR.cc
  src/conversions/pointcloudXYZIRADT.cc
  src/conversions/output_builder.cc
  src/conversions/func.cc
)
target_link_libraries(cloud_nodelet velodyne_rawdata ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS})

# workaround to allow deprecated header to build on both galactic and rolling
if(${tf2_geometry_msgs_VERSION} VERSION_LESS 0.18.0)
  target_compile_definitions(cloud_nodelet PRIVATE
    USE_TF2_GEOMETRY_MSGS_DEPRECATED_HEADER
  )
endif()

rclcpp_components_register_node(cloud_nodelet
  PLUGIN "velodyne_pointcloud::Convert"
  EXECUTABLE cloud_node
)

ament_auto_add_library(interpolate_nodelet SHARED
  src/conversions/interpolate.cc
  src/conversions/pointcloudXYZIR.cc
  src/conversions/pointcloudXYZIRADT.cc
  src/conversions/func.cc)
target_link_libraries(interpolate_nodelet ${OpenCV_LIBS})

# workaround to allow deprecated header to build on both galactic and rolling
if(${tf2_geometry_msgs_VERSION} VERSION_LESS 0.18.0)
  target_compile_definitions(interpolate_nodelet PRIVATE
    USE_TF2_GEOMETRY_MSGS_DEPRECATED_HEADER
  )
endif()

rclcpp_components_register_node(interpolate_nodelet
  PLUGIN "velodyne_pointcloud::Interpolate"
  EXECUTABLE interpolate_node
)

ament_auto_add_library(transform_nodelet SHARED
  src/conversions/transform.cc
  src/conversions/pointcloudXYZIR.cc
)
target_link_libraries(transform_nodelet ${YAML_CPP_LIBRARIES})

rclcpp_components_register_node(transform_nodelet
  PLUGIN "velodyne_pointcloud::Interpolate"
  EXECUTABLE transform_node
)
# add_subdirectory(src/conversions)

# install(PROGRAMS scripts/gen_calibration.py
#         DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

ament_auto_package(
  INSTALL_TO_SHARE
    launch
    params
)

# if (CATKIN_ENABLE_TESTING)
#   add_subdirectory(tests)
# endif()

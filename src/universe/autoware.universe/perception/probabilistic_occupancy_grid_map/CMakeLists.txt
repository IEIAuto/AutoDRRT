cmake_minimum_required(VERSION 3.14)
project(probabilistic_occupancy_grid_map)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP)
include_directories(
  include
  SYSTEM
  ${EIGEN3_INCLUDE_DIR}
)
# set_target_properties(fast_math PROPERTIES LINKER_LANGUAGE CXX)
# add_library(fast_math )
# add_library(CMSIS STATIC /home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/probabilistic_occupancy_grid_map/lib/libCMSISDSP.a)
# PointcloudBasedOccupancyGridMap
ament_auto_add_library(pointcloud_based_occupancy_grid_map SHARED
  src/pointcloud_based_occupancy_grid_map/pointcloud_based_occupancy_grid_map_node.cpp
  src/pointcloud_based_occupancy_grid_map/threadPool.cpp
  src/updater/occupancy_grid_map_binary_bayes_filter_updater.cpp
)
# target_include_directories(pointcloud_based_occupancy_grid_map PUBLIC 
#   "/home/orin/autoware_awsim_info/src/universe/external/CMSIS_5/CMSIS/Core/Include"
#   )


find_library(OCCUPANCY_GRID_MAP_OPT pointcloud_based_occupancy_grid_map_opt
            PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib/)

target_link_libraries(pointcloud_based_occupancy_grid_map
  ${OCCUPANCY_GRID_MAP_OPT}
)


if(OpenMP_CXX_FOUND)
  message(STATUS,"OPENMP ENABLED")
  target_link_libraries(pointcloud_based_occupancy_grid_map
    ${PCL_LIBRARIES}
    OpenMP::OpenMP_CXX
   
  )
  target_compile_options(pointcloud_based_occupancy_grid_map PUBLIC -fopenmp -march=native -ffast-math -ftree-loop-vectorize -ftree-vectorizer-verbose=2 -fopt-info-vec -O3 -ftree-vectorize)
endif()

rclcpp_components_register_node(pointcloud_based_occupancy_grid_map
  PLUGIN "occupancy_grid_map::PointcloudBasedOccupancyGridMapNode"
  EXECUTABLE pointcloud_based_occupancy_grid_map_node
)

# LaserscanBasedOccupancyGridMap
ament_auto_add_library(laserscan_based_occupancy_grid_map SHARED
  src/laserscan_based_occupancy_grid_map/laserscan_based_occupancy_grid_map_node.cpp
  src/laserscan_based_occupancy_grid_map/occupancy_grid_map.cpp
  src/updater/occupancy_grid_map_binary_bayes_filter_updater.cpp
)

target_link_libraries(laserscan_based_occupancy_grid_map
  ${PCL_LIBRARIES}
)

rclcpp_components_register_node(laserscan_based_occupancy_grid_map
  PLUGIN "occupancy_grid_map::LaserscanBasedOccupancyGridMapNode"
  EXECUTABLE laserscan_based_occupancy_grid_map_node
)

ament_auto_package(
  INSTALL_TO_SHARE
    launch
)

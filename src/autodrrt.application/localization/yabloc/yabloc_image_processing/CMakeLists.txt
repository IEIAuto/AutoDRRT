cmake_minimum_required(VERSION 3.5)
project(yabloc_image_processing)

find_package(autoware_cmake REQUIRED)
autoware_package()

# ===================================================
# Eigen3
find_package(Eigen3 REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# PCL
find_package(PCL REQUIRED COMPONENTS common)
message("opencvLibraries:",${OpenCV_LIBS})
# ===================================================
# Executable
# line segment detector
set(TARGET line_segment_detector_node)
set(OPENCV_XIMGPROC_LIB /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.2)
set(OPENCV_XIMGPROC_INCLUDE_DIRS /usr/include/opencv4/opencv2/imgproc/segmentation.hpp)

ament_auto_add_executable(${TARGET}
  src/line_segment_detector/line_segment_detector_node.cpp
  src/line_segment_detector/line_segment_detector_core.cpp)
target_include_directories(${TARGET} PUBLIC include )
target_include_directories(${TARGET} SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIRS})

target_link_libraries(${TARGET} ${OpenCV_LIBS} ${OPENCV_XIMGPROC_LIB})

# graph based segmentation
set(TARGET graph_segment_node)
ament_auto_add_executable(${TARGET}
  src/graph_segment/graph_segment_node.cpp
  src/graph_segment/graph_segment_core.cpp
  src/graph_segment/similar_area_searcher.cpp)
target_include_directories(${TARGET} PUBLIC include)
target_include_directories(${TARGET} SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIRS} )
target_link_libraries(${TARGET} ${OpenCV_LIBS} ${OPENCV_XIMGPROC_LIB})

# segment filter
set(TARGET segment_filter_node)
ament_auto_add_executable(${TARGET}
  src/segment_filter/segment_filter_node.cpp
  src/segment_filter/segment_filter_core.cpp)
target_include_directories(${TARGET} PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(${TARGET} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OPENCV_XIMGPROC_LIB})

# undistort
set(TARGET undistort_node)
ament_auto_add_executable(${TARGET}
  src/undistort/undistort_node.cpp)
target_link_libraries(${TARGET} ${OpenCV_LIBS} ${OPENCV_XIMGPROC_LIB})

# line_segments_overlay
set(TARGET line_segments_overlay_node)
ament_auto_add_executable(${TARGET}
  src/line_segments_overlay/line_segments_overlay_core.cpp
  src/line_segments_overlay/line_segments_overlay_node.cpp)
target_include_directories(${TARGET} PUBLIC include ${EIGEN_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(${TARGET} ${PCL_LIBRARIES} ${OPENCV_XIMGPROC_LIB})

# lanelet2_overlay
set(TARGET lanelet2_overlay_node)
ament_auto_add_executable(${TARGET}
  src/lanelet2_overlay/lanelet2_overlay_core.cpp
  src/lanelet2_overlay/lanelet2_overlay_node.cpp)
target_include_directories(${TARGET} PUBLIC include ${EIGEN_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(${TARGET} ${PCL_LIBRARIES} ${OPENCV_XIMGPROC_LIB})

# ===================================================
ament_auto_package(INSTALL_TO_SHARE config launch)

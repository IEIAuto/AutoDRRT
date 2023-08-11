#!/bin/bash
file="/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/src/scan_ground_filter_nodelet_ori.cpp"
if [ -f "$file" ];then

    mv "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/src/scan_ground_filter_nodelet.cpp" "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/src/scan_ground_filter_nodelet_heyi.cpp" 
    mv "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/src/scan_ground_filter_nodelet_ori.cpp" "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/src/scan_ground_filter_nodelet.cpp"
    mv "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/include/ground_segmentation/scan_ground_filter_nodelet.hpp" "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/include/ground_segmentation/scan_ground_filter_nodelet_heyi.hpp" 
    mv "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/include/ground_segmentation/scan_ground_filter_nodelet_ori.hpp" "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/include/ground_segmentation/scan_ground_filter_nodelet.hpp"
    mv "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/CMakeLists.txt" "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/CMakeLists_heyi.txt"
    mv "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/CMakeLists_ori.txt" "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/CMakeLists.txt"
    echo "code change into original"
else
    mv "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/src/scan_ground_filter_nodelet.cpp" "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/src/scan_ground_filter_nodelet_ori.cpp" 
    mv "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/src/scan_ground_filter_nodelet_heyi.cpp" "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/src/scan_ground_filter_nodelet.cpp"
    mv "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/include/ground_segmentation/scan_ground_filter_nodelet.hpp" "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/include/ground_segmentation/scan_ground_filter_nodelet_ori.hpp" 
    mv "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/include/ground_segmentation/scan_ground_filter_nodelet_heyi.hpp" "/home/orin/autoware_awsim_info/src/autoware.universe/perception/ground_segmentation/include/ground_segmentation/scan_ground_filter_nodelet.hpp"
    mv "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/CMakeLists.txt" "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/CMakeLists_ori.txt"
    mv "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/CMakeLists_heyi.txt" "/home/orin/autoware_awsim_info/src/universe/autoware.universe/perception/ground_segmentation/CMakeLists.txt"
    echo "code change into heyi"
fi
rm -rf build/ground_segmentation/
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=Off --packages-select=ground_segmentation
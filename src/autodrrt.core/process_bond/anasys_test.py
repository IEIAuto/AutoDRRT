#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import fileinput
plt.ion()
plt.figure(1)

index_list = []
retio = 1
string_list = []

# key_dict = [
#      "velodyne_convert_node",
#       "crop_box_filter_self",
#       "multi_object_tracker",
#       "controller_node_exe",
#      "motion_velocity_smoother",
#      "vehicle_cmd_gate",
#      "obstacle_stop_planner"

# ]
key_dict = [
     "/sensing/lidar/top/crop_box_filter_self",
      "crop_box_filter_self",
      "multi_object_tracker",
      "controller_node_exe",
     "motion_velocity_smoother",
     "vehicle_cmd_gate",
     "obstacle_stop_planner",
     "concatenate_data",
     "obstacle_segmentation/crop_box_filter",
]
node_dict = [
    "crop_box_filter_self",
    "crop_box_filter_mirror",
    "distortion_corrector_node",
    "ring_outlier_filter",
    "crop_box_filter_measurement_range",
    "voxel_grid_downsample_filter",
    "random_downsample_filter",
    "ndt_scan_matcher",
    "ekf_localizer",
    "concatenate_data",
    "obstacle_segmentation/crop_box_filter",
    "common_ground_filter",
    "occupancy_grid_map_node",
    "occupancy_grid_map_outlier_filter",
    "voxel_based_compare_map_filter",
    "voxel_grid_filter",
    "/clustering/outlier_filter",
    "/lidar_centerpoint",
    "euclidean_cluster",
    "shape_estimation",
    "detected_object_feature_remover",
    "object_association_merger ",
    "object_lanelet_filter",
    "multi_object_tracker",
    "map_based_prediction",
    "obstacle_stop_planner",
    "motion_velocity_smoother ",
    "controller_node_exe",
    "vehicle_cmd_gate"
]
def ifRemote(line):
    if line.find("data_received") == -1:#remote
        return False
    else:
        return True
def key_in_arrary(key,arrary):
    for i in arrary:
        if key in i:
            return True
    return False
def line_in_node(line):
    for i in node_dict[16:len(node_dict)]:
        if i in line:
            return True
    return False
def main():
    #Open PIPE to get information
    max_count = 10000
    max_show_length = 400
    temp_index = 0
    time_list = []
    result_total=[]
    result_average=[]
    result_node = {}
    plot_x=[0,0]
    plot_y=[1,1]
    # result_list_str = []
    result_list = []*max_count


    with fileinput.input() as f_input:
        for line in f_input:
            # print(line, end='')
            if line.find("debug_topic_consumer") == -1:
                continue
            else:
                result_list.append(int(line.split(":")[-2].rstrip("\n")))
    print("--平均时间--:{} --最小时间--:{} --最大时间--{} --记录总数--{}".format((sum(result_list)/len(result_list)),min(result_list),max(result_list),len(result_list)))        
                


if __name__=="__main__":
    main()
  
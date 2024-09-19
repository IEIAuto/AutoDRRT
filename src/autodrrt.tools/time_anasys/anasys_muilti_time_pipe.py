#!/usr/bin/env python3
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fileinput
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
target_path = [
  {
    "name": "sensing_lidar_top-sensing_lidar_concatenate",
    "path": [
      "/sensing/lidar/top/velodyne_convert_node",
      "/sensing/lidar/top/crop_box_filter_self",
      "/sensing/lidar/top/crop_box_filter_mirror",
      "/sensing/lidar/top/distortion_corrector_node",
      "/sensing/lidar/top/ring_outlier_filter",
      "/sensing/lidar/concatenate_data"
    ]
  },
  {
    "name": "sensing_lidar_top-localization_ekf_localizer",
    "path": [
      "/sensing/lidar/top/velodyne_convert_node",
      "/sensing/lidar/top/crop_box_filter_self",
      "/sensing/lidar/top/crop_box_filter_mirror",
      "/sensing/lidar/top/distortion_corrector_node",
      "/sensing/lidar/top/ring_outlier_filter",
      "/localization/util/crop_box_filter_measurement_range",
      "/localization/util/voxel_grid_downsample_filter",
      "/localization/util/random_downsample_filter",
      "/localization/pose_estimator/ndt_scan_matcher",
      "/localization/pose_twist_fusion_filter/ekf_localizer"
    ]
  },
  {
    "name": "e2e-lidar_top-obstacle_stop_planner-vehicle_cmd_gate",
    "path": [
    #   "/sensing/lidar/top/velodyne_convert_node",
    #   "/sensing/lidar/top/crop_box_filter_self",
    #   "/sensing/lidar/top/crop_box_filter_mirror",
    #   "/sensing/lidar/top/distortion_corrector_node",
    #   "/sensing/lidar/top/ring_outlier_filter",
    #   "/sensing/lidar/concatenate_data",
    #   "/perception/obstacle_segmentation/crop_box_filter",
    #   "/perception/obstacle_segmentation/common_ground_filter",
    #   "/perception/occupancy_grid_map/occupancy_grid_map_node",
    #   "/perception/obstacle_segmentation/occupancy_grid_map_outlier_filter",
      "/planning/scenario_planning/lane_driving/motion_planning/obstacle_stop_planner",
      "/planning/scenario_planning/scenario_selector",
      "/planning/scenario_planning/motion_velocity_smoother",
      "/control/trajectory_follower/controller_node_exe",
      "/control/vehicle_cmd_gate"
    ]
  },
  {
    "name": "e2e-lidar_top-map_based_prediction-planning-vehicle_cmd_gate",
    "path": [
      "/sensing/lidar/top/velodyne_convert_node",
      "/sensing/lidar/top/crop_box_filter_self",
      "/sensing/lidar/top/crop_box_filter_mirror",
      "/sensing/lidar/top/distortion_corrector_node",
      "/sensing/lidar/top/ring_outlier_filter",
      "/sensing/lidar/concatenate_data",
      "/perception/obstacle_segmentation/crop_box_filter",
      "/perception/obstacle_segmentation/common_ground_filter",
      "/perception/occupancy_grid_map/occupancy_grid_map_node",
      "/perception/obstacle_segmentation/occupancy_grid_map_outlier_filter",
      "/perception/object_recognition/detection/voxel_based_compare_map_filter",
      "/perception/object_recognition/detection/clustering/voxel_grid_filter.*",
      "/perception/object_recognition/detection/clustering/outlier_filter",
      "/perception/object_recognition/detection/clustering/euclidean_cluster",
      "/perception/object_recognition/detection/clustering/shape_estimation",
      "/perception/object_recognition/detection/clustering/detected_object_feature_remover",
      "/perception/object_recognition/detection/object_association_merger.*",
      "/perception/object_recognition/detection/object_association_merger.*",
      "/perception/object_recognition/detection/object_lanelet_filter",
      "/perception/object_recognition/tracking/multi_object_tracker",
      "/perception/object_recognition/prediction/map_based_prediction",
      "/planning/scenario_planning/lane_driving/behavior_planning/behavior_path_planner",
      "/planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner",
      "/planning/scenario_planning/lane_driving/motion_planning/obstacle_avoidance_planner",
      "/planning/scenario_planning/lane_driving/motion_planning/obstacle_velocity_limiter",
      "/planning/scenario_planning/lane_driving/motion_planning/obstacle_stop_planner",
      "/planning/scenario_planning/scenario_selector",
      "/planning/scenario_planning/motion_velocity_smoother",
      "/control/trajectory_follower/controller_node_exe",
      "/control/vehicle_cmd_gate"
    ]
  }
]
node_set = set()
key_dict = [
     "/sensing/lidar/top/velodyne_convert_node",
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
    
    for path in target_path:
        for node in path["path"]:
            node_set.add(node)
    # print(node_set)
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
    result_list = [[]]*max_count
    result_node_total_dt = [[]]
    result_node_total_static = [[]]*len(node_dict)
    result_node_total_count = [0]*len(node_dict)
    result_node_static_dt = [0.0]*len(node_dict)



    node_set_list = list(node_set)
    all_node_delay = [[]]*len(node_set_list)


    target_delay_total = [copy.deepcopy([]) for _ in range(len(target_path))]
    target_delay_average = [copy.deepcopy([]) for _ in range(len(target_path))]

    
    target_delay_node_detail = []
    target_delay_node_detail_dt = []
    result_node_total_dt = [[]]
    node_value_count = []
    showed_flags = [False] * len(target_path)

    for i in target_path:
        target_delay_node_detail.append([copy.deepcopy([]) for _ in range(len(i["path"]))])
        target_delay_node_detail_dt.append([copy.deepcopy([]) for _ in range(len(i["path"]))])
        node_value_count.append([0] * len(i["path"]))
    fig, axs = plt.subplots(len(target_path), figsize=(10, 5*len(target_path)),gridspec_kw={'hspace':0.5})
    # print(target_delay_node_detail_dt[0][0],len(target_delay_node_detail_dt))
    # target_delay_node_detail_dt[0][0].append(0)
    # target_delay_node_detail_dt[0][1].append(1)
    # target_delay_node_detail_dt[0][2].append(2)
    # print(target_delay_node_detail_dt,len(target_delay_node_detail_dt))
    # exit(0)
    # for i in range(0,len(result_node_total_static)):
    #     result_node_total_static[i]=[]#init
    with fileinput.input() as f_input:
        for line in f_input:
            # print(line, end='')
            if line.find("debug_topic_consumer") == -1:
                continue
            else:
                #find the radio header
                if (key_dict[0] in line):
                    #put the id in set
                    time_id_set = set()
                    time_id_set.add(line.split(":")[-1].rstrip("\n"))
                    # print(temp_index)
                    # result_list_str.append(line)
                    result_list[temp_index]=[]
                    result_list[temp_index].append(line)
                    time_list.append(time_id_set)
                    temp_index = temp_index + 1
                    showed_flags = [False]*len(target_path)
                    
                else:#need find where the id is
                    for index in range(0,len(time_list)):
                        for time_id in time_list[index].copy():#针对id进行查找，并把结果放入结果列表
                            if time_id in line:
                                # print(str(len(time_list[index])),str(line_in_node(line)))
                                if((len(time_list[index])>1 and line_in_node(line)) or len(time_list[index])==1):
                                    # print(len(time_list[index]))
                                    result_list[index].append(line)
                                    # print(index,time_list[index],line)
                                    if key_dict[2] in line  or key_dict[6] in line or key_dict[3] in line:#遇到转发节点，则将转发节点的时间戳加入id列表
                                        tmp_str_head = line.split(":")[-1].rstrip("\n")
                                        if(len(tmp_str_head)>2):
                                            time_list[index].add(tmp_str_head)
                                            # print(str(time_list[index]))

                                    if key_dict[8] in line:#key node
                                        if not (len(result_list[index]) >2 and key_dict[7] in  result_list[index][-2]):##key path
                                            time_list[index] = set()
                                            result_list[index] = []
                                            break
                                    if key_dict[5] in line:#last node, all node are here
                                        #step 1. get all node's delay
                                        # for text in result_list[index]:
                                        #     print(index,text)


                                        # for node_idex in range(0,len(node_set_list)):
                                        #     for result_idx in range(0,len(result_list[index])):
                                        #         if result_list[index][result_idx].find(node_set_list[node_idex]) != -1:#search nodes in the result
                                        #             # print(result_list[index][result_idx])
                                        #             if ifRemote(result_list[index][result_idx]):
                                        #                 all_node_delay[node_idex].append(int(result_list[index][result_idx].split(":")[5].rstrip("\n")))
                                        #             else:
                                        #                 all_node_delay[node_idex].append((result_list[index][result_idx].split(":")[3].rstrip("\n")))
                                        #             break


                                        # for node_idex in range(0,len(node_set_list)):
                                        #     print(all_node_delay[node_idex])
                                        #step 2. get all target delay
                                        for target_path_index in range(0,len(target_path)):
                                            # if target_path_index != 1:
                                            #     continue
                                            # print(showed_flags)
                                            nozero_index = []
                                            path_line = []
                                            # if showed_flags[target_path_index]:
                                            #     continue
                                            result_node_total_dt = [0]*len(target_path[target_path_index]["path"])
                                            for node_name_index in range(0,len(target_path[target_path_index]["path"])):
                                                    # print(node_name_index,len(path_line),len(result_list[index]))
                                                    insert_flag = 0
                                                    for result_idx in range(0,len(result_list[index])):
                                                        if result_list[index][result_idx].find(target_path[target_path_index]["path"][node_name_index]) != -1:#search nodes in the result
                                                            insert_flag = 1
                                                            nozero_index.append(node_name_index)
                                                            path_line.append(result_list[index][result_idx].rstrip("\n"))
                                                            node_value_count[target_path_index][node_name_index] = node_value_count[target_path_index][node_name_index] + 1
                                                            # print(result_list[index][result_idx])
                                                            if ifRemote(result_list[index][result_idx]):
                                                                target_delay_node_detail[target_path_index][node_name_index].append(int(result_list[index][result_idx].split(":")[4].rstrip("\n"))) 
                                                                target_delay_node_detail_dt[target_path_index][node_name_index].append(int(result_list[index][result_idx].split(":")[5].rstrip("\n")))
                                                                
                                                            else:
                                                                target_delay_node_detail[target_path_index][node_name_index].append(int(result_list[index][result_idx].split(":")[2].rstrip("\n")))
                                                                target_delay_node_detail_dt[target_path_index][node_name_index].append(int(result_list[index][result_idx].split(":")[3].rstrip("\n")))
                                                            break
                                                            
                                                    #  node did not in result
                                                    if insert_flag == 0:
                                                        target_delay_node_detail[target_path_index][node_name_index].append(0)
                                                        target_delay_node_detail_dt[target_path_index][node_name_index].append(0)
                                                        
                                            # print("--",path_line[0])
                                            # print("--",path_line[-1])
                                            # print("==",target_path[target_path_index]["path"][0])
                                            # print("==",target_path[target_path_index]["path"][-1])
                                            start_time = [0]*len(target_path)
                                            if len(path_line)>0 and path_line[0].find(target_path[target_path_index]["path"][0]) != -1 and path_line[-1].find(target_path[target_path_index]["path"][-1]) != -1:
                                                showed_flags[target_path_index] = True
                                                start = 0
                                                end = 0
                                                if ifRemote(path_line[0]):
                                                    start = int(path_line[0].split(":")[4].rstrip("\n"))
                                                else:
                                                    start = int(path_line[0].split(":")[2].rstrip("\n"))
                                                if ifRemote(path_line[-1]):
                                                    end = int(path_line[-1].split(":")[4].rstrip("\n"))
                                                else:
                                                    end = int(path_line[-1].split(":")[2].rstrip("\n"))
                                                time_ms = end - start
                                                start_time[target_path_index] = start
                                                if time_ms < 0 or len(result_list[index]) < 7 or time_ms > 1000:#ingore bad data
                                                    result_list[index] = []
                                                    break
                                                target_delay_total[target_path_index].append(time_ms)
                                                target_delay_average[target_path_index].append(sum(target_delay_total[target_path_index])/len(target_delay_total[target_path_index]))
                                            else:
                                                continue
                                            
                                            #static results
                                            for node_name_index in range(0,len(target_path[target_path_index]["path"])):
                                                total_time = target_delay_node_detail[target_path_index][node_name_index][-1]-start_time[target_path_index]
                                                if total_time < 0:
                                                    total_time = 0

                                                if(node_value_count[target_path_index][node_name_index] != 0):
                                                    print(target_path[target_path_index]["path"][node_name_index].ljust(80),": average time:",format(round(sum(target_delay_node_detail_dt[target_path_index][node_name_index])/node_value_count[target_path_index][node_name_index],4)).ljust(10),"ms current:",str(target_delay_node_detail_dt[target_path_index][node_name_index][-1]).ljust(4),"ms","total:",str(total_time).ljust(4),"ms"," count:",node_value_count[target_path_index][node_name_index])
                                                else:
                                                    print(target_path[target_path_index]["path"][node_name_index].ljust(80),": average time:",format(round(target_delay_node_detail_dt[target_path_index][node_name_index][-1],4)).ljust(10),"ms current:",str(target_delay_node_detail_dt[target_path_index][node_name_index][-1]).ljust(4),"ms","total:",str(total_time).ljust(4),"ms"," count:",node_value_count[target_path_index][node_name_index] )
                                                    # print()
                                            print(target_path_index,"------",target_path[target_path_index]["name"].ljust(60),": average time:",format(round(sum(target_delay_total[target_path_index])/len(target_delay_total[target_path_index]),4)).ljust(6),"ms current:",str(target_delay_total[target_path_index][-1]).ljust(4),"ms"," max: ",format(max(target_delay_total[target_path_index])).ljust(4),"ms"," min: ",format(min(target_delay_total[target_path_index])).ljust(4),"ms"," count:",len(target_delay_total[target_path_index]),"------")
                                            ax = axs[target_path_index]
                                            ax.clear()
                                            l_end = len(target_delay_total[target_path_index])
                                            if l_end - max_show_length > 0:
                                                l_start = l_end - max_show_length
                                            else:
                                                l_start = 0
                                            x_real=range(l_start,l_end)
                                            line_real, = ax.plot(x_real,target_delay_total[target_path_index][l_start:l_end],alpha=0.7,label= 'Real time')
                                            ax.annotate(f'{target_delay_total[target_path_index][-1]:.2f}',xy=(x_real[-1],target_delay_total[target_path_index][-1]),xytext=(x_real[-1]+0.2,target_delay_total[target_path_index][-1]+0.2),arrowprops=dict(facecolor=line_real.get_color(),shrink=0.05))

                                            x_average = range(l_start,l_end)
                                            line_average, = ax.plot(x_average,target_delay_average[target_path_index][l_start:l_end],alpha=0.7,label='Average time')
                                            ax.annotate(f'{target_delay_average[target_path_index][-1]:.2f}',xy=(x_average[-1],target_delay_average[target_path_index][-1]),xytext=(x_average[-1]+0.2,target_delay_average[target_path_index][-1]+0.2),arrowprops=dict(facecolor=line_average.get_color(),shrink=0.05))
                                            ax.set_xlabel('Iterations')
                                            ax.set_ylabel('time(ms)')
                                            ax.set_title(target_path[target_path_index]["name"])
                                            ax.legend()
                                            plt.pause(0.01)
                                            
                                        result_list[index] = [] 
                                        time_list[index]=set()
                                        break
                                    




if __name__=="__main__":
    main()
  
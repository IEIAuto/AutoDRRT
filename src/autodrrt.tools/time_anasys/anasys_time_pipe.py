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
    result_list = [[]]*max_count
    result_node_total_dt = [[]]
    result_node_total_static = [[]]*len(node_dict)
    result_node_total_count = [0]*len(node_dict)
    result_node_static_dt = [0.0]*len(node_dict)
    for i in range(0,len(result_node_total_static)):
        result_node_total_static[i]=[]#init
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
                                    if key_dict[5] in line:#last node
                                        # if(not key_in_arrary(key_dict[8],result_list[index])):
                                        #     break
                                        
                                        # for text in result_list[index]:
                                        #     print(index,text)
                                        if line.find("data_received") == -1:#remote
                                            if ifRemote(result_list[index][0]):
                                                start = int(result_list[index][0].split(":")[4].rstrip("\n"))
                                            else:
                                                start = int(result_list[index][0].split(":")[2].rstrip("\n"))
                                            end = int(line.split(":")[2])
                                        else:
                                            if ifRemote(result_list[index][0]):
                                                start = int(result_list[index][0].split(":")[4].rstrip("\n"))
                                            else:
                                                start = int(result_list[index][0].split(":")[2].rstrip("\n"))
                                            end = int(line.split(":")[4])
                                        time_ms = end - start
                                        if time_ms < 0 or len(result_list[index]) < 7 or time_ms > 1000:#ingore bad data
                                            result_list[index] = []
                                            break

                                
                    
                                        # print(index,result_list[index][0],total,"ms")#node all publised , clean it
                                        result_total.append(time_ms)
                                        average_time = sum(result_total)/len(result_total)
                                        result_average.append(average_time)
                                        #start to static every node's situation
                                        # result_node_list[temp_index]=[]
                                        result_node_list = [0]*len(node_dict)
                                        result_node_total_dt = [0]*len(node_dict)
                                        result_node_list_dt = [0]*len(node_dict)

                                        #search nodedict in order
                                        nozero_index = []
                                        for node_idex in range(0,len(node_dict)):
                                            insert_flag = 0
                                            for result_idx in range(0,len(result_list[index])):
                                                if result_list[index][result_idx].find(node_dict[node_idex]) != -1:#search nodes in the result
                                                    insert_flag = 1
                                                    nozero_index.append(node_idex)
                                                    # print(result_list[index][result_idx])
                                                    if ifRemote(result_list[index][result_idx]):
                                                        result_node_list[node_idex] = int(result_list[index][result_idx].split(":")[4].rstrip("\n"))
                                                        result_node_list_dt[node_idex] = int(result_list[index][result_idx].split(":")[5].rstrip("\n"))
                                                        
                                                    else:
                                                        result_node_list[node_idex] = int(result_list[index][result_idx].split(":")[2].rstrip("\n"))
                                                        result_node_list_dt[node_idex] = int(result_list[index][result_idx].split(":")[3].rstrip("\n"))
                                                    break
                                            #  node did not in result
                                            if insert_flag == 0:
                                                result_node_list[node_idex] = 0
                                        # print(nozero_index)
                                        for no_result_idx in range(0,len(nozero_index)):
                                            result_node_total_count[nozero_index[no_result_idx]] = result_node_total_count[nozero_index[no_result_idx]] + 1
                                            if nozero_index[no_result_idx] == 0:
                                                continue
                                            elif no_result_idx == 0:
                                                continue
                                                
                                            else:# calc the dt
                                                #cal total
                                                result_node_total_dt[nozero_index[no_result_idx]] = result_node_list[nozero_index[no_result_idx]] -  result_node_list[nozero_index[0]]
                                                # print(no_result_idx,nozero_index[no_result_idx],nozero_index[no_result_idx-1])

                                                # result_node_list_dt[nozero_index[no_result_idx]] = abs(result_node_list[nozero_index[no_result_idx]] -  result_node_list[nozero_index[no_result_idx-1]])

                                                #average cost for every node
                            
                                                temp_int = (result_node_static_dt[nozero_index[no_result_idx]] + result_node_list_dt[nozero_index[no_result_idx]] )
                                            
                                                result_node_static_dt[nozero_index[no_result_idx]] = temp_int
                                        
                                        
                                        for i in range(0,len(node_dict)):
                                            if(result_node_total_count[i] != 0):
                                                print(node_dict[i].ljust(40),": average time:",format(round(result_node_static_dt[i]/result_node_total_count[i],4)).ljust(10),"ms current:",str(result_node_list_dt[i]).ljust(4),"ms"," count:",result_node_total_count[i]," total",(result_node_total_dt[i]),"ms")
                                            else:
                                                print(node_dict[i].ljust(40),": average time:",format(round(result_node_static_dt[i],4)).ljust(10),"ms current:",str(result_node_list_dt[i]).ljust(4),"ms"," count:",result_node_total_count[i]," total",(result_node_total_dt[i]),"ms")
                                            
                                                
                                        print("======total average time:",average_time,"ms ","current:",time_ms, "ms max_delay:",max(result_total), "ms min_delay:",min(result_total),"ms count:",len(result_total),"======")
                                        
                                        plt.cla()
                                        l_end = len(result_total)
                                        if l_end - max_show_length > 0:
                                            l_start = l_end - max_show_length
                                        else:
                                            l_start = 0
                                        x_real=range(l_start,l_end)
                                        line_real, = plt.plot(x_real,result_total[l_start:l_end],alpha=0.7,label='Real time')
                                        plt.annotate(f'{result_total[-1]:.2f}',xy=(x_real[-1],result_total[-1]),xytext=(x_real[-1]+0.2,result_total[-1]+0.2),arrowprops=dict(facecolor=line_real.get_color(),shrink=0.05))
                                        
                                        x_average = range(l_start,l_end)


                                        line_average, = plt.plot(x_average,result_average[l_start:l_end],alpha=0.7,label='Average time')
                                        plt.annotate(f'{result_average[-1]:.2f}',xy=(x_average[-1],result_average[-1]),xytext=(x_average[-1]+0.2,result_average[-1]+0.2),arrowprops=dict(facecolor=line_average.get_color(),shrink=0.05))
                                        plt.xlabel('Iterations')
                                        plt.ylabel('time(ms)')
                                        plt.title('Path delay')
                                        plt.legend()
                                        plt.pause(0.01)
                                        time_list[index]=set()
                                        result_list[index]=[]
                                        # print(index,result_list[index])
                                        
                                        break
            # plt.pause(0)
            if temp_index >= max_count:
                temp_index = 0
                time_list = []
                result_total=[]
                result_average=[]
                result_list = [[]]*max_count

                


if __name__=="__main__":
    main()
  
  docker run -it --net=host --gpus all -e DISPLAY=$DISPLAY -v /data/liuhonggang:/data/liuhonggang -v /tmp/.X11-unix/:/tmp/.X11-unix -w /data/liuhonggang -v /dev/shm:/dev/shm --name=lhg_bevdet 56cfe1e58943
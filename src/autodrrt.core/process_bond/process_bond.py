import os
import subprocess
import time

# 指定要监控的进程名称列表和相应的配置项值
processes = [
    {
        'name': 'component_container_mt_pointcloud_container',
        'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container_mt --ros-args -r __node:=pointcloud_container',
        'cpus_value': '6',
        'mems_value': '0',
        'cpu_exclusive_value': '0',
        'priority': 1  # 进程优先级
    },
    #  {
    #     'name': 'component_container_mt_pointcloud_container_velodyne_node_container',
    #     'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container_mt --ros-args -r __node:=velodyne_node_container -r __ns:=/sensing/lidar/top/pointcloud_preprocessor',
    #     'cpus_value': '0',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 5  # 进程优先级
    # },
    {
        'name': 'component_container_mt_pointcloud_container_velodyne_node_container',
        'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container_mt --ros-args -r __node:=velodyne_node_container -r __ns:=/sensing/lidar/top/pointcloud_preprocessor',
        'cpus_value': '6',
        'mems_value': '0',
        'cpu_exclusive_value': '0',
        'priority': 5  # 进程优先级
    },
    {
        'name': 'component_container_mt_euclidean_cluster_container',
        'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container --ros-args -r __node:=euclidean_cluster_container',
        'cpus_value': '9',
        'mems_value': '0',
        'cpu_exclusive_value': '0',
        'priority': 10  # 进程优先级
    },
     {
        'name': 'shape_estimation',
        'cmd': '/home/orin/autoware/install/shape_estimation/lib/shape_estimation/shape_estimation',
        'cpus_value': '9',
        'mems_value': '0',
        'cpu_exclusive_value': '0',
        'priority': 10  # 进程优先级
    },
     {
        'name': 'detected_object_feature_remover',
        'cmd': '/home/orin/autoware/install/detected_object_feature_remover/lib/detected_object_feature_remover/detected_object_feature_remover',
        'cpus_value': '9',
        'mems_value': '0',
        'cpu_exclusive_value': '0',
        'priority': 10  # 进程优先级
    },
    # {
    #     'name': 'detected_object_feature_remover',
    #     'cmd': '/home/orin/autoware/install/detected_object_feature_remover/lib/detected_object_feature_remover/detected_object_feature_remover',
    #     'cpus_value': '2',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    

    {
        'name': 'lidar_centerpoint_node',
        'cmd': '/home/orin/autoware/install/lidar_centerpoint/lib/lidar_centerpoint/lidar_centerpoint_node',
        'cpus_value': '9',
        'mems_value': '0',
        'cpu_exclusive_value': '0',
        'priority': 10  # 进程优先级
    },
    {
        'name': 'component_container_mrm_comfortable_stop_operator_container',
        'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container --ros-args -r __node:=mrm_comfortable_stop_operator_container',
        'cpus_value': '9',
        'mems_value': '0',
        'cpu_exclusive_value': '0',
        'priority': 2  # 进程优先级
    },
    {
        'name': 'component_container_mrm_emergency_stop_operator_container',
        'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container --ros-args -r __node:=mrm_emergency_stop_operator_container',
        'cpus_value': '10',
        'mems_value': '0',
        'cpu_exclusive_value': '0',
        'priority': 3  # 进程优先级
    },
    {
        'name': 'component_container_map_container',
        'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container --ros-args -r __node:=map_container',
        'cpus_value': '10',
        'mems_value': '0',
        'cpu_exclusive_value': '0',
        'priority': 4  # 进程优先级
    },

    {
        'name': 'ndt_scan_matcher',
        'cmd': '/home/orin/autoware/install/ndt_scan_matcher/lib/ndt_scan_matcher/ndt_scan_matcher',
        'cpus_value': '10',
        'mems_value': '0',
        'cpu_exclusive_value': '0',
        'priority': 10  # 进程优先级
    },
    {
        'name': 'ekf_localizer',
        'cmd': '/home/orin/autoware/install/ekf_localizer/lib/ekf_localizer/ekf_localizer',
        'cpus_value': '10',
        'mems_value': '0',
        'cpu_exclusive_value': '0',
        'priority': 10  # 进程优先级
    },

    
    # {
    #     'name': 'component_container_traffic_light_node_container',
    #     'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container_mt --ros-args -r __node:=traffic_light_node_container',
    #     'cpus_value': '3',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    # {
    #     'name': 'component_container_behavior_planning_container',
    #     'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container_mt --ros-args -r __node:=behavior_planning_container',
    #     'cpus_value': '4',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    # {
    #     'name': 'component_container_mt_motion_planning_container',
    #     'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container_mt --ros-args -r __node:=motion_planning_container',
    #     'cpus_value': '4',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    # {
    #     'name': 'detection_by_tracker',
    #     'cmd': '/home/orin/autoware/install/detection_by_tracker/lib/detection_by_tracker/detection_by_tracker',
    #     'cpus_value': '5',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    #  {
    #     'name': 'object_association_merger_node',
    #     'cmd': '/home/orin/autoware/install/object_merger/lib/object_merger/object_association_merger_node',
    #     'cpus_value': '5',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    #  {
    #     'name': 'object_lanelet_filter_node',
    #     'cmd': '/home/orin/autoware/install/detected_object_validation/lib/detected_object_validation/object_lanelet_filter_node',
    #     'cpus_value': '5',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    #  {
    #     'name': 'multi_object_tracker',
    #     'cmd': '/home/orin/autoware/install/multi_object_tracker/lib/multi_object_tracker/multi_object_tracker',
    #     'cpus_value': '5',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    #  {
    #     'name': 'map_based_prediction',
    #     'cmd': '/home/orin/autoware/install/map_based_prediction/lib/map_based_prediction/map_based_prediction',
    #     'cpus_value': '5',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
       

    # {
    #     'name': 'component_container_mt_parking_container',
    #     'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container --ros-args -r __node:=parking_container',
    #     'cpus_value': '6',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    # {
    #     'name': 'component_container_control_container',
    #     'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container --ros-args -r __node:=control_container',
    #     'cpus_value': '6',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    # {
    #     'name': 'component_container_default_ad_api',
    #     'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container_mt --ros-args -r __node:=container -r __ns:=/default_ad_api',
    #     'cpus_value': '7',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    # {
    #     'name': 'component_container_awapi_relay_container',
    #     'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container --ros-args -r __node:=awapi_relay_container -r __ns:=/awapi',
    #     'cpus_value': '7',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    
    # {
    #     'name': 'component_container_mt_pointcloud_autoware_iv_adaptor',
    #     'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container_mt --ros-args -r __node:=autoware_iv_adaptor -r __ns:=/autoware_api/external',
    #     'cpus_value': '8',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
    # {
    #     'name': 'component_container_autoware_iv_adaptor',
    #     'cmd': '/home/orin/ros2/install/rclcpp_components/lib/rclcpp_components/component_container_mt --ros-args -r __node:=autoware_iv_adaptor -r __ns:=/autoware_api/internal',
    #     'cpus_value': '8',
    #     'mems_value': '0',
    #     'cpu_exclusive_value': '0',
    #     'priority': 10  # 进程优先级
    # },
   
    {
        'name': 'rviz',
        'cmd': '/home/orin/autoware/install/autoware_launch/share/autoware_launch/rviz/autoware.rviz',
        'cpus_value': '11',
        'mems_value': '0',
        'cpu_exclusive_value': '0',
        'priority': 10  # 进程优先级
    }
    # 添加更多进程...
]
cpu_cores = [6,7,8,9,10,11]
# 指定用于创建子目录的目录路径
base_directory = "/sys/fs/cgroup/cpuset/autoware_test"

# 记录已处理的进程 PID 和对应的 cpuset 子目录路径
processed_pids = {}

def set_process_priority(pid, priority):
    # 设置进程优先级
    os.sched_setscheduler(pid, os.SCHED_FIFO, os.sched_param(priority))

    # 输出日志
    print("PID {} 设置优先级为 {}".format(pid, priority))

def delete_cgroup(directory_name):
    # 检查 tasks 文件是否为空
    with open(os.path.join(directory_name, "tasks"), "r") as tasks_file:
        tasks = tasks_file.read().strip()

    if tasks:
        print("子目录 {} 中仍有进程正在使用，不能删除".format(directory_name))
        return

    # 删除子目录
    subprocess.getoutput("rmdir {}".format(directory_name))
    # shutil.rmtree(directory_name)

    # 输出日志
    print("子目录 {} 已删除".format(directory_name))
    

# 修改 create_cgroup 函数，添加设置进程优先级的步骤
def create_cgroup(pid, process_name, cpus_value, mems_value, cpu_exclusive_value, priority):
    # 创建子目录，使用进程名字+PID的形式作为目录名称
    directory_name = os.path.join(base_directory, "cpu_{}/{}_{}".format(cpus_value,process_name, pid))
    os.makedirs(directory_name)
    print(directory_name)
    # 设置 cpuset.cpus
    with open(os.path.join(directory_name, "cpuset.cpus"), "w") as cpus_file:
        cpus_file.write(cpus_value)

    # 设置 cpuset.mems
    with open(os.path.join(directory_name, "cpuset.mems"), "w") as mems_file:
        mems_file.write(mems_value)

    # 设置 cpuset.cpu_exclusive
    with open(os.path.join(directory_name, "cpuset.cpu_exclusive"), "w") as cpu_exclusive_file:
        cpu_exclusive_file.write(cpu_exclusive_value)

    # 写入 cpuset 的 tasks
    with open(os.path.join(directory_name, "tasks"), "w") as tasks_file:
        tasks_file.write(pid)

    # 写入 cpuset 的 procs
    with open(os.path.join(directory_name, "cgroup.procs"), "w") as procs_file:
        procs_file.write(pid)
    print(subprocess.getoutput("renice -20 '{}'".format(pid)))
    
    # 设置进程优先级
    # set_process_priority(int(pid), priority)

    # 输出日志
    print("PID {} 创建了子目录 {}, 已经配置完成".format(pid, directory_name))
def init_directorys(cpus_value, mems_value, cpu_exclusive_value):
    directory_name = os.path.join(base_directory, "cpu_{}".format(cpus_value))
    os.makedirs(directory_name)
    # 设置 cpuset.cpus
    with open(os.path.join(directory_name, "cpuset.cpus"), "w") as cpus_file:
        cpus_file.write(cpus_value)

    # 设置 cpuset.mems
    with open(os.path.join(directory_name, "cpuset.mems"), "w") as mems_file:
        mems_file.write(mems_value)

    # 设置 cpuset.cpu_exclusive
    with open(os.path.join(directory_name, "cpuset.cpu_exclusive"), "w") as cpu_exclusive_file:
        cpu_exclusive_file.write(cpu_exclusive_value)


for cpu in  cpu_cores:
    try:
        init_directorys(str(cpu),str(0),str(1))
    except FileExistsError:
        continue
# 在主循环中，调用 set_process_priority 函数来更新进程优先级
while True:
    current_pids = []
    for process in processes:
        process_name = process['name']
        process_cmd = process['cmd']
        cpus_value = process['cpus_value']
        mems_value = process['mems_value']
        cpu_exclusive_value = process['cpu_exclusive_value']
        priority = process['priority']

        # 查找进程的 PID
        pids = subprocess.getoutput("pgrep -f '{}'".format(process_cmd))
        # pids = subprocess.getoutput("pidof {}".format(process_cmd))
        # pids = subprocess.check_output(['pgrep','-f',process_cmd])
        
        pids = pids.split()  # 处理多个进程的情况
        del pids[-1]
        print(pids)
        # continue
        
        for index,pid in enumerate(pids):
            current_pids.append(pid)
            # print(processed_pids)
            if pid not in processed_pids:
                create_cgroup(pid, process_name, cpus_value, mems_value, cpu_exclusive_value, priority)
                # create_cgroup(pid, process_name, str(index), mems_value, cpu_exclusive_value, priority)
                processed_pids[pid] = os.path.join(base_directory, "cpu_{}/{}_{}".format(cpus_value,process_name, pid))
            elif pid in processed_pids and processed_pids[pid] != os.path.join(base_directory, "cpu_{}/{}_{}".format(cpus_value,process_name, pid)):
                old_directory_name = processed_pids[pid]
                delete_cgroup(old_directory_name)
                create_cgroup(pid, process_name, cpus_value, mems_value, cpu_exclusive_value, priority)
                # create_cgroup(pid, process_name, str(index), mems_value, cpu_exclusive_value, priority)
                processed_pids[pid] = os.path.join(base_directory, "cpu_{}/{}_{}".format(cpus_value,process_name, pid))
    processed_pids_tmp = processed_pids.copy()
    for pid, directory_name in processed_pids_tmp.items():
        # print(pid,directory_name,pids)
        if pid not in current_pids:
            delete_cgroup(directory_name)
            del processed_pids[pid]

    time.sleep(1)

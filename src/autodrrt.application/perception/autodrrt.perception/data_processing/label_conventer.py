import json
import numpy as np
import cv2
import os
from vis_pb_lidar_tools import visualize_camera
from scipy.spatial.transform import Rotation as R


intrinsics_array = np.array([
                540,
                0.0,
                540,
                0.0,
                540,
                360,
                0.0,
                0.0,
                1.0
            ]).reshape((3,3))
cam2img = np.eye(4)
cam2img[:3, :3] = intrinsics_array
cam2ego_RT = np.eye(4)

# 图像的坐标转为雷达或者ego的坐标
# 先对矩阵取逆，然后输入位移值
# 最后用个点验证一下


cam2ego_dir = {}

cam2ego_dir['front'] = np.array([
    [0, -1,  0],
    [0,  0, -1],
    [1,  0,  0]
])

cam2ego_dir['left'] = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0]
])

cam2ego_dir['right'] = np.array([
    [-1, 0,  0],
    [0,  0, -1],
    [0, -1,  0]
])

cam2ego_dir['back'] = np.array([
    [0,  1,  0],
    [0,  0, -1],
    [-1, 0,  0]
])


lidar2ego = np.array([
    [1,  0,  0, 1.0],
    [0,  1,  0, 0],
    [0,  0,  1, 1.8],
    [0,  0,  0, 1],
])

##  直接从下面拿来取逆
front2cam = np.array([
    [0,  0,  1, 0],
    [-1,  0, 0, 0],
    [0,  -1,  0, 0],
    [0,  0,  0, 1]
])

##  直接输入图像中的生成
f_cam2ego = np.array([
    [1,  0,  0, 3.9],
    [0,  1,  0, 0],
    [0,  0,  1, 0.9],
    [0,  0,  0, 1],
])

front2ego = np.dot(f_cam2ego, front2cam)

front2lidar = np.dot(np.linalg.inv(lidar2ego), front2ego)

# [[ 0.   0.   1.   2.9]
#  [-1.   0.   0.   0. ]
#  [ 0.  -1.   0.  -0.9]
#  [ 0.   0.   0.   1. ]]
print(front2lidar)

print(front2ego)


intrinsics = {
    "distortion_model": "rational_polynomial",
    "height": 1080,
    "width": 1920,
    "k": [
        1144.08,
        0.0,
        960,
        0.0,
        1144.08,
        540,
        0.0,
        0.0,
        1.0
    ],
    "p": [
        1990.7486572265625,
        0.0,
        1021.9733276367188,
        0.0,
        0.0,
        1987.7210693359375,
        658.0899047851562,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0
    ],
    "d": [
        -0.5742809772491455,
        0.2926959991455078,
        0.0004140000091865659,
        -0.002566999988630414,
        0.0,
        7.397613903345349e+31,
        1.16709573755925e-32,
        1.3563156426940112e-19
    ]
}


['CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK']

label_path = "/home/liry/swpld/Data_Processing/label"
img_path = '/home/liry/swpld/Data_Processing/img'

result = []
dir_len = 4

# {4, 6, 8, 17, 18, 19, 20, 21}
# 4是人，6是car 8是自行车 17骑着自行车和摩托车的人  18是卡车 19是大巴 20是消防车 21 摩托车和三轮车   21和17有些重复的。
#class_list = ["pedestrian", "car", "bicycle", "other", "truck", "bus", "construction_vehicle", "motorcycle"]
class_list = ["car", "bicycle", "other", "truck", "bus", "construction_vehicle", "motorcycle"]
class_static = set()
CLASSES = {
    #4:0,
    6:0,
    8:1,
    17:2,
    18:3,
    19:4,
    20:5,
    21:6
}

def get_subdirectories(directory):
    subdirectories = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)
    return subdirectories


from pathlib import Path

directory = Path("/home/liry/swpld/Data_Processing/label/CAM_FRONT")
folders = [item.name for item in directory.iterdir() if item.is_dir()]
lidar_label_path = "/home/liry/swpld/Data_Processing/label/LIDAR_TOP"
img_path = "/home/liry/swpld/Data_Processing/img"
cam_path = "/home/liry/swpld/Data_Processing/label"

for timestamp in folders:
    
    box_id_list = []
    for cam_index_dir in ['CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK']:
        cam_path_dir = os.path.join(cam_path,f'{cam_index_dir}/{timestamp}/CameraInfo.json')

        with open(cam_path_dir,'r') as cam_json_file:
            data = json.load(cam_json_file)
        
        tmp_result = list({item["id"] for item in data["bboxes3D"]})
        box_id_list.append(tmp_result)
    
    combined_list = [item for sublist in box_id_list for item in sublist]
    print(box_id_list)
    print(combined_list)

    index_result = []
    for index in ['CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK']:
        
    # for index in ['front']:
        # path = os.path.join(label_path,f'{index}',timestamp)
        path = os.path.join(lidar_label_path,f'{timestamp}.json')
        print(path)
        if os.path.exists(path):
            if os.path.isdir(path):
                print(path, "目录存在")
            else:
                print("路径存在，但不是一个目录")
        else:
            print("路径不存在")
            continue
 
        print(path)
        with open(path,'r') as json_file:
            data = json.load(json_file)

        value = path.rsplit('/', 2)[-2]
        
        print(value)

        single_label_dic = {}
        img_name = "Color" + f'{timestamp}' + ".png"
        single_label_dic["image_path"] = os.path.join(img_path,f'{index}',"Color",img_name)        

        print(single_label_dic["image_path"])
        print(os.path.join(lidar_label_path,f"{timestamp}.pcd"))

        single_label_dic["lidar_bin_path"] = os.path.join(lidar_label_path,f"{timestamp}.pcd")
        
        single_label_dic["objects"] = []

        single_label_dic["intrinsics"] = intrinsics

        combined_list = [item for sublist in box_id_list for item in sublist]

        if index in ['CAM_FRONT_RIGHT' , 'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK']:
            for bboxes in data["bboxes3D"]:
                tmp_dic = {}

                sensor_pos = bboxes["relativePos"] 
                sensor_rot = bboxes["relativeRot"] 
                size = bboxes['size']
                class_ = bboxes['type']
                id = bboxes["id"]
                
                if id not in combined_list:
                    continue

                class_static.add(class_)

                # tmp_dic["label"] = class_
                tmp_dic["label"] = CLASSES[class_]
                tmp_dic["existence_probability"] = 1
                tmp_dic["id"] = id
                
                xyz = {}
                xyz["x"] = sensor_pos[0]
                xyz["y"] = sensor_pos[1]
                xyz["z"] = sensor_pos[2] 
                
                rpy = {}
                rpy["x"] = 0
                rpy["y"] = 0
                rpy["z"] = sensor_rot[2] 

                tmp_dic["Pose"] = {}
                tmp_dic["Pose"]["position"] = xyz
                tmp_dic["Pose"]["orientation"] = rpy
                
                lwh = {}
                lwh["x"] = size[0]
                lwh["y"] = size[1]
                lwh["z"] = size[2]
                
                tmp_dic["shape"] = lwh

                single_label_dic["objects"].append(tmp_dic)
                
        index_result.append(single_label_dic) 
    
    result.append(index_result)
    
    # break


file_path = "/home/liry/swpld/Data_Processing/label/data.json"
with open(file_path, "w") as json_file:
    json.dump(result, json_file, indent=4)

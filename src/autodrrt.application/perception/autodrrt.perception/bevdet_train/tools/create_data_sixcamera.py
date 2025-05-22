import argparse
#import data_convert.meg_converter as meg_converter
#from data_convert.create_gt_database import create_groundtruth_database
import json 
import re
import os
from os import path as osp
import mmcv
import pyquaternion
import numpy as np 

# CLASSES = {
#     0:"unkonwn",
#     1:"car",
#     2:"truck",
#     3:"bus",
#     4:"motorcycle",
#     5:"bicycle",
#     6:"pedestrian",

# }


sensor2lidar_translation_set={}
sensor2lidar_rotation_set={}
lidar2cam_set={}
sensor2ego_rotation_set = {}

sensor2lidar_translation_set['CAM_FRONT_RIGHT']=[1.10000, -1.00000, -0.90000]
sensor2lidar_rotation_set['CAM_FRONT_RIGHT']=np.array([-0.76604, 0.00000, -0.64279, 0.64279, 0.00000, -0.76604, 0.00000, -1.00000, 0.00000]).reshape(3,3)
sensor2ego_rotation_set['CAM_FRONT_RIGHT']= [0.24184, -0.24184, -0.66446, 0.66446]

sensor2lidar_translation_set['CAM_FRONT_LEFT']=[1.10000, 1.10000, -0.90000]
sensor2lidar_rotation_set['CAM_FRONT_LEFT']=np.array([0.76604, 0.00000, -0.64279, 0.64279, 0.00000, 0.76604, 0.00000, -1.00000, 0.00000]).reshape(3,3)
sensor2ego_rotation_set['CAM_FRONT_LEFT']= [0.66446, -0.66446, -0.24184, 0.24184]

sensor2lidar_translation_set['CAM_BACK_RIGHT']=[-0.80000, -1.00000, -0.90000]
sensor2lidar_rotation_set['CAM_BACK_RIGHT']=np.array([-0.70711, 0.00000, 0.70711, -0.70711, 0.00000, -0.70711, 0.00000, -1.00000, 0.00000]).reshape(3,3)
sensor2ego_rotation_set['CAM_BACK_RIGHT']= [-0.27060, 0.27060, -0.65328, 0.65328]

sensor2lidar_translation_set['CAM_FRONT']=[2.9, 0.0, -0.9]
sensor2lidar_rotation_set['CAM_FRONT']=np.array([0.00000, 0.00000, 1.00000, -1.00000, 0.00000, 0.00000, 0.00000, -1.00000, 0.00000]).reshape(3,3)
sensor2ego_rotation_set['CAM_FRONT'] = [-0.5, 0.5, -0.5, 0.5]

sensor2lidar_translation_set['CAM_BACK_LEFT']=[-0.8, 1.00000, -0.90000]
sensor2lidar_rotation_set['CAM_BACK_LEFT']=np.array([0.70710, 0.00000, 0.70710, -0.70710, 0.00000, 0.70710, 0.00000, -1.00000, 0.00000]).reshape(3,3)
sensor2ego_rotation_set['CAM_BACK_LEFT']= [0.65328, -0.65328, 0.27060, -0.27060]

sensor2lidar_translation_set['CAM_BACK']=[-2.05000, 0.00000, -0.90000]
sensor2lidar_rotation_set['CAM_BACK']=np.array([-0.00000, 0.00000, -1.00000, 1.00000, 0.00000, -0.00000, 0.00000, -1.00000, 0.00000]).reshape(3,3)
sensor2ego_rotation_set['CAM_BACK']= [0.50000, -0.50000, -0.50000, 0.50000, ]

lidar2cam_set['CAM_FRONT_RIGHT']=[[-0.76604, 0.64279, 0.00000, 1.48544], [0.00000, 0.00000, -1.00000, -0.90000], [-0.64279, -0.76604, 0.00000, -0.05898], [0.00000, 0.00000, 0.00000, 1.00000]]

lidar2cam_set['CAM_FRONT_LEFT']=[[0.76604, 0.64279, 0.00000, -1.54972], [0.00000, 0.00000, -1.00000, -0.90000], [-0.64279, 0.76604, 0.00000, -0.13558], [0.00000, 0.00000, 0.00000, 1.00000]]

lidar2cam_set['CAM_BACK_RIGHT']=[[-0.70711, -0.70711, 0.00000, -1.27279], [0.00000, 0.00000, -1.00000, -0.90000], [0.70711, -0.70711, 0.00000, -0.14142], [0.00000, 0.00000, 0.00000, 1.00000]]

lidar2cam_set['CAM_FRONT']=[[0.00000,  -1.00000,  0.00000, 0.00000],[0.00000,  0.00000,  -1.00000,  -0.90000],[1.00000,  0.00000,  0.00000,  -2.90000],[0.00000,  0.00000,  0.00000,  1.00000]]

lidar2cam_set['CAM_BACK_LEFT']=[[0.70711, -0.70711, 0.00000, 1.27279], [0.00000, 0.00000, -1.00000, -0.90000], [0.70711, 0.70711, 0.00000, -0.14142], [0.00000, 0.00000, 0.00000, 1.00000]]
lidar2cam_set['CAM_BACK']=[[-0.00000, 1.00000, 0.00000, -0.00000], [0.00000, 0.00000, -1.00000, -0.90000], [-1.00000, -0.00000, 0.00000, -2.05000], [0.00000, 0.00000, 0.00000, 1.00000]]


camera_types = ['CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK']


def quaternion_to_euler(x, y, z, w):
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x**2 + y**2)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y**2 + z**2)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z


def _fill_trainval_infos(root_path, info_prefix):

    train_infos = []
    val_infos = []
    
    with open(root_path, 'r') as file:
        data = json.load(file)
    
    data_len = len(data)
    
    z_result = []

    for i in range(data_len):
    
        # match = re.search(r'\d+', data[i]["image_path"])
        
        info = {
            "frame_id": data[i][0]["image_path"],
            "token":data[i][0]["image_path"],
            "lidar_path": data[i][0]["lidar_bin_path"],
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": "",
            "lidar2ego_rotation": "",
            "timestamp": 0,
        }
         
        camera_types = ['CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK']
        camera_types_dic = {
            "CAM_FRONT_RIGHT":0,
            "CAM_FRONT_LEFT":1,
            "CAM_BACK_RIGHT":2,
            "CAM_FRONT":3,
            "CAM_BACK_LEFT":4,
            "CAM_BACK":5
        }
        
  
        for cam in camera_types:
            viewpad_extrinsics = np.eye(4)
            # viewpad_extrinsics[:3, :4] = np.array(data[i]["extrinsics"]).reshape((3, 4))
            
            # intrinsic = np.array(data[i]["intrinsics"]["k"]).reshape((3, 3))
            # viewpad_intrinsic = np.eye(4)
            # viewpad_intrinsic[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            # lidar2img = (viewpad_intrinsic @ viewpad_extrinsics)
            camera_index = camera_types_dic[cam]
            cam_info = dict(
                data_path=data[i][camera_index]["image_path"],
                type=cam,
                
                # 目前测试只需要sensor2ego_translation、sensor2ego_rotation和cam_intrinsic这三个参数，另外ego2global_translation和ego2global_rotation还未找到哪里在使用这两个参数
                # https://staff.aist.go.jp/k.koide/workspace/matrix_converter/matrix_converter.html
                # 外参矩阵为
                # [
                #     [-0.01868, -0.99973,  -0.01368, 0.01382],
                #     [-0.06003,    0.01478,    -0.99807,    -0.42437],
                #     [0.99800,     -0.01782,    -0.06059, -0.39388],
                #     [0.0,      0.0,     0.0,    1.0]
                # ]
                # 对其秋逆，然后再求其旋转平移
                # 下面为对其求逆的结果

                #### FRONT CAMERA
                sensor2ego_translation=sensor2lidar_translation_set[cam],
                sensor2ego_rotation=sensor2ego_rotation_set[cam],
                ego2global_translation=[0.0, 0.0, 0.0],
                ego2global_rotation=[1.0, 0.0, 0.0, 0.0], 
                sensor2lidar_translation=sensor2lidar_translation_set[cam],
                sensor2lidar_rotation=sensor2lidar_rotation_set[cam],
                lidar2cam=lidar2cam_set[cam],
                cam_intrinsic=np.array(data[i][camera_index]["intrinsics"]["k"]).reshape((3, 3)),
            )
            info["cams"].update({cam: cam_info})

        info['lidar2ego_translation'] = [0.0, 0.0, 0.0]
        info['lidar2ego_rotation'] = [1.0, 0.0, 0.0, 0.0]
        info['ego2global_translation'] = [0.0, 0.0, 0.0]
        info['ego2global_rotation'] = [1.0, 0.0, 0.0, 0.0]

        gt_boxes = []
        gt_names = []
        
        for cam_gt in camera_types:
            camera_index_gt = camera_types_dic[cam_gt]
            for j in range(len(data[i][camera_index_gt]["objects"])):

                # print(data[i][camera_index_gt])
                pose = data[i][camera_index_gt]["objects"][j]["Pose"]
                label = data[i][camera_index_gt]["objects"][j]["label"]
                shape = data[i][camera_index_gt]["objects"][j]["shape"]
                orientation = pose["orientation"]

                # gt_names.append(CLASSES[label])
                gt_names.append(label)

                # roll, pitch, yaw = quaternion_to_euler(orientation["x"], orientation["y"], orientation["z"], orientation["w"])
                roll, pitch, yaw = orientation["x"], orientation["y"], orientation["z"]

                gt_boxes.append([
                    pose["position"]["x"],
                    pose["position"]["y"],
                    pose["position"]["z"],
                    shape["x"],
                    shape["y"],
                    shape["z"],
                    yaw,
                    0.0,
                    0.0
                ])
                
                # print("xyz")
                # print(shape["x"],shape["y"],shape["z"])

                # if i % 10 == 0:
                # z_result.append(pose["position"]["z"])
            
        info["gt_boxes"] = np.array(gt_boxes)
        info["gt_names"] = np.array(gt_names)

        info["ann_infos"] = [info["gt_boxes"], info["gt_names"]]
        
        # print("===============info ann==================")
        # print(info["ann_infos"])
                
        # val_infos.append(info)
        # train_infos.append(info)
        
        # print(data[i][camera_index]["image_path"].rsplit('/', 8)[-3])

        # if data[i]["image_path"].rsplit('/', 8)[-3] not in camera_types:
        #     train_infos.append(info)
        # else:
        #     val_infos.append(info)
        if (i % 1000 ==0) :
        # if (i < 0) :
            # print(data[i][camera_index]["image_path"])
            val_infos.append(info)
            # print(info["frame_id"])
        else:
            train_infos.append(info)
        
        # print(val_infos)
        # break

    return train_infos, val_infos

def meg_data_prep(root_path, info_prefix, dataset_name, out_dir):
    train_info, val_info = _fill_trainval_infos(root_path, info_prefix)
    
    metadata = dict(version="kunyi-v1.0")

    print(
        "train sample: {}, val sample: {}".format(
            len(train_info), len(val_info)
        )
    )
    data = dict(infos=train_info, metadata=metadata)
    info_path = osp.join("/home/liry/swpld/AD1129/label", "{}_infos_train.pkl".format(info_prefix))
    mmcv.dump(data, info_path)
    data["infos"] = val_info
    info_val_path = osp.join("/home/liry/swpld/AD1129/label", "{}_infos_val.pkl".format(info_prefix))
    mmcv.dump(data, info_val_path)
    

parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="kunyi", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="/home/liry/swpld/Data_Processing/label/data.json",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="./",
    required=False,
    help="name of info pkl",
)
parser.add_argument("--extra-tag", type=str, default="meg")


args = parser.parse_args()

if __name__ == "__main__":

    if args.dataset == "kunyi":
        meg_data_prep(
            root_path=args.root_path,
            info_prefix="kunyi",
            dataset_name="kunyiDataset",
            out_dir=args.out_dir
        )

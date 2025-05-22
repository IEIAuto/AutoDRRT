import json
import numpy as np
import cv2
import os
from vis_pb_lidar_tools import visualize_camera
from scipy.spatial.transform import Rotation as R

# 3d bbox 转化切换到 右手坐标系下的 xyz z向上 

v = 9
print("数据集版本 ：",v)


dir_index = 'CAM_BACK_RIGHT'
cam_path = "/home/liry/swpld/Data_Processing/label"
path =   '/home/liry/swpld/Data_Processing/label/LIDAR_TOP'

if dir_index == 'CAM_FRONT':
    img_path = '/home/liry/swpld/Data_Processing/img/CAM_FRONT/Color/'
    len = 6001

if dir_index == 'CAM_FRONT_LEFT':
    img_path = '/home/liry/swpld/Data_Processing/img/CAM_FRONT_LEFT/Color/'
    len = 6001

if dir_index == 'CAM_BACK_LEFT':
    img_path = '/home/liry/swpld/Data_Processing/img/CAM_BACK_LEFT/Color/'
    len = 6001

if dir_index == 'CAM_BACK':
    img_path = '/home/liry/swpld/Data_Processing/img/CAM_BACK/Color/'
    len = 6001

if dir_index == 'CAM_BACK_RIGHT':
    img_path = '/home/liry/swpld/Data_Processing/img/CAM_BACK_RIGHT/Color/'
    len = 6001

if dir_index == 'CAM_FRONT_RIGHT':
    img_path = '/home/liry/swpld/Data_Processing/img/CAM_FRONT_RIGHT/Color/'
    len = 6001


intrinsics = np.array([
                1144.083499,
                0.0,
                960,
                0.0,
                1144.083499,
                540,
                0.0,
                0.0,
                1.0
            ]).reshape((3,3))


cam2img = np.eye(4)
cam2img[:3, :3] = intrinsics

for index in range(0,len):
    
    if(index % 1000!=0):
        continue

    cam_idx = 600 + index*12
    print(cam_idx)

    box_id_list = []
    for cam_index_dir in ['CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK']:
        cam_path_dir = os.path.join(cam_path,f'{cam_index_dir}/{cam_idx}/CameraInfo.json')

        with open(cam_path_dir,'r') as cam_json_file:
            data = json.load(cam_json_file)
        
        tmp_result = list({item["id"] for item in data["bboxes3D"]})
        box_id_list.append(tmp_result)


    combined_list = [item for sublist in box_id_list for item in sublist]


    # if((cam_idx < 708) or (cam_idx > 708)):
    #     continue
    

    stamp = index*0.5 + 0.5

    path_ = os.path.join(path,f'{cam_idx}.json')
    
    with open(path_,'r') as json_file:
        data = json.load(json_file)
    # print(stamp)
    images = cv2.imread(img_path + f"Color{cam_idx}.png")

    for bboxes in data["bboxes3D"]:
      
        sensor_pos = bboxes["relativePos"] #   xyz  ->> 相机坐标系
        sensor_rot = bboxes["relativeRot"] #   xyz  --> 相机坐标系
        args_pos =  bboxes['pos']
        args_rot =  bboxes['rot']
        size = bboxes['size']
        id = bboxes['id']
        type = bboxes['type']
        
        if id not in combined_list:
            continue

        if type not in [4, 6, 8, 17, 18, 19, 20, 21]:
            continue
            
    
        x  = sensor_pos[0] # y
        y  = sensor_pos[1] # z
        z  = sensor_pos[2] # x

        bboxes_ = np.array([[x,y,z, 
                             size[0], 
                             size[1], 
                             size[2], 
                             sensor_rot[2]]])

        if dir_index == 'CAM_FRONT':
            transformation_matrix_RT = np.array([[0.00000,  -1.00000,  0.00000, 0.00000],[0.00000,  0.00000,  -1.00000,  -0.90000],[1.00000,  0.00000,  0.00000,  -2.90000],[0.00000,  0.00000,  0.00000,  1.00000]])
        elif dir_index == 'CAM_FRONT_LEFT':
            transformation_matrix_RT = np.array([[0.76604, 0.64279, 0.00000, -1.54972], [0.00000, 0.00000, -1.00000, -0.90000], [-0.64279, 0.76604, 0.00000, -0.13558], [0.00000, 0.00000, 0.00000, 1.00000]])
        elif dir_index == 'CAM_FRONT_RIGHT':
            transformation_matrix_RT = [[-0.76604, 0.64279, 0.00000, 1.48544], [0.00000, 0.00000, -1.00000, -0.90000], [-0.64279, -0.76604, 0.00000, -0.05898], [0.00000, 0.00000, 0.00000, 1.00000]]
        elif dir_index == 'CAM_BACK':
            transformation_matrix_RT = [[-0.00000, 1.00000, 0.00000, -0.00000], [0.00000, 0.00000, -1.00000, -0.90000], [-1.00000, -0.00000, 0.00000, -2.05000], [0.00000, 0.00000, 0.00000, 1.00000]]
        elif dir_index == 'CAM_BACK_LEFT':
            transformation_matrix_RT = [[0.70711, -0.70711, 0.00000, 1.27279], [0.00000, 0.00000, -1.00000, -0.90000], [0.70711, 0.70711, 0.00000, -0.14142], [0.00000, 0.00000, 0.00000, 1.00000]]
        elif dir_index == 'CAM_BACK_RIGHT':
            transformation_matrix_RT = [[-0.70711, -0.70711, 0.00000, -1.27279], [0.00000, 0.00000, -1.00000, -0.90000], [0.70711, -0.70711, 0.00000, -0.14142], [0.00000, 0.00000, 0.00000, 1.00000]]
        
        transform =  cam2img @ transformation_matrix_RT  
        images =  visualize_camera(images,bboxes=bboxes_,transform=transform)

    if dir_index == 'CAM_FRONT':
        cv2.imwrite(f'/home/liry/swpld/Data_Processing/img_3d/CAM_FRONT/FRONT_' + str(cam_idx) + '.png', images)
    elif dir_index == 'CAM_FRONT_LEFT':
        cv2.imwrite(f'/home/liry/swpld/Data_Processing/img_3d/CAM_FRONT_LEFT/FRONT_LEFT_' + str(cam_idx) + '.png', images)
    elif dir_index == 'CAM_FRONT_RIGHT':
        cv2.imwrite(f'/home/liry/swpld/Data_Processing/img_3d/CAM_FRONT_RIGHT/FRONT_RIGHT_' + str(cam_idx) + '.png', images)
    elif dir_index == 'CAM_BACK':
        cv2.imwrite(f'/home/liry/swpld/Data_Processing/img_3d/CAM_BACK/BACK_' + str(cam_idx) + '.png', images)
    elif dir_index == 'CAM_BACK_LEFT':
        cv2.imwrite(f'/home/liry/swpld/Data_Processing/img_3d/CAM_BACK_LEFT/BACK_LEFT_' + str(cam_idx) + '.png', images)
    elif dir_index == 'CAM_BACK_RIGHT':
        cv2.imwrite(f'/home/liry/swpld/Data_Processing/img_3d/CAM_BACK_RIGHT/BACK_RIGHT_' + str(cam_idx) + '.png', images)

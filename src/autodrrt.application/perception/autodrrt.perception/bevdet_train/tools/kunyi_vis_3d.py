import json
import numpy as np
import cv2
import os
from vis_pb_lidar_tools import visualize_camera
from scipy.spatial.transform import Rotation as R
import pickle
# 3d bbox 转化切换到 右手坐标系下的 xyz z向上 

v = 9
# path = "new_data/camera/直道城市/1200P F80 K"
# path = "new_data/camera/相机测试"
print("数据集版本 ：",v)

class_list = ["pedestrian","car","bicycle","other","truck","bus","motorcycle"]
class_static = set()
CLASSES = {
    0:"pedestrian",
    1:"car",
    2:"bicycle",
    3:"other",
    4:"truck",
    5:"bus",
    6:"motorcycle"
}

dir_index = 'front'

# intrinsics = np.array([540, 0.0, 540,
#                        0.0, 540, 360,
#                        0.0, 0.0, 1.0]).reshape((3,3))



intrinsics_1 = np.array([1144.08, 0.0, 960,
                       0.0, 1144.08, 540,
                       0.0, 0.0, 1.0]).reshape((3,3))

intrinsics_2 = np.array([1144.08, 0.0, 960,
                       0.0, 1144.08, 540,
                       0.0, 0.0, 1.0]).reshape((3,3))



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
    [1, -1,  0]
])

cam2ego_dir['back'] = np.array([
    [0,  1,  0],
    [0,  0, -1],
    [-1, 0,  0]
])

transformation_matrix = cam2ego_dir[dir_index]
transformation_matrix_RT = np.eye(4)
transformation_matrix_RT[:3, :3] = transformation_matrix


def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
    # print(data["infos"])
    return data 


def inference_result(json_path):
    with open(json_path,'r') as json_file:
        result_json_data = json.load(json_file)
        # print("json is \n")
        # print(len(result_json_data))
        # print(result_json_data[0])
    
    pkl_data = read_pkl("/home/liry/swpld/AD1129/label/kunyi_infos_train.pkl")

    for index, data in enumerate(result_json_data):
        
        # if(index>20):
        #     break
        # if index%100!=0:
        #     continue

        print('index = ', index)


        for cam_type in ['CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK']:

            # img_path = pkl_data["infos"][index]["cams"]["CAM_FRONT"]["data_path"]
            # img_path = pkl_data["infos"][index]["cams"]["CAM_FRONT_RIGHT"]["data_path"]
            img_path = pkl_data["infos"][index]["cams"][cam_type]["data_path"]
            # img_path = pkl_data["infos"][index]["cams"]["CAM_BACK"]["data_path"]
            

            # camera_image = cv2.imread(img_path["frame_id"])  
            
            camera_image = cv2.imread(img_path)


            bbox = data["boxes_3d"]
            label = data["labels_3d"]
            score = data["scores_3d"]

            for idx,obj in enumerate(bbox):

                
                # 按照激光雷达的坐标系可视化
                # xyz = [obj[0], obj[1], obj[2]+0.5*obj[5]]
                # lwh = [obj[3], obj[4], obj[5]]
                # print("yaw")
                # print(obj[6])
                # yaw = obj[6]
                # transform =  cam2img @ transformation_matrix_RT 
                # bboxes_ = np.array([[xyz[0],xyz[1],xyz[2], lwh[0],lwh[1],lwh[2],yaw]])
                # print(bboxes_)

                # result = transform @ np.array([xyz[0],xyz[1],xyz[2],1])
                # print(result/result[2])
                
                # 转换为相机坐标系进行可视化
                
                # transformation_matrix  = np.array([[ 0,   -1.0,   0,   0],
                #     [0,   0,   -1.0,   -0.9, ],
                #     [ 1.0,  0,   0,  -2.9,],
                #     [ 0,   0,   0,   1 ]])
                
                if cam_type=="CAM_FRONT_RIGHT":
                # front right
                    transformation_matrix  = np.array([[-0.76604, 0.64279, 0.00000, 1.48544], [0.00000, 0.00000, -1.00000, -0.90000], [-0.64279, -0.76604, 0.00000, -0.05898], [0.00000, 0.00000, 0.00000, 1.00000]])
                if cam_type=="CAM_FRONT_LEFT":
                # front left
                    transformation_matrix  = np.array([[0.76604, 0.64279, 0.00000, -1.54972], [0.00000, 0.00000, -1.00000, -0.90000], [-0.64279, 0.76604, 0.00000, -0.13558], [0.00000, 0.00000, 0.00000, 1.00000]])
                if cam_type=="CAM_BACK_RIGHT":
                # back
                    transformation_matrix  = np.array([[-0.70711, -0.70711, 0.00000, -1.27279], [0.00000, 0.00000, -1.00000, -0.90000], [0.70711, -0.70711, 0.00000, -0.14142], [0.00000, 0.00000, 0.00000, 1.00000]])
                if cam_type=="CAM_FRONT":
                # back
                    transformation_matrix  = np.array([[0.00000,  -1.00000,  0.00000, 0.00000],[0.00000,  0.00000,  -1.00000,  -0.90000],[1.00000,  0.00000,  0.00000,  -2.90000],[0.00000,  0.00000,  0.00000,  1.00000]])
                if cam_type=="CAM_BACK_LEFT":
                    transformation_matrix  = np.array([[0.70711, -0.70711, 0.00000, 1.27279], [0.00000, 0.00000, -1.00000, -0.90000], [0.70711, 0.70711, 0.00000, -0.14142], [0.00000, 0.00000, 0.00000, 1.00000]])
                if cam_type=="CAM_BACK":
                    transformation_matrix  = np.array([[-0.00000, 1.00000, 0.00000, -0.00000], [0.00000, 0.00000, -1.00000, -0.90000], [-1.00000, -0.00000, 0.00000, -2.05000], [0.00000, 0.00000, 0.00000, 1.00000]])

                # print([obj[0],obj[1],obj[2],1])
                # lidar_box = transformation_matrix @ np.array([obj[0],obj[1],obj[2],1]).reshape(-1,1)
                
                # print("======lidar_box==========")
                # print(lidar_box)

                #if lidar_box[2][0]<80:
                #    continue

                # print('swpld')
                # if score[idx]<0.1:
                #     continue

                bboxes_ = np.array([[obj[0],obj[1],obj[2]+obj[5] * 0.5, obj[3], obj[4], obj[5], obj[6]]])
                
                # bboxes_ = np.array([[lidar_box[0][0], lidar_box[1][0]-0.5*obj[5], lidar_box[2][0], 
                #                      obj[3], obj[4], obj[5], 1.57]])
                
                # if cam_type=="CAM_BACK_LEFT" or cam_type=="CAM_BACK_RIGHT":
                #     cam2img =  np.eye(4)
                #     cam2img[:3, :3] = intrinsics_2
                #     cam2ego_RT = np.eye(4)
                #     transform = cam2img
                # else:
                cam2img =  np.eye(4)
                cam2img[:3, :3] = intrinsics_1
                cam2ego_RT = np.eye(4)
                transform = cam2img @ transformation_matrix
                # transform =  cam2img @ transformation_matrix_RT

                print("bboxes_")
                print(bboxes_)
                print(type(bboxes_))
                print(CLASSES[label[idx]])

                camera_image = visualize_camera(camera_image, bboxes=bboxes_,transform=transform) 
                
                # text = CLASSES[label[idx]]
                # xyz = np.array([lidar_box[0][0],lidar_box[1][0]-0.5*obj[5],lidar_box[2][0],1]).reshape(4,1)
                # xy = cam2img @  xyz
                # x = xy[0][0]/xy[2][0]
                # y = xy[1][0]/xy[2][0]

                # org = (int(x),int(y))
                # font = cv2.FONT_HERSHEY_SIMPLEX  
                # font_scale = 1  
                # color = (0, 255, 0)  
                # thickness = 2 
                # camera_image = cv2.putText(camera_image, text, org, font, font_scale, color, thickness)
    
            # value = img_path["frame_id"].rsplit('/', 2)[-1]
            value = img_path.rsplit('/', 2)[-1]
            
            print(value)        
            
            final_path = '/home/liry/swpld/BEVDet-export/tools/test_result_qdq_percentile/' + str(index) + "/"
            try:
                os.makedirs(final_path, exist_ok=True)
                print(f"目录 '{final_path}' 已创建或已存在。")
            except Exception as e:
                print(f"创建目录 '{final_path}' 时出错: {e}")
            
            cv2.imwrite(final_path + cam_type + value, camera_image)

        

inference_result("/home/liry/swpld/BEVDet-export/kunyi_test_output_qdq_percentile.json")

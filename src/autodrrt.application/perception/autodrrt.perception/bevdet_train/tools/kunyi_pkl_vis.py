import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from vis_pb_lidar_tools import visualize_camera
import os 
def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def pkl_reformat(pkl_data):
    
    for index in range(0,len(pkl_data["infos"])):
        for cam_index in ['CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_BACK']:
            # print(pkl_data["infos"][0]["cams"][cam_index])
            
            if not os.path.exists(pkl_data["infos"][index]["cams"][cam_index]['data_path']):
                continue

            images = cv2.imread(pkl_data["infos"][index]["cams"][cam_index]['data_path'])
            # print(pkl_data["infos"][index]["cams"][cam_index]['data_path'])
            
            filename = pkl_data["infos"][index]["cams"][cam_index]['data_path'].split('/')[-1]  # 获取 "Color4008.png"
            number = filename.replace('Color', '').split('.')[0]

            intrinsics = np.array(pkl_data["infos"][index]["cams"][cam_index]['cam_intrinsic']).reshape((3,3))
            cam2img = np.eye(4)
            cam2img[:3, :3] = intrinsics
            
            transformation_matrix_RT = np.array(pkl_data["infos"][index]["cams"][cam_index]['lidar2cam']).reshape((4,4))

            for box in pkl_data["infos"][index]["gt_boxes"]:
                bboxes_ = np.array([[
                    box[0],
                    box[1],
                    box[2],
                    box[3], 
                    box[4], 
                    box[5], 
                    box[6]]])
                transform =  cam2img @ transformation_matrix_RT  
                images =  visualize_camera(images,bboxes=bboxes_,transform=transform)
            
            os.makedirs('/home/liry/swpld/BEVDet-export/pkl_vis_result/' + str(number), exist_ok=True)

            cv2.imwrite(f'/home/liry/swpld/BEVDet-export/pkl_vis_result/' + str(number) + '/' + str(cam_index) + '.png', images)
    # print(c)

if __name__ == "__main__":
    pkl_file_path = "/home/liry/swpld/AD1129/label/kunyi_infos_train.pkl"
    pkl_data = load_pkl_file(pkl_file_path)
    
    # print(pkl_data)

    pkl_reformat(pkl_data)
    # visualize_3d_boxes(pkl_data)
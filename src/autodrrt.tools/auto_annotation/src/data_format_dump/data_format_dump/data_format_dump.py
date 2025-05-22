import rclpy
import json
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2,Image,CameraInfo
from autoware_auto_perception_msgs.msg import DetectedObjects
from rclpy.node import Node
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
class DataFormatDump(Node):

    def __init__(self):
        super().__init__('data_format_dump')
        self.ini_camera_extrinsics()
        self.image_info_subscription = self.create_subscription(Header,'/message_sync/out/images_info',
            self.listener_callback_image_info,1)
        self.pcl_subscription = self.create_subscription(PointCloud2,'/message_sync/out/pointcloud',
            self.listener_callback_pcl,1)
        self.pcl_info_subscription = self.create_subscription(Header,'/message_sync/out/pointcloud_info',
            self.listener_callback_pcl_info,1)
        self.image_raw_subscription = self.create_subscription(Image,'/message_sync/out/image_raw',
            self.listener_callback_image_raw,1)
        self.camera_info_subscription = self.create_subscription(CameraInfo,'/message_sync/out/camera_info',
            self.listener_callback_camera_info,1)
        self.obj_subscription = self.create_subscription(DetectedObjects,'/message_sync/out/objects',
            self.listener_callback_obj_subscription,1)
            
        # self.image_info_subscription  # prevent unused variable warning
        # self.pcl_subscription  # prevent unused variable warning
        # self.image_raw_subscription  # prevent unused variable warning
        # self.camera_info_subscription  # prevent unused variable warning
        # self.obj_subscription  # prevent unused variable warning

        self.image_info_subscription_flag = False
        self.pcl_subscription_flag = False
        self.image_raw_subscription_flag = False
        self.camera_info_subscription_flag = False
        self.obj_subscription_flag = False
        self.pcl_info_subscription_flag = False
        self.dump_data = []
        
        # self.image_info_msg 
        # self.pcl_msg
        # self.image_raw_msg
        # self.camera_info_msg
        # self.obj_msg


    def listener_callback_image_info(self, msg):
        self.image_info_subscription_flag = True
        self.image_info_msg = msg
        self.is_all_arrived()
        self.get_logger().info('I heard: listener_callback_image_info')

    def listener_callback_pcl_info(self, msg):
        self.pcl_info_subscription_flag = True
        self.pcl_info_msg = msg
        self.is_all_arrived()
        self.get_logger().info('I heard: listener_callback_pcl_info')

    def listener_callback_pcl(self, msg):
        self.pcl_subscription_flag = True
        self.pcl_msg = msg
        self.get_logger().info('I heard: listener_callback_pcl')
        self.is_all_arrived()
    def listener_callback_image_raw(self, msg):
        self.image_raw_subscription_flag = True
        self.image_raw_msg = msg
        self.get_logger().info('I heard: listener_callback_image_raw')
        self.is_all_arrived()

    def listener_callback_camera_info(self, msg):
        self.camera_info_subscription_flag = True
        self.camera_info_msg = msg
        self.get_logger().info('I heard: listener_callback_camera_info')
        self.is_all_arrived()

    def listener_callback_obj_subscription(self, msg):
        self.obj_subscription_flag = True
        self.obj_msg = msg
        self.get_logger().info('I heard: listener_callback_obj_subscription')
        self.is_all_arrived()

    def is_all_arrived(self):
        if self.image_info_subscription_flag and self.pcl_subscription_flag and self.image_raw_subscription_flag \
           and self.camera_info_subscription_flag and self.obj_subscription_flag and self.pcl_info_subscription_flag:
            self.image_info_subscription_flag = False
            self.pcl_subscription_flag = False
            self.image_raw_subscription_flag = False
            self.camera_info_subscription_flag = False
            self.obj_subscription_flag = False
            self.pcl_info_subscription_flag = False
            self.get_logger().info('All message are arrived')
            data = {}
            objects = []
            for obj in self.obj_msg.objects:
                object = {}
                object["label"] = self.find_max_label(obj.classification)
                object["existence_probability"] = obj.existence_probability
                object["Pose"] = self.get_pose(obj.kinematics.pose_with_covariance.pose)
                object["shape"] = self.get_shape(obj.shape.dimensions)
                objects.append(object)
            data["image_path"] = self.image_info_msg.frame_id
            data["lidar_bin_path"] = self.pcl_info_msg.frame_id
            data["objects"] = objects
            data["intrinsics"] = self.get_camera_info(self.camera_info_msg)
            data["extrinsics"] = self.camera_extrinsics[self.camera_info_msg.header.frame_id]
            self.dump_data.append(data)
            with open('data.json','w') as json_file:
                json.dump(self.dump_data,json_file,indent=2)
            # print(self.dump_data)
            try:
                #test projection
                camera_image = cv2.imread(data["image_path"])
                camera_matrix = np.array(data["intrinsics"]["k"]).reshape((3,3))
                camera_extrinsics = np.array(data["extrinsics"]).reshape((3,4))
                camera_extrinsics_rotation = camera_extrinsics[:,:3]
                camera_extrinsics_tran = camera_extrinsics[:,-1].reshape((1,3))
                # print("camera_matrix",camera_matrix)
                # print("camera_extrinsics_rotation",camera_extrinsics_rotation)
                # print("camera_extrinsics_tran",camera_extrinsics_tran)
                recv_matrix, _ = cv2.Rodrigues(camera_extrinsics_rotation) 
                recv = np.array([recv_matrix[0, 0], recv_matrix[1, 0], recv_matrix[2, 0]])
                tran = np.array([camera_extrinsics_tran[0, 0], camera_extrinsics_tran[0, 1], camera_extrinsics_tran[0, 2]])
                dist_coeffs = np.array([-0.574281, 0.292696, 0.000414, -0.002567]).reshape(4,1)
                for obj in data["objects"]:
                    point_3d = [obj["Pose"]["position"]["x"], obj["Pose"]["position"]["y"], obj["Pose"]["position"]["z"]]

                    object_position_radar = np.array([obj["Pose"]["position"]["x"], obj["Pose"]["position"]["y"], obj["Pose"]["position"]["z"]], dtype=np.float64)
                    object_rotation_quaternion = [obj["Pose"]["orientation"]["x"], obj["Pose"]["orientation"]["y"], obj["Pose"]["orientation"]["z"], obj["Pose"]["orientation"]["w"]]
                    object_size = np.array([obj["shape"]["x"], obj["shape"]["y"], obj["shape"]["z"]], dtype=np.float64)
                    length = obj["shape"]["x"]
                    width = obj["shape"]["y"]
                    height = obj["shape"]["z"]



                    center_x = object_position_radar[0]
                    center_y = object_position_radar[1]
                    center_z = object_position_radar[2]
                    
                    object_points_radar = np.array([[center_x - length*0.5, center_y - width*0.5, center_z - height*0.5],                # 物体的坐标系原点
                                                    [0.5*length + center_x, center_y - width*0.5, center_z - height*0.5],               # 
                                                    [0.5*length + center_x, center_y + width*0.5, center_z - height*0.5],              # 物体右下角
                                                    [center_x - length*0.5, center_y + width*0.5, center_z - height*0.5],               # 物体左下角
                                                    [center_x - length*0.5, center_y - width*0.5, center_z + height*0.5],               # 物体上方
                                                    [0.5*length + center_x, center_y - width*0.5, center_z + height*0.5],              # 物体右上方
                                                    [0.5*length + center_x, center_y + width*0.5, center_z + height*0.5],             # 物体右下方
                                                    [center_x - length*0.5, center_y + width*0.5, center_z + height*0.5]], dtype=np.float64)  # 物体左下方


                    # object_points_radar = np.array([[0, 0, 0],                # 物体的坐标系原点
                    #                                 [length, 0, 0],               # 
                    #                                 [length, width, 0],              # 物体右下角
                    #                                 [0, width, 0],               # 物体左下角
                    #                                 [0, 0, height],               # 物体上方
                    #                                 [length, 0, height],              # 物体右上方
                    #                                 [length, width, height],             # 物体右下方
                    #                                 [0, width, height]], dtype=np.float64)  # 物体左下方
                    
                    rotation_matrix_radar = Rotation.from_quat(object_rotation_quaternion).as_matrix()
                    # object_position_radar_center =  np.array([object_position_radar[0] -  length*0.5,
                    #                                          object_position_radar[1] -  width*0.5,
                    #                                          object_position_radar[2] -  height*0.5])
                    # object_points_camera = np.dot(rotation_matrix_radar, object_points_radar.T).T + object_position_radar_center
                    # object_points_camera = np.dot(rotation_matrix_radar, object_points_radar.T).T 
                    # print(object_points_camera)
                    np_points_lidar = np.array(point_3d)
                    left_up_bottom = np.array([center_x + length*0.5, center_y + width*0.5, center_z - height*0.5])
                    left_bottom_bottom = np.array([center_x + length*0.5, center_y - width*0.5, center_z - height*0.5])
                    right_bottom_bottom = np.array([center_x - length*0.5, center_y - width*0.5, center_z - height*0.5])
                    right_up_bottom = np.array([center_x - length*0.5, center_y + width*0.5, center_z - height*0.5])

                    left_up_up = np.array([center_x + length*0.5, center_y + width*0.5, center_z + height*0.5])
                    left_bottom_up = np.array([center_x + length*0.5, center_y - width*0.5, center_z + height*0.5])
                    right_bottom_up = np.array([center_x - length*0.5, center_y - width*0.5, center_z + height*0.5])
                    right_up_up = np.array([center_x - length*0.5, center_y + width*0.5, center_z + height*0.5])

                    left_up_bottom_image, _ = cv2.projectPoints(left_up_bottom, recv, tran, camera_matrix, dist_coeffs)
                    left_bottom_bottom_image, _ = cv2.projectPoints(left_bottom_bottom, recv, tran, camera_matrix, dist_coeffs)
                    right_bottom_bottom_image, _ = cv2.projectPoints(right_bottom_bottom, recv, tran, camera_matrix, dist_coeffs)
                    lright_up_bottom_image, _ = cv2.projectPoints(right_up_bottom, recv, tran, camera_matrix, dist_coeffs)

                    left_up_up_image, _ = cv2.projectPoints(left_up_up, recv, tran, camera_matrix, dist_coeffs)
                    left_bottom_up_image, _ = cv2.projectPoints(left_bottom_up, recv, tran, camera_matrix, dist_coeffs)
                    right_bottom_up_image, _ = cv2.projectPoints(right_bottom_up, recv, tran, camera_matrix, dist_coeffs)
                    right_up_up_image, _ = cv2.projectPoints(right_up_up, recv, tran, camera_matrix, dist_coeffs)

                    # cv2.circle(camera_image, tuple((left_up_bottom_image[0,0].astype(int))), 10, (0, 255, 0), -1)
                    # cv2.circle(camera_image, tuple((left_bottom_bottom_image[0,0].astype(int))), 10, (255, 0, 0), -1)

                    # cv2.circle(camera_image, tuple((right_bottom_bottom_image[0,0].astype(int))), 10, (0, 255, 0), -1)
                    # cv2.circle(camera_image, tuple((lright_up_bottom_image[0,0].astype(int))), 10, (255, 0, 0), -1)

                    # cv2.circle(camera_image, tuple((left_up_up_image[0,0].astype(int))), 10, (0, 255, 0), -1)
                    # cv2.circle(camera_image, tuple((left_bottom_up_image[0,0].astype(int))), 10, (255, 0, 0), -1)

                    # cv2.circle(camera_image, tuple((right_bottom_up_image[0,0].astype(int))), 10, (0, 255, 0), -1)
                    # cv2.circle(camera_image, tuple((right_up_up_image[0,0].astype(int))), 10, (255, 0, 0), -1)

                    try:

                        camera_image = cv2.line(camera_image, tuple(left_up_bottom_image[0,0].astype(int)),tuple(left_bottom_bottom_image[0,0].astype(int)), (0, 0, 255), 2)

                        camera_image = cv2.line(camera_image, tuple(left_bottom_bottom_image[0,0].astype(int)),tuple(right_bottom_bottom_image[0,0].astype(int)), (0, 0, 255), 2)
                        camera_image = cv2.line(camera_image, tuple(right_bottom_bottom_image[0,0].astype(int)),tuple(lright_up_bottom_image[0,0].astype(int)), (0, 0, 255), 2)
                        camera_image = cv2.line(camera_image, tuple(lright_up_bottom_image[0,0].astype(int)),tuple(left_up_bottom_image[0,0].astype(int)), (0, 0, 255), 2)

                        camera_image = cv2.line(camera_image, tuple(left_up_up_image[0,0].astype(int)),tuple(left_bottom_up_image[0,0].astype(int)), (0, 0, 255), 2)

                        camera_image = cv2.line(camera_image, tuple(left_bottom_up_image[0,0].astype(int)),tuple(right_bottom_up_image[0,0].astype(int)), (0, 0, 255), 2)
                        camera_image = cv2.line(camera_image, tuple(right_bottom_up_image[0,0].astype(int)),tuple(right_up_up_image[0,0].astype(int)), (0, 0, 255), 2)
                        camera_image = cv2.line(camera_image, tuple(right_up_up_image[0,0].astype(int)),tuple(left_up_up_image[0,0].astype(int)), (0, 0, 255), 2)


                        camera_image = cv2.line(camera_image, tuple(left_up_up_image[0,0].astype(int)),tuple(left_up_bottom_image[0,0].astype(int)), (0, 0, 255), 2)

                        camera_image = cv2.line(camera_image, tuple(left_bottom_up_image[0,0].astype(int)),tuple(left_bottom_bottom_image[0,0].astype(int)), (0, 0, 255), 2)
                        camera_image = cv2.line(camera_image, tuple(right_bottom_up_image[0,0].astype(int)),tuple(right_bottom_bottom_image[0,0].astype(int)), (0, 0, 255), 2)
                        camera_image = cv2.line(camera_image, tuple(right_up_up_image[0,0].astype(int)),tuple(lright_up_bottom_image[0,0].astype(int)), (0, 0, 255), 2)


                        # np_points_left_bottom = np.array([center_x - width*0.5, center_y - length*0.5, center_z - height*0.5])
                        
                        # np_points_left_bottom_image, _ = cv2.projectPoints(np_points_left_bottom, recv, tran, camera_matrix, dist_coeffs)
                        # np_points_right_bottom_image, _ = cv2.projectPoints(np_points_right_bottom, recv, tran, camera_matrix, dist_coeffs)

                        # np_points_left_bottom_image, _ = cv2.projectPoints(np_points_left_bottom, recv, tran, camera_matrix, dist_coeffs)
                        # np_points_right_bottom_image, _ = cv2.projectPoints(np_points_right_bottom, recv, tran, camera_matrix, dist_coeffs)




                        # image_point, _ = cv2.projectPoints(object_points_camera, recv, tran, camera_matrix, dist_coeffs)
                        # image_points = image_point.astype(int)
                        # camera_image = cv2.polylines(camera_image, [image_points[:4]], True, (0, 0, 255), 2)  # 绘制物体底部的轮廓
                        # camera_image = cv2.polylines(camera_image, [image_points[4:]], True, (0, 0, 255), 2)  # 绘制物体上部的轮廓
                        # camera_image = cv2.line(camera_image, tuple(image_points[0,0]),tuple(image_points[4,0]), (0, 0, 255), 2)
                        # camera_image = cv2.line(camera_image, tuple(image_points[1,0]),tuple(image_points[5,0]), (0, 0, 255), 2)
                        # camera_image = cv2.line(camera_image, tuple(image_points[2,0]),tuple(image_points[6,0]), (0, 0, 255), 2)
                        # camera_image = cv2.line(camera_image, tuple(image_points[3,0]),tuple(image_points[7,0]), (0, 0, 255), 2)
                        # print("np_points_image",np_points_image)
                    except Exception as e:
                        print("np_points_lidar",np_points_lidar,"left_up_bottom_image",left_up_bottom_image)
                        print(e)
                    np_points_image, _ = cv2.projectPoints(np_points_lidar, recv, tran, camera_matrix, dist_coeffs)
                    cv2.circle(camera_image, tuple((np_points_image[0,0].astype(int))), 10, (0, 0, 255), -1)
                
                    # cv2.circle(camera_image, tuple((np_points_left_bottom_image[0,0].astype(int))), 10, (0, 255, 0), -1)
                    # cv2.circle(camera_image, tuple((np_points_right_bottom_image[0,0].astype(int))), 10, (255, 0, 0), -1)
                    # print("point_3d", point_3d, "np_points_left_bottom", np_points_left_bottom)
                cv2.imwrite('output.jpg',camera_image)
            except Exception as e:
                print(e)
            return True
        else:
            return False
        

    def find_max_label(self, classification):
        probability = 0
        label = None
        for cla in classification:
            if cla.probability > probability:
                probability = cla.probability
                label = cla.label
        return label
    
    def get_pose(self, pose_ini):
        pose = {}
        position = {}
        orientation = {}
        position["x"] = pose_ini.position.x
        position["y"] = pose_ini.position.y
        position["z"] = pose_ini.position.z
        orientation["x"] =  pose_ini.orientation.x
        orientation["y"] =  pose_ini.orientation.y
        orientation["z"] =  pose_ini.orientation.z
        orientation["w"] =  pose_ini.orientation.w
        pose["position"] = position
        pose["orientation"] = orientation
        return pose
    def get_shape(self,shape_ini):
        shape = {}
        shape["x"] = shape_ini.x
        shape["y"] = shape_ini.y
        shape["z"] = shape_ini.z
        return shape
    def get_camera_info(self,camera_info_ini):
        camera_info = {}
        camera_info["distortion_model"] = camera_info_ini.distortion_model
        camera_info["height"] = camera_info_ini.height
        camera_info["width"] = camera_info_ini.width
        camera_info["k"] = camera_info_ini.k.tolist()
        camera_info["p"] = camera_info_ini.p.tolist()
        camera_info["d"] = camera_info_ini.d.tolist()
        return camera_info


    def ini_camera_extrinsics(self):

        lidar2camera_front = [-0.01868,-0.99973,-0.01368,0.01382,-0.06003,0.01478,-0.99807,-0.42437,0.99800,-0.01782,-0.06059,-0.39388]
        lidar2camera_front_right = [-0.80019,0.52965,-0.28135, 0.66076,0.49450,0.31723,-0.80922,-0.12643,-0.33935,-0.78666,-0.51576,-0.83938]
        lidar2camera_front_left = [0.67640,0.73651,-0.00637, 1.31756,0.13062,-0.12847,-0.98307,-0.46398,-0.72486, 0.66412, -0.18310,-2.77461]
        lidar2camera_rare_left = [0.82807,-0.54493,0.13173, 10.06543,0.02932,-0.19255,-0.98085,-1.18875,0.55986,0.81607,-0.14347, 5.35462]
        lidar2camera_rare_right =[-0.76772,-0.62478,-0.14234,-8.80969,0.04450,0.16961,-0.98451,-0.92781,0.63924,-0.76216,-0.10241, 6.04771]
        lidar2camera_rare_center = [0.11000,0.99041,0.08355,1.28100,0.57775,0.00469,-0.81620, 7.30028, -0.80877,0.13805,-0.57170, -9.68777]
        self.camera_extrinsics = {}
        self.camera_extrinsics["front_camera_optical_link"] = lidar2camera_front
        self.camera_extrinsics["front_left_camera_optical_link"] = lidar2camera_front_left
        self.camera_extrinsics["front_right_camera_optical_link"] = lidar2camera_front_right
        self.camera_extrinsics["rare_center_camera_optical_link"] = lidar2camera_rare_center
        self.camera_extrinsics["rare_left_camera_optical_link"] = lidar2camera_rare_left
        self.camera_extrinsics["rare_right_camera_optical_link"] = lidar2camera_rare_right

def main(args=None):
    rclpy.init(args=args)

    data_format_dump = DataFormatDump()

    rclpy.spin(data_format_dump)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    data_format_dump.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
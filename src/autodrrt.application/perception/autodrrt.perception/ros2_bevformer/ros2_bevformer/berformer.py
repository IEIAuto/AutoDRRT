import rclpy
from rclpy.node import Node

from mmdeploy.backend.tensorrt import load_tensorrt_plugin
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import torch
import mmcv
import copy
import numpy as np
from mmcv import Config
import os
from typing import List, Optional, Tuple
# import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import message_filters
import torch
from collections import OrderedDict
from ros2_bevformer.realtime_loading import *
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from autoware_auto_perception_msgs.msg import DetectedObjects
from ros2_bevformer.utils import *
from sensor_msgs.msg import Image
import yaml
import importlib
import numpy as np 
# from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset
# from BEVFormer_tensorrt_deploy.tools.bevformer import evaluate_trt

import sys
sys.path.append(os.getcwd() + "/ros2_bevformer/BEVFormer_tensorrt_deploy")

from third_party.bev_mmdet3d.models.builder import build_model
from det2trt.utils.tensorrt import (
    get_logger,
    create_engine_context,
    allocate_buffers,
    do_inference,
)

class BEVmodule(Node) :
    def __init__(self) :
        super().__init__('bevformer_module')
        
        # 创建一个新的 QoSProfile，根据需要设置 best_effort 策略
        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5,
        ) 
        
        self.declare_parameter("trt_model_path", "")
        self.declare_parameter("config_path", "")

        self.declare_parameter("/camera_0/raw_image", "/camera_0/raw_image")
        self.declare_parameter("/camera_1/raw_image", "/camera_1/raw_image")
        self.declare_parameter("/camera_2/raw_image", "/camera_2/raw_image")
        self.declare_parameter("/camera_3/raw_image", "/camera_3/raw_image")
        self.declare_parameter("/camera_4/raw_image", "/camera_4/raw_image")
        self.declare_parameter("/camera_5/raw_image", "/camera_5/raw_image")
        
    
        front_sub = self.get_parameter("/camera_0/raw_image").get_parameter_value().string_value
        front_right_sub = self.get_parameter("/camera_1/raw_image").get_parameter_value().string_value
        front_left_sub = self.get_parameter("/camera_2/raw_image").get_parameter_value().string_value
        back_sub = self.get_parameter("/camera_3/raw_image").get_parameter_value().string_value
        back_right_sub = self.get_parameter("/camera_4/raw_image").get_parameter_value().string_value
        back_left_sub = self.get_parameter("/camera_5/raw_image").get_parameter_value().string_value

        
    
        trt_model = self.get_parameter("trt_model_path").get_parameter_value().string_value
        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        config = Config.fromfile(config_path)
        
        self.declare_parameter("bevformer_config_path", "bevformer_config_path")

        with open(self.get_parameter("bevformer_config_path").get_parameter_value().string_value, "r") as f:
            self.bevformer_config_path = yaml.safe_load(f)["/**"]["ros__parameters"]
        
        self.input_shapes_ = self.bevformer_config_path["input_shapes"]
        self.output_shapes_ = self.bevformer_config_path["output_shapes"]
        self.class_names_ = self.bevformer_config_path["class_names"]

        self.prev_bev = np.array([0.0] * 9600, dtype=np.float32)
        self.use_prev_bev = np.array([0.0] * 1, dtype=np.float32)
        self.can_bus = np.array([0.0] * 18, dtype=np.float32)
        self.lidar2img = np.array([0.0] * 96, dtype=np.float32)

        # 使用 best_effort_qos 作为 Subscriber 的 qos_profile 参数
        self.front_sub = message_filters.Subscriber(self, Image, front_sub, qos_profile=best_effort_qos)
        self.front_right_sub = message_filters.Subscriber(self, Image, front_right_sub, qos_profile=best_effort_qos)
        self.front_left_sub = message_filters.Subscriber(self, Image, front_left_sub, qos_profile=best_effort_qos)
        self.back_sub = message_filters.Subscriber(self, Image, back_sub, qos_profile=best_effort_qos)
        self.back_right_sub = message_filters.Subscriber(self, Image, back_right_sub, qos_profile=best_effort_qos)
        self.back_left_sub = message_filters.Subscriber(self, Image, back_left_sub, qos_profile=best_effort_qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
           [self.front_sub, 
            self.front_right_sub, 
            self.front_left_sub,
            self.back_sub, 
            self.back_right_sub, 
            self.back_left_sub], 
            100,
            .2, 
            allow_headerless=True
            )
        self.ts.registerCallback(self.bev_callback)

        self.get_logger().info("bevformer start")
        
        self.od_pub = self.create_publisher(DetectedObjects, 'pub_detection', 10)

        load_tensorrt_plugin()
        #  print([pc.name for pc in trt.get_plugin_registry().plugin_creator_list])        
        if hasattr(config, "plugin"):
            if isinstance(config.plugin, list):
                print(config.plugin)
                for plu in config.plugin:
                    importlib.import_module(plu)
            else:
                importlib.import_module(config.plugin)
        self.pth_model = build_model(config.model, test_cfg=config.get("test_cfg"))    
        self.get_logger().info("prepare env end")
        
        TRT_LOGGER = get_logger(trt.Logger.INTERNAL_ERROR)
        # self.engine, self.context = create_engine_context(trt_model, TRT_LOGGER)
        self.engine_bev, self.context_bev = create_engine_context(trt_model, TRT_LOGGER)
        self.stream = cuda.Stream()
        

    #CALL_BACK
    def bev_callback(self, f_msg, fr_msg, fl_msg, b_msg, br_msg, bl_msg) :
    # def bev_callback(self) :
        
        # from PIL import Image
        # f_msg = Image.open('/home/orin/disk/bevformer_ws/src/bevformer_ros2/BEVFormer_tensorrt_deploy/data/nuscenes/sweeps/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603612404.jpg')
        # fr_msg = Image.open('/home/orin/disk/bevformer_ws/src/bevformer_ros2/BEVFormer_tensorrt_deploy/data/nuscenes/sweeps/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603620482.jpg')
        # fl_msg = Image.open('/home/orin/disk/bevformer_ws/src/bevformer_ros2/BEVFormer_tensorrt_deploy/data/nuscenes/sweeps/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603604799.jpg')
        # b_msg = Image.open('/home/orin/disk/bevformer_ws/src/bevformer_ros2/BEVFormer_tensorrt_deploy/data/nuscenes/sweeps/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603637558.jpg')
        # br_msg = Image.open('/home/orin/disk/bevformer_ws/src/bevformer_ros2/BEVFormer_tensorrt_deploy/data/nuscenes/sweeps/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603628113.jpg')
        # bl_msg = Image.open('/home/orin/disk/bevformer_ws/src/bevformer_ros2/BEVFormer_tensorrt_deploy/data/nuscenes/sweeps/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603647405.jpg')

        self.get_logger().info('callback start')

        ##################################################################################
        #                                IMAGE CONVERT                                   #
        ##################################################################################
        #<------LoadMultiViewImageFromFiles------>
        img_results_0=load_images(f_msg, fr_msg, fl_msg, b_msg, br_msg, bl_msg)
        # #<---------------ImageAug3D-------------->
        img_results_1=img_augmentation(img_results_0)
        # #<-------------ImageNormalize------------>
        img_results_2=img_normalize(img_results_1)

        # self.f = bridge.imgmsg_to_cv2(f_msg, "bgr8")
        # self.fr = bridge.imgmsg_to_cv2(fr_msg, "bgr8")
        # self.fl = bridge.imgmsg_to_cv2(fl_msg, "bgr8")
        # self.b = bridge.imgmsg_to_cv2(b_msg, "bgr8")
        # self.br = bridge.imgmsg_to_cv2(br_msg, "bgr8")
        # self.bl = bridge.imgmsg_to_cv2(bl_msg, "bgr8")

        # cv2.imwrite('/home/nvidia/BEVfusion/data/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg', self.f)
        # cv2.imwrite('/home/nvidia/BEVfusion/data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402927620339.jpg', self.fr)
        # cv2.imwrite('/home/nvidia/BEVfusion/data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_LEFT__1532402927604844.jpg', self.fl)
        # cv2.imwrite('/home/nvidia/BEVfusion/data/nuscenes/samples/CAM_BACK/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', self.b)
        # cv2.imwrite('/home/nvidia/BEVfusion/data/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_BACK_RIGHT__1532402927627893.jpg', self.br)
        # cv2.imwrite('/home/nvidia/BEVfusion/data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-07-24-11-22-45+0800__CAM_BACK_LEFT__1532402927647423.jpg', self.bl)


        img = np.array(img_results_2['img'][0], dtype=np.float32)
        
        
        inputs, outputs, bindings = allocate_buffers(
            self.engine_bev, self.context_bev, input_shapes=self.input_shapes_, output_shapes=self.output_shapes_
        )

        for inp in inputs:
            if inp.name == "image":
                inp.host = img.reshape(-1).astype(np.float32)
            elif inp.name == "prev_bev":
                inp.host = self.prev_bev.reshape(-1).astype(np.float32)
            elif inp.name == "use_prev_bev":
                inp.host = self.use_prev_bev.reshape(-1).astype(np.float32)
            elif inp.name == "can_bus":
                inp.host = self.can_bus.reshape(-1).astype(np.float32)
            elif inp.name == "lidar2img":
                inp.host = self.lidar2img.reshape(-1).astype(np.float32)
            else:
                raise RuntimeError(f"Cannot find input name {inp.name}.")
        
        trt_outputs, t = do_inference(
            self.context_bev, bindings=bindings, inputs=inputs, outputs=outputs, stream=self.stream
        )
        trt_outputs = {
            out.name: out.host.reshape(*self.output_shapes_[out.name]) for out in trt_outputs
        }
        
        self.prev_bev = trt_outputs.pop("bev_embed")
        # prev_frame_info["prev_pos"] = tmp_pos
        # prev_frame_info["prev_angle"] = tmp_angle
        
        trt_outputs = {k: torch.from_numpy(v) for k, v in trt_outputs.items()}
        
        img_metas = {"box_type_3d":'third_party.bev_mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'}
        result = self.pth_model.post_process(**trt_outputs, img_metas=img_metas)
        
        pts_bbox = result[0]['pts_bbox']
        boxes_3d_enc = pts_bbox['boxes_3d']
        scores_3d = pts_bbox['scores_3d']
        labels_3d = pts_bbox['labels_3d']

        filter = scores_3d >= 0.0
        boxes_3d_enc.tensor = boxes_3d_enc.tensor[filter]
        boxes_3d = boxes_3d_enc.tensor.numpy() # [[cx, cy, cz, w, l, h, rot, vx, vy]]
        scores_3d = scores_3d[filter].numpy()
        labels_3d = labels_3d[filter].numpy()
        custom_boxes_3d = []
        for i, box_3d in enumerate(boxes_3d):
            box3d = CustomBox3D(labels_3d[i], scores_3d[i],
                                box_3d[0],box_3d[1],box_3d[2],
                                box_3d[3],box_3d[4],box_3d[5],
                                box_3d[6],box_3d[7],box_3d[8])
            custom_boxes_3d.append(box3d)
            
        output_msg = DetectedObjects()
        obj_num = len(boxes_3d)
        for i, box3d in enumerate(custom_boxes_3d):
            obj = box3DToDetectedObject(box3d, self.class_names_, True, False)
            output_msg.objects.append(obj)

        output_msg.header.stamp = self.get_clock().now().to_msg() #rclpy.time.Time()
        output_msg.header.frame_id = "base_link"
        
        self.od_pub.publish(output_msg)
        
        self.get_logger().info('callback end')

def main() -> None:
    rclpy.init()
   
    node = BEVmodule()
    # rclpy.spin(node)
    try :
        rclpy.spin(node)
    except KeyboardInterrupt :
        node.get_logger().info('Stopped by Keyboard')
    finally :
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "bevfusion/bevfusion.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"
#include "common/visualize.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <vector>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <autoware_auto_perception_msgs/msg/detected_object_kinematics.hpp>
#include <autoware_auto_perception_msgs/msg/detected_objects.hpp>
#include <autoware_auto_perception_msgs/msg/object_classification.hpp>
#include <autoware_auto_perception_msgs/msg/shape.hpp>
#include <tier4_autoware_utils/geometry/geometry.hpp>
#include <tier4_autoware_utils/math/constants.hpp>

#include <algorithm>
#include <dirent.h>

#include <cstdio>
#include <fstream>
#include <thread>
#include <cassert>

#include <yaml-cpp/yaml.h>
#include "bevdet.h"
#include "bevdet_ros2.h"
#include "cpu_jpegdecoder.h"


using namespace std::chrono_literals;
using Label = autoware_auto_perception_msgs::msg::ObjectClassification;

using std::chrono::duration;
using std::chrono::high_resolution_clock;


half __nvhalf2half(nvtype::half a) {
    half b;
    std::memcpy(&b, &a, sizeof(half));
    return b;
}


static inline nvtype::half __internal_float2half(const float f)
{
      unsigned int x;
      unsigned int u;
      unsigned int result;
      unsigned int sign;
      unsigned int remainder;
      (void)memcpy(&x, &f, sizeof(f));
      u = (x & 0x7fffffffU);
      sign = ((x >> 16U) & 0x8000U);
      // NaN/+Inf/-Inf
      if (u >= 0x7f800000U)
      {
          remainder = 0U;
          result = ((u == 0x7f800000U) ? (sign | 0x7c00U) : 0x7fffU);
      }
      else if (u > 0x477fefffU)
      { // Overflows
          remainder = 0x80000000U;
          result = (sign | 0x7bffU);
      }
      else if (u >= 0x38800000U)
      { // Normal numbers
          remainder = u << 19U;
          u -= 0x38000000U;
          result = (sign | (u >> 13U));
      }
      else if (u < 0x33000001U)
      { // +0/-0
          remainder = u;
          result = sign;
      }
      else
      { // Denormal numbers
          const unsigned int exponent = u >> 23U;
          const unsigned int shift = 0x7eU - exponent;
          unsigned int mantissa = (u & 0x7fffffU);
          mantissa |= 0x800000U;
          remainder = mantissa << (32U - shift);
          result = (sign | (mantissa >> shift));
          result &= 0x0000FFFFU;
      }

      unsigned short x_tmp = static_cast<unsigned short>(result);
      if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((x & 0x1U) != 0U)))
      {
          x_tmp++;
      }

    nvtype::half half_out;
    (void)memcpy(&half_out, &x_tmp, sizeof(x_tmp));
    return half_out;
}


void Getinfo(void) {
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem >> 10);
        printf("  Shared memory in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0],
                prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1],
                prop.maxGridSize[2]);
    }
    printf("\n");
}


void Boxes2Txt(const std::vector<Box> &boxes, std::string file_name, bool with_vel=false) {
    std::ofstream out_file;
    out_file.open(file_name, std::ios::out);
    if (out_file.is_open()) {
        for (const auto &box : boxes) {
            out_file << box.x << " ";
            out_file << box.y << " ";
            out_file << box.z << " ";
            out_file << box.l << " ";
            out_file << box.w << " ";
            out_file << box.h << " ";
            out_file << box.r << " ";
            if (with_vel) {
                out_file << box.vx << " ";
                out_file << box.vy << " ";
            }
            out_file << box.score << " ";
            out_file << box.label << "\n";
        }
    }
    out_file.close();
    return;
};


void Egobox2Lidarbox(const std::vector<Box>& ego_boxes, std::vector<Box> &lidar_boxes,
                     const Eigen::Quaternion<float> &lidar2ego_rot, const Eigen::Translation3f &lidar2ego_trans) {
    for (size_t i = 0; i < ego_boxes.size(); i++) {
        Box b = ego_boxes[i];
        Eigen::Vector3f center(b.x, b.y, b.z);
        center -= lidar2ego_trans.translation();
        center = lidar2ego_rot.inverse().matrix() * center;
        b.r -= lidar2ego_rot.matrix().eulerAngles(0, 1, 2).z();
        b.x = center.x();
        b.y = center.y();
        b.z = center.z();
        lidar_boxes.push_back(b);
    }
}


//static void visualize(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, const nv::Tensor& lidar_points, const std::vector<unsigned char*> images, const nv::Tensor& lidar2image, const std::string& save_path, cudaStream_t stream) {
static void visualize(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, const nv::Tensor& lidar_points, 
                      const unsigned char** images, const size_t camera_num, 
		      const nv::Tensor& lidar2image, const std::string& save_path, cudaStream_t stream) {
    std::vector<nv::Prediction> predictions(bboxes.size());
    memcpy(predictions.data(), bboxes.data(), bboxes.size() * sizeof(nv::Prediction));

    int padding = 300;
    int lidar_size = 1024;
    int content_width = lidar_size + padding * 3;
    int content_height = 1080;
    nv::SceneArtistParameter scene_artist_param;
    scene_artist_param.width = content_width;
    scene_artist_param.height = content_height;
    scene_artist_param.stride = scene_artist_param.width * 3;

    nv::Tensor scene_device_image(std::vector<int>{scene_artist_param.height, scene_artist_param.width, 3}, nv::DataType::UInt8);
    scene_device_image.memset(0x00, stream);

    scene_artist_param.image_device = scene_device_image.ptr<unsigned char>();
    auto scene = nv::create_scene_artist(scene_artist_param);

    nv::BEVArtistParameter bev_artist_param;
    bev_artist_param.image_width = content_width;
    bev_artist_param.image_height = content_height;
    bev_artist_param.rotate_x = 70.0f;
    bev_artist_param.norm_size = lidar_size * 0.5f;
    bev_artist_param.cx = content_width * 0.5f;
    bev_artist_param.cy = content_height * 0.5f;
    bev_artist_param.image_stride = scene_artist_param.stride;

    auto points = lidar_points.to_device();
    auto bev_visualizer = nv::create_bev_artist(bev_artist_param);
    bev_visualizer->draw_lidar_points(points.ptr<nvtype::half>(), points.size(0));
    bev_visualizer->draw_prediction(predictions, false);
    bev_visualizer->draw_ego();
    bev_visualizer->apply(scene_device_image.ptr<unsigned char>(), stream);

    nv::ImageArtistParameter image_artist_param;
    image_artist_param.num_camera = camera_num;
    image_artist_param.image_width = 1600;
    image_artist_param.image_height = 900;
    image_artist_param.image_stride = image_artist_param.image_width * 3;
    image_artist_param.viewport_nx4x4.resize(camera_num * 4 * 4);
    memcpy(image_artist_param.viewport_nx4x4.data(), lidar2image.ptr<float>(), sizeof(float) * image_artist_param.viewport_nx4x4.size());

    int gap = 0;
    int camera_width = 500;
    int camera_height = static_cast<float>(camera_width / (float)image_artist_param.image_width * image_artist_param.image_height);
    int offset_cameras[][3] = {
        {-camera_width / 2, -content_height / 2 + gap, 0},
        {content_width / 2 - camera_width - gap, -content_height / 2 + camera_height / 2, 0},
        {-content_width / 2 + gap, -content_height / 2 + camera_height / 2, 0},
        {-camera_width / 2, +content_height / 2 - camera_height - gap, 1},
        {-content_width / 2 + gap, +content_height / 2 - camera_height - camera_height / 2, 0},
        {content_width / 2 - camera_width - gap, +content_height / 2 - camera_height - camera_height / 2, 1}};

    auto visualizer = nv::create_image_artist(image_artist_param);
    for (size_t icamera = 0; icamera < camera_num; ++icamera) {
        int ox = offset_cameras[icamera][0] + content_width / 2;
        int oy = offset_cameras[icamera][1] + content_height / 2;
        bool xflip = static_cast<bool>(offset_cameras[icamera][2]);
        visualizer->draw_prediction(icamera, predictions, xflip);

        nv::Tensor device_image(std::vector<int>{900, 1600, 3}, nv::DataType::UInt8);
        device_image.copy_from_host(images[icamera], stream);

        if (xflip) {
            auto clone = device_image.clone(stream);
            scene->flipx(clone.ptr<unsigned char>(), clone.size(1), clone.size(1) * 3, clone.size(0), device_image.ptr<unsigned char>(), device_image.size(1) * 3, stream);
            checkRuntime(cudaStreamSynchronize(stream));
        }
        visualizer->apply(device_image.ptr<unsigned char>(), stream);

        scene->resize_to(device_image.ptr<unsigned char>(), ox, oy, ox + camera_width, oy + camera_height, device_image.size(1), device_image.size(1) * 3, device_image.size(0), 0.8f, stream);
        checkRuntime(cudaStreamSynchronize(stream));
    }

    printf("Save to %s\n", save_path.c_str());
    stbi_write_jpg(save_path.c_str(), scene_device_image.size(1), scene_device_image.size(0), 3, scene_device_image.to_host(stream).ptr(), 100);
}


std::shared_ptr<bevfusion::Core> create_core(const std::string& model, const std::string& precision) {
    printf("Create by %s, %s\n", model.c_str(), precision.c_str());
    bevfusion::camera::NormalizationParameter normalization;
    normalization.image_width = 1600;
    normalization.image_height = 900;
    normalization.output_width = 704;
    normalization.output_height = 256;
    normalization.num_camera = 6;
    normalization.resize_lim = 0.48f;
    normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;

    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);

    bevfusion::lidar::VoxelizationParameter voxelization;
    voxelization.min_range = nvtype::Float3(-54.0f, -54.0f, -5.0);
    voxelization.max_range = nvtype::Float3(+54.0f, +54.0f, +3.0);
    voxelization.voxel_size = nvtype::Float3(0.075f, 0.075f, 0.2f);

    voxelization.grid_size = voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);

    voxelization.max_points_per_voxel = 10;
    voxelization.max_points = 300000;
    voxelization.max_voxels = 160000;
    voxelization.num_feature = 5;

    bevfusion::lidar::SCNParameter scn;
    scn.voxelization = voxelization;
    //scn.model = nv::format("model/%s/lidar.backbone.xyz.onnx", model.c_str());
    scn.model = nv::format("./src/ROS2-CUDA-BEVFusion/model/%s/lidar.backbone.xyz.onnx", model.c_str());
    scn.order = bevfusion::lidar::CoordinateOrder::XYZ;

    if (precision == "int8") {
        scn.precision = bevfusion::lidar::Precision::Int8;
    } else {
        scn.precision = bevfusion::lidar::Precision::Float16;
    }

    bevfusion::camera::GeometryParameter geometry;
    geometry.xbound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
    geometry.ybound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
    geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
    geometry.dbound = nvtype::Float3(1.0, 60.0f, 0.5f);
    geometry.image_width = 704;
    geometry.image_height = 256;
    geometry.feat_width = 88;
    geometry.feat_height = 32;
    geometry.num_camera = 6;
    geometry.geometry_dim = nvtype::Int3(360, 360, 80);

    bevfusion::head::transbbox::TransBBoxParameter transbbox;
    transbbox.out_size_factor = 8;
    transbbox.pc_range = {-54.0f, -54.0f};
    transbbox.post_center_range_start = {-61.2, -61.2, -10.0};
    transbbox.post_center_range_end = {61.2, 61.2, 10.0};
    transbbox.voxel_size = {0.075, 0.075};
    //transbbox.model = nv::format("model/%s/build/head.bbox.plan", model.c_str());
    transbbox.model = nv::format("./src/ROS2-CUDA-BEVFusion/model/%s/build/head.bbox.plan", model.c_str());
    transbbox.confidence_threshold = 0.12f;
    transbbox.sorted_bboxes = true;

    bevfusion::CoreParameter param;
    //param.camera_model = nv::format("model/%s/build/camera.backbone.plan", model.c_str());
    param.camera_model = nv::format("./src/ROS2-CUDA-BEVFusion/model/%s/build/camera.backbone.plan", model.c_str());
    param.normalize = normalization;
    param.lidar_scn = scn;
    param.geometry = geometry;
    //param.transfusion = nv::format("model/%s/build/fuser.plan", model.c_str());
    param.transfusion = nv::format("./src/ROS2-CUDA-BEVFusion/model/%s/build/fuser.plan", model.c_str());
    param.transbbox = transbbox;
    //param.camera_vtransform = nv::format("model/%s/build/camera.vtransform.plan", model.c_str());
    param.camera_vtransform = nv::format("./src/ROS2-CUDA-BEVFusion/model/%s/build/camera.vtransform.plan", model.c_str());
    return bevfusion::create_core(param);
}


std::vector<bevfusion::head::transbbox::BoundingBox> bbox_forward(std::shared_ptr<bevfusion::Core> core, std::shared_ptr<BEVDet> bevdet, nv::Tensor pcl_data, const unsigned char** uchar_images, size_t camera_num, camsData sampleData , bool img_got, bool pts_got) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    core->print();
    core->set_timer(false);

    bool fus_got = false;
    if (img_got && pts_got) { fus_got = true; } 

    std::string test_data = "./src/ROS2-CUDA-BEVFusion/dump/";
    const char* data = test_data.c_str();
  
    // Load matrix to host
    auto camera2lidar = nv::Tensor::load(nv::format("%s/camera2lidar.tensor", data), false);
    auto camera_intrinsics = nv::Tensor::load(nv::format("%s/camera_intrinsics.tensor", data), false);
    auto lidar2image = nv::Tensor::load(nv::format("%s/lidar2image.tensor", data), false);
    auto img_aug_matrix = nv::Tensor::load(nv::format("%s/img_aug_matrix.tensor", data), false);
    core->update(camera2lidar.ptr<float>(), camera_intrinsics.ptr<float>(), lidar2image.ptr<float>(), img_aug_matrix.ptr<float>(), stream);
    // core->free_excess_memory();

    // Load image and lidar to host
    auto images = uchar_images;
    auto lidar_points = pcl_data;
    auto datain = sampleData;
    
    std::vector<Box> ego_boxes;
    ego_boxes.clear();
 
    std::vector<bevfusion::head::transbbox::BoundingBox> bboxes;
    bboxes.clear();

    float time = 0.f; 
    if (fus_got) {
        std::cout << "fusion running" << std::endl;
        bevdet->DoInfer_1(datain, time);
        const nvtype::half* img_bevfeat = bevdet->get_bevfeat();

        auto pts_bevfeat = core->forward_pts(lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);
	bboxes = core->forward_fusion(img_bevfeat, pts_bevfeat, stream);
    } else if (pts_got) {
        std::cout << "lidar running" << std::endl;
        auto pts_bevfeat = core->forward_pts(lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);

        nvtype::half* null_img_bevfeat;
        checkRuntime(cudaMalloc(&null_img_bevfeat, 1 * 80 * 180 * 180 * sizeof(nvtype::half)));
        checkRuntime(cudaMemset(null_img_bevfeat, 0, 1 * 80 * 180 * 180 * sizeof(nvtype::half)));
        const nvtype::half* blank_img_bevfeat = null_img_bevfeat;
 
        bboxes = core->forward_fusion(blank_img_bevfeat, pts_bevfeat, stream); 
        checkRuntime(cudaFree(null_img_bevfeat));
    } else if (img_got) {
        std::cout << "camera running" << std::endl;
        bevdet->DoInfer_1(datain, time);
        bevdet->DoInfer_2(datain, ego_boxes, time);
 
        std::vector<Box> lidar_boxes;
        lidar_boxes.clear();
        Egobox2Lidarbox(ego_boxes, lidar_boxes, datain.param.lidar2ego_rot, datain.param.lidar2ego_trans);

        for (auto & box3d: lidar_boxes) {
        //for (auto & box3d: ego_boxes) {
            bevfusion::head::transbbox::BoundingBox elem; 
            elem.position.x  = box3d.x; 
            elem.position.y  = box3d.y;
            elem.position.z  = box3d.z;
            elem.size.w      = box3d.w;
            elem.size.l      = box3d.l;
            elem.size.h      = box3d.h;
            elem.velocity.vx = box3d.vx;
            elem.velocity.vy = box3d.vy;
            elem.z_rotation  = box3d.r;
            elem.score       = box3d.score;
            elem.id          = box3d.label;
            bboxes.push_back(elem);
        }
    }

/***************************************************************/

    // visualize and save to jpg
    visualize(bboxes, lidar_points, images, camera_num, lidar2image, test_data + "/cuda-bevfusion.jpg", stream);
  
    // destroy memory
    //free_images(images);
    delete images;
    checkRuntime(cudaStreamDestroy(stream));
  
    return bboxes;
}

void box3DToDetectedObject(const bevfusion::head::transbbox::BoundingBox & box3d, autoware_auto_perception_msgs::msg::DetectedObject & obj) {
  
    obj.existence_probability = box3d.score;

    // classification
    autoware_auto_perception_msgs::msg::ObjectClassification classification;
    classification.probability = 1.0f;
  
    if (box3d.id == 0) {
        classification.label = Label::CAR;
    } else if (box3d.id == 1) {
        classification.label = Label::TRUCK;
    } else if (box3d.id == 3) {
        classification.label = Label::BUS;
    } else if (box3d.id == 4) {
        classification.label = Label::TRAILER;
    } else if (box3d.id == 7) {
        classification.label = Label::BICYCLE;
    } else if (box3d.id == 6) {
        classification.label = Label::MOTORCYCLE;
    } else if (box3d.id == 8) {
        classification.label = Label::PEDESTRIAN;
    } else {
        classification.label = Label::UNKNOWN;
    }

    obj.classification.emplace_back(classification);

    float yaw = -box3d.z_rotation - tier4_autoware_utils::pi / 2;
    obj.kinematics.pose_with_covariance.pose.position =
        tier4_autoware_utils::createPoint(box3d.position.x, box3d.position.y, box3d.position.z);
    obj.kinematics.pose_with_covariance.pose.orientation =
        tier4_autoware_utils::createQuaternionFromYaw(yaw);
    obj.shape.type = autoware_auto_perception_msgs::msg::Shape::BOUNDING_BOX;
    obj.shape.dimensions = tier4_autoware_utils::createTranslation(box3d.size.l, box3d.size.w, box3d.size.h);

    // twist
    float vel_x = box3d.velocity.vx;
    float vel_y = box3d.velocity.vy;
    geometry_msgs::msg::Twist twist;
    twist.linear.x = std::sqrt(std::pow(vel_x, 2) + std::pow(vel_y, 2));
    twist.angular.z = 2 * (std::atan2(vel_y, vel_x) - yaw);
    obj.kinematics.twist_with_covariance.twist = twist;
    obj.kinematics.has_twist = "true";
}

class BEVFusionNode : public rclcpp::Node
{
    public:
       BEVFusionNode() : Node("bevfusion"), count_(0)
        { 
            Getinfo();
            //std::string pkg_path_ = ament_index_cpp::get_package_share_directory("bevfusion");
            YAML::Node config = YAML::LoadFile(config_file);

            size_t img_N = config["N"].as<size_t>();
            int img_w    = config["W"].as<int>();
            int img_h    = config["H"].as<int>();
            std::string data_info_path         = config["dataset_info"].as<std::string>();
            std::string model_config           = config["ModelConfig"].as<std::string>();
            std::string imgstage_file          = config["ImgStageEngine"].as<std::string>();
            std::string bevstage_file          = config["BEVStageEngine"].as<std::string>();
            std::string output_dir             = config["OutputDir"].as<std::string>();
            YAML::Node camconfig               = YAML::LoadFile(config["CamConfig"].as<std::string>());
            YAML::Node sample                  = config["sample"];
            std::vector<std::string> cams_name = config["cams"].as<std::vector<std::string>>();

            for (auto file: sample) {
                imgs_file.push_back("./" + file.second.as<std::string>());
                imgs_name.push_back(file.first.as<std::string>());
            }

            sampleData.param = camParams(camconfig, img_N, imgs_name);

            //nuscenes_ = std::make_shared<DataLoader>(img_N, img_h, img_w, data_info_path, cams_name);
            //bevdet_ = std::make_shared<BEVDet>(model_config, img_N, nuscenes_->get_cams_intrin(), nuscenes_->get_cams2ego_rot(), nuscenes_->get_cams2ego_trans(), imgstage_file, bevstage_file);
            bevdet_ = std::make_shared<BEVDet>(model_config, img_N, sampleData.param.cams_intrin, sampleData.param.cams2ego_rot, sampleData.param.cams2ego_trans, imgstage_file, bevstage_file);

            std::cout << "BEVDet start"<< std::endl;

            core_ = create_core(model, precision);
            if (core_ == nullptr) {
                printf("Core has been failed.\n");
                return;
            }
            
            std::cout << "BEVFusion start"<< std::endl;

            uchar_images = new unsigned char*[img_N];
            for (size_t i = 0; i < img_N; i++) { uchar_images[i] = new unsigned char[img_w * img_h * 3]; }
            CHECK_CUDA(cudaMalloc((void**)&imgs_dev, img_N * 3 * img_w * img_h * sizeof(uchar)));

            objects_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>("/bevfusion/boundingbox", rclcpp::QoS(1));

            using std::placeholders::_1;
            using std::placeholders::_2;
            using std::placeholders::_3;
            using std::placeholders::_4;
            using std::placeholders::_5;
            using std::placeholders::_6;
            using std::placeholders::_7;

            rmw_qos_profile_t image_rmw_qos(rclcpp::SensorDataQoS().get_rmw_qos_profile());
            image_rmw_qos.depth = 7;
            image_rmw_qos.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;

            image_sub_sync[0].subscribe(this, "/front/image_rawtime", image_rmw_qos);
            image_sub_sync[1].subscribe(this, "/front_Right/image_rawtime", image_rmw_qos);
            image_sub_sync[2].subscribe(this, "/front_Left/image_rawtime", image_rmw_qos);
            image_sub_sync[3].subscribe(this, "/Back/image_rawtime", image_rmw_qos);
            image_sub_sync[4].subscribe(this, "/Back_Left/image_rawtime", image_rmw_qos);
            image_sub_sync[5].subscribe(this, "/Back_Right/image_rawtime", image_rmw_qos);
            pcl_sub_sync.subscribe(this, "/sensing/lidar/top/outlier_filtered/pointcloud", image_rmw_qos);
           
            sync_stream.reset(new message_filters::Synchronizer<approximate_policy_stream>(approximate_policy_stream(30), image_sub_sync[0], image_sub_sync[1], image_sub_sync[2], image_sub_sync[3], image_sub_sync[4], image_sub_sync[5], pcl_sub_sync));

            sync_stream->registerCallback(std::bind(&BEVFusionNode::sync_stream_callback, this, _1, _2, _3, _4, _5, _6, _7));
        }

        ~BEVFusionNode() 
        {
            delete imgs_dev;
            for (size_t i = 0; i < img_N; i++) { delete uchar_images[i]; }
            delete uchar_images;
        }

        void image_trans_to_uncharptr_to_gpu(const std::vector<sensor_msgs::msg::Image>& msg_total, unsigned char* & out_imgs_dev, unsigned char** & out_imgs_host)
        {
            cv::Mat cv_image[img_N];
            // 定义一个变量指向gpu
            unsigned char* temp_gpu = nullptr;
            unsigned char* temp = new unsigned char[img_w * img_h * 3];
            // gpu分配内存 存储1张图像的大小 sizeof(uchar) = 1
            CHECK_CUDA(cudaMalloc(&temp_gpu, img_h * img_w * 3));

            for (size_t i = 0; i < img_N; i++) {
                cv_image[i] = cv_bridge::toCvCopy(msg_total[i], sensor_msgs::image_encodings::BGR8)->image;
                cv::resize(cv_image[i], cv_image[i], cv::Size(img_w, img_h));
                std::memcpy(out_imgs_host[i], cv_image[i].data, img_w * img_h * 3 * sizeof(unsigned char));
                cv::Mat test_img(img_h, img_w, CV_8UC3, out_imgs_host[i]);
                CHECK_CUDA(cudaMemcpy(temp_gpu, out_imgs_host[i], img_w * img_h * 3, cudaMemcpyHostToDevice));
                convert_RGBHWC_to_BGRCHW(temp_gpu, out_imgs_dev + i * img_w * img_h * 3, 3, img_h, img_w);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaFree(temp_gpu));
            delete[] temp;
        }

        static nv::Tensor Pcl2tensor(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*msg, *cloud);

            nvtype::half *points = new nvtype::half[cloud->points.size() * 5];

            auto timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 0.00000001;

            for (size_t i = 0; i < cloud->points.size(); i++) {
                points[i*5 + 0] = __internal_float2half(cloud->points[i].x);
                points[i*5 + 1] = __internal_float2half(cloud->points[i].y);
                points[i*5 + 2] = __internal_float2half(cloud->points[i].z);
                points[i*5 + 3] = __internal_float2half(cloud->points[i].intensity);
                points[i*5 + 4] = __internal_float2half(timestamp);
            }
            std::vector<int32_t> shape{cloud->points.size(), 5};
            nv::Tensor nvtensor_cloud = nv::Tensor::from_data_reference(points, shape, nv::DataType::Float16, false);

            return nvtensor_cloud;
        }

    private:
        void sync_stream_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg0, const sensor_msgs::msg::Image::ConstSharedPtr &msg1, const sensor_msgs::msg::Image::ConstSharedPtr &msg2, const sensor_msgs::msg::Image::ConstSharedPtr &msg3, const sensor_msgs::msg::Image::ConstSharedPtr &msg4, const sensor_msgs::msg::Image::ConstSharedPtr &msg5, const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg6) {

            std::vector<sensor_msgs::msg::Image> msg_total;

            msg_total.emplace_back(*msg0);
            msg_total.emplace_back(*msg1);
            msg_total.emplace_back(*msg2);
            msg_total.emplace_back(*msg3);
            msg_total.emplace_back(*msg4);
            msg_total.emplace_back(*msg5);

            image_trans_to_uncharptr_to_gpu(msg_total, imgs_dev, uchar_images);
            sampleData.imgs_dev = imgs_dev;

            auto pcl_data = Pcl2tensor(msg6);
            const unsigned char** image_data_ptr = const_cast<const unsigned char**>(uchar_images);

            img_got = true;
            pts_got = true;

            auto bboxes = bbox_forward(core_, bevdet_, pcl_data, image_data_ptr, img_N, sampleData, img_got, pts_got);
        }
/*
        void timer_callback(std::shared_ptr<bevfusion::Core> core, std::shared_ptr<BEVDet> bevdet, std::shared_ptr<DataLoader> nuscenes)
        {
            auto bboxes = bbox_forward(core, bevdet, rootDirectory, nuscenes, img_got, pts_got, count_);

            autoware_auto_perception_msgs::msg::DetectedObjects output_msg;
            // output_msg.header = input_pointcloud_msg->header;
            for (const auto & box3d : bboxes) {
                autoware_auto_perception_msgs::msg::DetectedObject obj;
                box3DToDetectedObject(box3d, obj);
                output_msg.objects.emplace_back(obj);
            }
            objects_pub_->publish(output_msg);
            count_++;
        }
*/

        bool img_got = false;
        bool pts_got = false;
        size_t count_;

        //rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<autoware_auto_perception_msgs::msg::DetectedObjects>::SharedPtr objects_pub_;

        std::shared_ptr<BEVDet> bevdet_;
        std::shared_ptr<DataLoader> nuscenes_;
        std::shared_ptr<bevfusion::Core> core_;

        message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_sync[6];
        message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pcl_sub_sync;

        using approximate_policy_stream = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud2>;
        typedef message_filters::Synchronizer<approximate_policy_stream> Synchron_stream;
        //std::shared_ptr<Synchron_stream> sync_stream;
        std::unique_ptr<Synchron_stream> sync_stream;

        size_t img_N;
        int img_w;
        int img_h;
	
        unsigned char* imgs_dev = nullptr;
        unsigned char** uchar_images;
        camsData sampleData;
        std::vector<std::string> imgs_file;
        std::vector<std::string> imgs_name;

        const std::string model = this->declare_parameter<std::string>("model", "resnet50");
        const std::string precision = this->declare_parameter<std::string>("precision", "fp16");
        const std::string rootDirectory = this->declare_parameter<std::string>("rootDirectory", "./src/ROS2-CUDA-BEVFusion/dump/");
        const std::string config_file = "./src/ROS2-CUDA-BEVFusion/configure.yaml"; 
};


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BEVFusionNode>());
    rclcpp::shutdown();
    return 0;
}

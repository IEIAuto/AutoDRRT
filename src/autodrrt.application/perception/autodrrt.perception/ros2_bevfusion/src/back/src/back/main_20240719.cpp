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

double getime()
{
    struct timespec time1 = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time1);
    return (time1.tv_sec + time1.tv_nsec * 0.000000001);
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


static std::vector<unsigned char*> load_images(const std::string& root) {
    const char* file_names[] = {"0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
                                "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"};

    std::vector<unsigned char*> images;
    for (int i = 0; i < 6; ++i) {
        char path[200];
        sprintf(path, "%s/%s", root.c_str(), file_names[i]);

        int width, height, channels;
        images.push_back(stbi_load(path, &width, &height, &channels, 0));
        //std::cout << stbi_load(path, &width, &height, &channels, 0) << std::endl;
        // printf("Image info[%d]: %d x %d : %d\n", i, width, height, channels);
    }
    return images;
}

static void free_images(std::vector<unsigned char*>& images) {
    for (size_t i = 0; i < images.size(); ++i) stbi_image_free(images[i]);
    images.clear();
}

static void visualize(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, const nv::Tensor& lidar_points,
                      const std::vector<unsigned char*> images, const nv::Tensor& lidar2image, const std::string& save_path, cudaStream_t stream) {
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
    image_artist_param.num_camera = images.size();
    image_artist_param.image_width = 1600;
    image_artist_param.image_height = 900;
    image_artist_param.image_stride = image_artist_param.image_width * 3;
    image_artist_param.viewport_nx4x4.resize(images.size() * 4 * 4);
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
    for (size_t icamera = 0; icamera < images.size(); ++icamera) {
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


// 递归遍历文件夹并将子文件夹名称存储在vector中
void TraverseDirectories(const std::string& path, std::vector<std::string>& subDirectories) {
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        std::cerr << "Error opening directory: " << path << std::endl;
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        if (entry->d_type == DT_DIR)
        {
            std::string dirName = entry->d_name;
            if (dirName != "." && dirName != "..")
            {
                subDirectories.push_back(dirName);
                std::string subDirPath = path + "/" + dirName;
                TraverseDirectories(subDirPath, subDirectories);
            }
        }
    }

    closedir(dir);
}

std::vector<bevfusion::head::transbbox::BoundingBox> bbox_forward(int data_index, std::string model, std::string precision, std::string rootDirectory, YAML::Node &config, nvtype::half* img_bevfeat_device) {
//std::vector<bevfusion::head::transbbox::BoundingBox> bbox_forward(std::shared_ptr<bevfusion::Core> core, std::shared_ptr<BEVDet> bevdet, nvtype::half* img_bevfeat_device, std::string rootDirectory, std::shared_ptr<DataLoader> nuscenes, bool img_got, bool pts_got, int data_index) {

    size_t img_N = config["N"].as<size_t>();
    int img_w = config["W"].as<int>();
    int img_h = config["H"].as<int>();
    std::string data_info_path = config["dataset_info"].as<std::string>();
    std::string model_config = config["ModelConfig"].as<std::string>();
    std::string imgstage_file = config["ImgStageEngine"].as<std::string>();
    std::string bevstage_file = config["BEVStageEngine"].as<std::string>();
    std::string output_dir = config["OutputDir"].as<std::string>();
    std::vector<std::string> cams_name = config["cams"].as<std::vector<std::string>>();

    std::cout << "BEVDet start"<< std::endl;

    DataLoader nuscenes(img_N, img_h, img_w, data_info_path, cams_name);
    BEVDet bevdet(model_config, img_N, nuscenes.get_cams_intrin(), nuscenes.get_cams2ego_rot(), nuscenes.get_cams2ego_trans(), imgstage_file, bevstage_file);

    double sum_time = 0;
    int  cnt = 0;

    auto core = create_core(model, precision);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
 
    core->print();
    core->set_timer(false);

    bool img_got = false;
    bool pts_got = true;

    bool fus_got = false;
    if (img_got && pts_got) { fus_got = true; } 

    std::vector<std::string> subDirectories;
    TraverseDirectories(rootDirectory, subDirectories);
    std::sort(subDirectories.begin(), subDirectories.end());
    std::string dirName = subDirectories[data_index];
    std::string test_data = rootDirectory + dirName;

    const char* data = test_data.c_str();
  
    // Load matrix to host
    auto camera2lidar = nv::Tensor::load(nv::format("%s/camera2lidar.tensor", data), false);
    auto camera_intrinsics = nv::Tensor::load(nv::format("%s/camera_intrinsics.tensor", data), false);
    auto lidar2image = nv::Tensor::load(nv::format("%s/lidar2image.tensor", data), false);
    auto img_aug_matrix = nv::Tensor::load(nv::format("%s/img_aug_matrix.tensor", data), false);
    core->update(camera2lidar.ptr<float>(), camera_intrinsics.ptr<float>(), lidar2image.ptr<float>(), img_aug_matrix.ptr<float>(), stream);
    // core->free_excess_memory();

    // Load image and lidar to host
    auto images = load_images(data);
    auto lidar_points = nv::Tensor::load(nv::format("%s/points.tensor", data), false);

    auto datain = nuscenes.data(data_index);

    std::vector<Box> ego_boxes;
    ego_boxes.clear();
 
    std::vector<bevfusion::head::transbbox::BoundingBox> bboxes;
    bboxes.clear();

    float time = 0.f; 
    
    auto lidar_feature = core->forward_pts(lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);
    cudaStreamSynchronize(stream);

    checkRuntime(cudaMemset(img_bevfeat_device, 0, 1 * 80 * 180 * 180 * sizeof(nvtype::half)));
    //checkRuntime(cudaMemset(img_bevfeat_device, __float2half(0.0f), 1 * 80 * 180 * 180 * sizeof(nvtype::half)));

    nvtype::half* test = new nvtype::half[1, 80, 180, 180];

    cudaStreamSynchronize(stream);
    checkRuntime(cudaMemcpy(test, img_bevfeat_device, 1 * 80 * 180 * 180 * sizeof(nvtype::half), cudaMemcpyDeviceToHost));
    cudaStreamSynchronize(stream);

    for (size_t i = 0; i < 1 * 80 * 180 * 180; i++) {
        std::cout << "i = " << i << std::endl;
        std::cout << __half2float(__nvhalf2half(*(test + i))) << " ";
    }
    std::cout << std::endl; 
 
    const nvtype::half* camera_feature = img_bevfeat_device;

    cudaStreamSynchronize(stream);

    bboxes = core->forward_fusion(camera_feature, lidar_feature, stream);
    cudaStreamSynchronize(stream);

    /* 
    auto time5 = high_resolution_clock::now();
    bevdet.DoInfer_1(datain, time, data_index);
    bevdet.DoInfer_2(datain, ego_boxes, time, data_index);
    auto time6 = high_resolution_clock::now();

    duration<double> post_3 = time6 - time5;

    std::cout << "img_stream bevdet_total = " << post_3.count() * 1000 << std::endl;
 
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
    */


    /*
    if (fus_got) {
        std::cout << "~~~~~~~~~~~~~~~~~Fusion stream~~~~~~~~~~~~~~~" << std::endl;

        auto time1 = high_resolution_clock::now();
        bevdet.DoInfer_1(datain, bevstage_buffer, time, data_index);
        auto time2 = high_resolution_clock::now();

        auto time3 = high_resolution_clock::now();
        bboxes = core->forward_bb((const unsigned char**)images.data(), img_bevfeat_device, lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);
        auto time4 = high_resolution_clock::now();

        duration<double> post_1 = time2 - time1;
        duration<double> post_2 = time4 - time3;

        std::cout << "fus_stream img_extract = " << post_1.count() * 1000 << std::endl;
        std::cout << "fus_stream lidar+fusion+head = " << post_2.count() * 1000 << std::endl;
    } else if (img_got) {
        std::cout << "~~~~~~~~~~~~~~~~~Camera stream~~~~~~~~~~~~~~~" << std::endl;

        auto time5 = high_resolution_clock::now();
        bevdet.DoInfer_1(datain, bevstage_buffer, time, data_index);
        bevdet.DoInfer_2(bevstage_buffer, ego_boxes, time);
        auto time6 = high_resolution_clock::now();

        duration<double> post_3 = time6 - time5;

        std::cout << "img_stream bevdet_total = " << post_3.count() * 1000 << std::endl;
 
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
    } else if (pts_got) {
        // warmup 
        std::cout << "~~~~~~~~~~~~~~~~~Lidar stream~~~~~~~~~~~~~~~" << std::endl;

        auto time7 = high_resolution_clock::now();
        bboxes = core->forward_bb((const unsigned char**)images.data(), img_bevfeat_device, lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);
        auto time8 = high_resolution_clock::now();

        duration<double> post_4 = time8 - time7;
        std::cout << "lidar_stream bevfusion_total = " << post_4.count() * 1000 << std::endl; 
    }
    */    


/***************************************************************/
/*
    auto bboxes = core->forward((const unsigned char**)images.data(), lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);

    // evaluate inference time
    for (int i = 0; i < 5; ++i) {
        core->forward((const unsigned char**)images.data(), lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);
    }
*/
  
    // visualize and save to jpg
    visualize(bboxes, lidar_points, images, lidar2image, test_data + "/cuda-bevfusion.jpg", stream);
  
    // destroy memory
    free_images(images);
    checkRuntime(cudaStreamDestroy(stream));
  
    return bboxes;
}

void box3DToDetectedObject(const bevfusion::head::transbbox::BoundingBox & box3d, 
                           autoware_auto_perception_msgs::msg::DetectedObject & obj) {
  
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

class MinimalPublisher : public rclcpp::Node
{
    public:
        MinimalPublisher() : Node("ros2_bevfusion"), count_(0)
        { 
            Getinfo();

            checkRuntime(cudaMalloc(&img_bevfeat_device, 1 * 80 * 180 * 180 * sizeof(nvtype::half)));

            objects_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>("/bevfusion/boundingbox", 10);
            timer_ = this->create_wall_timer(100ms, std::bind(&MinimalPublisher::timer_callback, this));

            checkRuntime(cudaFree(img_bevfeat_device));

        }

    private:
        void timer_callback()
        {
            YAML::Node config = YAML::LoadFile(config_file);
            auto bboxes = bbox_forward(count_, model, precision, rootDirectory, config, img_bevfeat_device);

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

/*
        void timer_callback(std::shared_ptr<bevfusion::Core> core, std::shared_ptr<BEVDet> bevdet, std::shared_ptr<DataLoader> nuscenes)
        {
            auto bboxes = bbox_forward(core, bevdet, img_bevfeat_device, rootDirectory, nuscenes, img_got, pts_got, count_);

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
        //bool img_got = false;
        //bool pts_got = false;
        size_t count_;

        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<autoware_auto_perception_msgs::msg::DetectedObjects>::SharedPtr objects_pub_;
	nvtype::half* img_bevfeat_device = nullptr;

        //std::shared_ptr<bevfusion::Core> core_;
        //std::shared_ptr<BEVDet> bevdet_;
        //std::shared_ptr<DataLoader> nuscenes_;

        const std::string model = this->declare_parameter<std::string>("model", "resnet50");
        const std::string precision = this->declare_parameter<std::string>("precision", "fp16");
        const std::string rootDirectory = this->declare_parameter<std::string>("rootDirectory", "./src/ROS2-CUDA-BEVFusion/dump/");
        const std::string config_file = "./src/ROS2-CUDA-BEVFusion/configure.yaml"; 
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalPublisher>());
    rclcpp::shutdown();
    return 0;
}

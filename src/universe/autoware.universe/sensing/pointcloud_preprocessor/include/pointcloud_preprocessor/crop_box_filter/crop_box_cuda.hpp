// Copyright 2022 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef POINTCLOUD_PREPROCESSOR_CROP_BOX_CUDA_HPP
#define POINTCLOUD_PREPROCESSOR_CROP_BOX_CUDA_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sensor_msgs/msg/point_cloud2.h>
#include <cstdint>
namespace pointcloud_preprocessor
{
    // const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input;
    // void filterPointCloudCUDA(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input, sensor_msgs::msg::PointCloud2 & output);
    void filterPointCloudCUDA(  const uint8_t* input_data, const int input_size, const int point_step,
        const float min_x, const float max_x, const float min_y, const float max_y, 
        const float min_z, const float max_z, const bool negative,
        float* output_data, int* output_size,int* out_offset);
    void test_cuda();
    // void filterPointCloudCUDA();

}  // namespace pointcloud_preprocessor

#endif  // POINTCLOUD_PREPROCESSOR_CROP_BOX_CUDA_HPP

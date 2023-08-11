// Copyright 2021 Tier IV, Inc. All rights reserved.
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

#ifndef GROUND_SEGMENTATION_OPT__SCAN_GROUND_FILTER_NODELET_HPP_
#define GROUND_SEGMENTATION_OPT__SCAN_GROUND_FILTER_NODELET_HPP_

#include "pointcloud_preprocessor/filter.hpp"

#include <vehicle_info_util/vehicle_info.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/transform_datatypes.h>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_eigen/tf2_eigen.h>
#else
#include <tf2_eigen/tf2_eigen.hpp>
#endif

#include <tf2_ros/transform_listener.h>

#include <memory>
#include <string>
#include <vector>
#include <atomic>

#include "threadPool.hpp"

namespace ground_segmentation_opt
{
using vehicle_info_util::VehicleInfo;
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using PointCloud2ConstPtr = sensor_msgs::msg::PointCloud2::ConstSharedPtr;
using IndicesPtr = pcl::IndicesPtr;

  void filter(
    const PointCloud2ConstPtr & input, const IndicesPtr & indices, PointCloud2 & output,const int thread_numbers, const VehicleInfo & vehicle_info_);



}  // namespace ground_segmentation_opt

#endif  // GROUND_SEGMENTATION_OPT__SCAN_GROUND_FILTER_NODELET_HPP_

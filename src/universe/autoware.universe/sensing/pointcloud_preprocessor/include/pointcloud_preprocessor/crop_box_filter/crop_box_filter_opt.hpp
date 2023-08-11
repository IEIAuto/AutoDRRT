// Copyright 2020 Tier IV, Inc.
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

#ifndef POINTCLOUD_PREPROCESSOR__OUTLIER_FILTER__CROP_BOX_FILTER_OPT_HPP_
#define POINTCLOUD_PREPROCESSOR__OUTLIER_FILTER__CROP_BOX_FILTER_OPT_HPP_

#include <algorithm>
#include <vector>
#include <point_cloud_msg_wrapper/point_cloud_msg_wrapper.hpp>
#include <memory>
#include <string>
#include <vector>

// PCL includes
#include <boost/thread/mutex.hpp>
#include "autoware_point_types/types.hpp"
#include <pcl/filters/filter.h>
#include <sensor_msgs/msg/point_cloud2.h>
// PCL includes
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/msg/model_coefficients.h>
#include <pcl_msgs/msg/point_indices.h>

// Include TF
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/transform_listener.h>

// Include tier4 autoware utils
#include <tier4_autoware_utils/ros/debug_publisher.hpp>
#include <tier4_autoware_utils/system/stop_watch.hpp>
namespace crop_box_filter_opt
{
using autoware_point_types::PointXYZI;
using point_cloud_msg_wrapper::PointCloud2Modifier;
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using PointCloud2ConstPtr = sensor_msgs::msg::PointCloud2::ConstSharedPtr;

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudConstPtr = PointCloud::ConstPtr;

using PointIndices = pcl_msgs::msg::PointIndices;
using PointIndicesPtr = PointIndices::SharedPtr;
using PointIndicesConstPtr = PointIndices::ConstSharedPtr;

using ModelCoefficients = pcl_msgs::msg::ModelCoefficients;
using ModelCoefficientsPtr = ModelCoefficients::SharedPtr;
using ModelCoefficientsConstPtr = ModelCoefficients::ConstSharedPtr;

using IndicesPtr = pcl::IndicesPtr;
using IndicesConstPtr = pcl::IndicesConstPtr;


struct CropBoxParam
  {
    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float min_z;
    float max_z;
    bool negative{false};
  } pram_trans;


void filter(
    const PointCloud2ConstPtr & input, [[maybe_unused]] const IndicesPtr & indices,
    PointCloud2 & output, const CropBoxParam & pram_trans);

}  // namespace pointcloud_preprocessor
#endif  // POINTCLOUD_PREPROCESSOR__OUTLIER_FILTER__CROP_BOX_FILTER_OPT_HPP_

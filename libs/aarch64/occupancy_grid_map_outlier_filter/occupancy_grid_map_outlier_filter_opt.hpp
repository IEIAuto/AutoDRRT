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

#ifndef OCCUPANCY_GRID_MAP_OUTLIER_FILTER_OPT__OCCUPANCY_GRID_MAP_OUTLIER_FILTER_NODELET_HPP_
#define OCCUPANCY_GRID_MAP_OUTLIER_FILTER_OPT__OCCUPANCY_GRID_MAP_OUTLIER_FILTER_NODELET_HPP_
#include <pcl/common/impl/common.hpp>
#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/header.hpp>
// #include <pcl/search/impl/kdtree_omp.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include <memory>
#include <string>
namespace occupancy_grid_map_outlier_filter_opt
{
 pcl::search::Search<pcl::PointXY>::Ptr create_opt_kdtree();

}  // namespace pointcloud_preprocessor
#endif  // OCCUPANCY_GRID_MAP_OUTLIER_FILTER_OPT__OCCUPANCY_GRID_MAP_OUTLIER_FILTER_NODELET_HPP_

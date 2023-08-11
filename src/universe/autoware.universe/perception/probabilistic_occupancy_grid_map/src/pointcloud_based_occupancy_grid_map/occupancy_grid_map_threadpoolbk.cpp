// Copyright 2021 Tier IV, Inc.
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
/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, 2013, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Eitan Marder-Eppstein
 *         David V. Lu!!
 *********************************************************************/

#include "pointcloud_based_occupancy_grid_map/occupancy_grid_map.hpp"

#include "cost_value.hpp"

#include <pcl_ros/transforms.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <sensor_msgs/point_cloud2_iterator.hpp>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#else
#include <tf2_eigen/tf2_eigen.hpp>

#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#endif
#include<omp.h>
#include <algorithm>
#include <pcl/point_types.h>
namespace
{
void transformPointcloud(
  const sensor_msgs::msg::PointCloud2 & input, const geometry_msgs::msg::Pose & pose,
  sensor_msgs::msg::PointCloud2 & output)
{
  const auto transform = tier4_autoware_utils::pose2transform(pose);
  Eigen::Matrix4f tf_matrix = tf2::transformToEigen(transform).matrix().cast<float>();

  pcl_ros::transformPointCloud(tf_matrix, input, output);
  output.header.stamp = input.header.stamp;
  output.header.frame_id = "";
}
}  // namespace

namespace costmap_2d
{
using sensor_msgs::PointCloud2ConstIterator;

OccupancyGridMap::OccupancyGridMap(
  const unsigned int cells_size_x, const unsigned int cells_size_y, const float resolution)
: Costmap2D(cells_size_x, cells_size_y, resolution, 0.f, 0.f, occupancy_cost_value::NO_INFORMATION)
{
  using tier4_autoware_utils::DebugPublisher;
  using tier4_autoware_utils::StopWatch;
  stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
  stop_watch_ptr_->tic("processing_time_debuger");
  stop_watch_ptr_->tic("small_time_debuger");
  int num_threads = 10;
  omp_set_num_threads(num_threads);
  
}

bool OccupancyGridMap::worldToMap(double wx, double wy, unsigned int & mx, unsigned int & my) const
{
  if (wx < origin_x_ || wy < origin_y_) {
    return false;
  }

  mx = static_cast<int>(std::floor((wx - origin_x_) / resolution_));
  my = static_cast<int>(std::floor((wy - origin_y_) / resolution_));

  if (mx < size_x_ && my < size_y_) {
    return true;
  }

  return false;
}

void OccupancyGridMap::updateOrigin(double new_origin_x, double new_origin_y)
{
  // project the new origin into the grid
  int cell_ox{static_cast<int>(std::floor((new_origin_x - origin_x_) / resolution_))};
  int cell_oy{static_cast<int>(std::floor((new_origin_y - origin_y_) / resolution_))};

  // compute the associated world coordinates for the origin cell
  // because we want to keep things grid-aligned
  double new_grid_ox{origin_x_ + cell_ox * resolution_};
  double new_grid_oy{origin_y_ + cell_oy * resolution_};

  // To save casting from unsigned int to int a bunch of times
  int size_x{static_cast<int>(size_x_)};
  int size_y{static_cast<int>(size_y_)};

  // we need to compute the overlap of the new and existing windows
  int lower_left_x{std::min(std::max(cell_ox, 0), size_x)};
  int lower_left_y{std::min(std::max(cell_oy, 0), size_y)};
  int upper_right_x{std::min(std::max(cell_ox + size_x, 0), size_x)};
  int upper_right_y{std::min(std::max(cell_oy + size_y, 0), size_y)};

  unsigned int cell_size_x = upper_right_x - lower_left_x;
  unsigned int cell_size_y = upper_right_y - lower_left_y;

  // we need a map to store the obstacles in the window temporarily
  unsigned char * local_map = new unsigned char[cell_size_x * cell_size_y];

  // copy the local window in the costmap to the local map
  copyMapRegion(
    costmap_, lower_left_x, lower_left_y, size_x_, local_map, 0, 0, cell_size_x, cell_size_x,
    cell_size_y);

  // now we'll set the costmap to be completely unknown if we track unknown space
  resetMaps();

  // update the origin with the appropriate world coordinates
  origin_x_ = new_grid_ox;
  origin_y_ = new_grid_oy;

  // compute the starting cell location for copying data back in
  int start_x{lower_left_x - cell_ox};
  int start_y{lower_left_y - cell_oy};

  // now we want to copy the overlapping information back into the map, but in its new location
  copyMapRegion(
    local_map, 0, 0, cell_size_x, costmap_, start_x, start_y, size_x_, cell_size_x, cell_size_y);

  // make sure to clean up
  delete[] local_map;
}
// Create angle bins
  struct BinInfo
  {
    BinInfo() = default;
    BinInfo(const double _range, const double _wx, const double _wy)
    : range(_range), wx(_wx), wy(_wy)
    {
    }
    double range;
    double wx;
    double wy;
  };
void raw_point_classifer( const PointCloud2 & raw_pointcloud,  PointCloud2 & trans_raw_pointcloud,std::vector</*angle bin*/ std::vector<BinInfo>> & raw_pointcloud_angle_bins, const double &min_angle, const double &angle_increment , int angle_bin_num,int max, int min)
{
  
  // RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"raw_point_classifer come in");
   for (PointCloud2ConstIterator<float> iter_x(raw_pointcloud, "x"), iter_y(raw_pointcloud, "y"),
       iter_wx(trans_raw_pointcloud, "x"), iter_wy(trans_raw_pointcloud, "y");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_wx, ++iter_wy) {
        // itertime = itertime + stop_watch_ptr_->toc("small_time_debuger",true);
    const double angle = atan2(*iter_y, *iter_x);
    const int angle_bin_index = (angle - min_angle) / angle_increment;
    // angletime = angletime + stop_watch_ptr_->toc("small_time_debuger",true);
    if(angle_bin_index >= min && angle_bin_index < max)
    {
      raw_pointcloud_angle_bins.at(angle_bin_index)
      .emplace_back(BinInfo(std::hypot(*iter_y, *iter_x), *iter_wx, *iter_wy));
    }
    
      // vectortime = vectortime + stop_watch_ptr_->toc("small_time_debuger",true);
  }
  // --OccupancyGridMap::thread_pools.busyBus;
}
void OccupancyGridMap::updateWithPointCloud(
  const PointCloud2 & raw_pointcloud, const PointCloud2 & obstacle_pointcloud,
  const Pose & robot_pose)
{
  RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"start time is %f",stop_watch_ptr_->toc("processing_time_debuger", true));
  constexpr double min_angle = tier4_autoware_utils::deg2rad(-180.0);
  constexpr double max_angle = tier4_autoware_utils::deg2rad(180.0);
  constexpr double angle_increment = tier4_autoware_utils::deg2rad(1);
  const size_t angle_bin_size = ((max_angle - min_angle) / angle_increment) + size_t(1 /*margin*/);

  // Transform to map frame
  PointCloud2 trans_raw_pointcloud, trans_obstacle_pointcloud;
  transformPointcloud(raw_pointcloud, robot_pose, trans_raw_pointcloud);
  transformPointcloud(obstacle_pointcloud, robot_pose, trans_obstacle_pointcloud);
  RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"transformPointcloud time is %f,sizebin is %ld",stop_watch_ptr_->toc("processing_time_debuger", true),angle_bin_size);
  // // Create angle bins
  // struct BinInfo
  // {
  //   BinInfo() = default;
  //   BinInfo(const double _range, const double _wx, const double _wy)
  //   : range(_range), wx(_wx), wy(_wy)
  //   {
  //   }
  //   double range;
  //   double wx;
  //   double wy;
  // };
  std::vector</*angle bin*/ std::vector<BinInfo>> obstacle_pointcloud_angle_bins;
  std::vector</*angle bin*/ std::vector<BinInfo>> raw_pointcloud_angle_bins;
  obstacle_pointcloud_angle_bins.resize(angle_bin_size);
  raw_pointcloud_angle_bins.resize(angle_bin_size);
  RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"resize time is %f",stop_watch_ptr_->toc("processing_time_debuger", true));
  // stop_watch_ptr_->toc("small_time_debuger",true);
  // double itertime = 0;
  // double angletime = 0;
  // double vectortime = 0;
  // int count = 0;

  // //---------------------origin
  // for (PointCloud2ConstIterator<float> iter_x(raw_pointcloud, "x"), iter_y(raw_pointcloud, "y"),
  //      iter_wx(trans_raw_pointcloud, "x"), iter_wy(trans_raw_pointcloud, "y");
  //      iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_wx, ++iter_wy) {
  //       // itertime = itertime + stop_watch_ptr_->toc("small_time_debuger",true);
  //   const double angle = atan2(*iter_y, *iter_x);
  //   const int angle_bin_index = (angle - min_angle) / angle_increment;
  //   // angletime = angletime + stop_watch_ptr_->toc("small_time_debuger",true);
  //   raw_pointcloud_angle_bins.at(angle_bin_index)
  //     .emplace_back(BinInfo(std::hypot(*iter_y, *iter_x), *iter_wx, *iter_wy));
  //     // vectortime = vectortime + stop_watch_ptr_->toc("small_time_debuger",true);
  // }

  //---------------------threadpool
  for(int i =0 ; i < 4; i++)
  {
      int step = angle_bin_size / 4;
      int min = step*i;
      int max = step*(i+1);
      thread_pools.submit(&raw_point_classifer,raw_pointcloud, trans_raw_pointcloud, raw_pointcloud_angle_bins,min_angle,angle_increment, i,max,min);
      // ++thread_pools.busyBus;
  }







 RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"for raw_pointcloud time is %f",stop_watch_ptr_->toc("processing_time_debuger", true));
  // for (PointCloud2ConstIterator<float> iter_x(raw_pointcloud, "x"), iter_y(raw_pointcloud, "y"),
  //      iter_wx(trans_raw_pointcloud, "x"), iter_wy(trans_raw_pointcloud, "y");
  //      iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_wx, ++iter_wy) {
  //       // itertime = itertime + stop_watch_ptr_->toc("small_time_debuger",true);
  //   const double angle = atan2(*iter_y, *iter_x);
  //   const int angle_bin_index = (angle - min_angle) / angle_increment;
  // }

  // // RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"for itertime angletime vectortime time is %f,%f,%f,%d",itertime,angletime,vectortime,count);
  // RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"for iter_tan time is %f",stop_watch_ptr_->toc("processing_time_debuger", true));

//---------------------openMP
//  获取点云数据的迭代器
// const auto& raw_pointcloud_data = raw_pointcloud.data;
// const auto& trans_raw_pointcloud_data = trans_raw_pointcloud.data;
// const auto point_step = raw_pointcloud.point_step;
// std::vector</*angle bin*/ std::vector<BinInfo>> raw_pointcloud_angle_bins_omp;
// raw_pointcloud_angle_bins_omp.resize(angle_bin_size);
// //  获取点云数据中 x 和 y 坐标的偏移量

// const auto x_offset = 0;
// const auto y_offset = 4;
// const auto trans_x_offset = 0;
// const auto trans_y_offset = 4;

// //  遍历点云数据
// #pragma omp parallel for
// for (size_t i = 0; i < raw_pointcloud_data.size(); i += point_step) {
//   //  获取点云数据的 x 和 y 坐标
//   const float x = *reinterpret_cast<const float*>(&raw_pointcloud_data[i + x_offset]);
//   const float y = *reinterpret_cast<const float*>(&raw_pointcloud_data[i + y_offset]);
//   const float trans_x = *reinterpret_cast<const float*>(&trans_raw_pointcloud_data[i + trans_x_offset]);
//   const float trans_y = *reinterpret_cast<const float*>(&trans_raw_pointcloud_data[i + trans_y_offset]);

//   //  计算点云数据的角度和分配到对应的角度分组中
//   const double angle = atan2(y, x);
//   const int angle_bin_index = (angle - min_angle) / angle_increment;
//   #pragma omp critical
//   raw_pointcloud_angle_bins_omp.at(angle_bin_index)
//     .push_back(BinInfo(std::hypot(y, x), trans_x, trans_y));
// }
// RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"for raw_pointcloud_omp time is %f",stop_watch_ptr_->toc("processing_time_debuger", true));




  for (PointCloud2ConstIterator<float> iter_x(obstacle_pointcloud, "x"),
       iter_y(obstacle_pointcloud, "y"), iter_wx(trans_obstacle_pointcloud, "x"),
       iter_wy(trans_obstacle_pointcloud, "y");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_wx, ++iter_wy) {
    const double angle = atan2(*iter_y, *iter_x);
    int angle_bin_index = (angle - min_angle) / angle_increment;
    obstacle_pointcloud_angle_bins.at(angle_bin_index)
      .emplace_back(BinInfo(std::hypot(*iter_y, *iter_x), *iter_wx, *iter_wy));
  }

  // int cloud_size = raw_pointcloud.width*raw_pointcloud.height;
  // PointCloud2ConstIterator<float> iter_x(obstacle_pointcloud, "x");
  // PointCloud2ConstIterator<float> iter_y(obstacle_pointcloud, "y");
  // PointCloud2ConstIterator<float> iter_wx(trans_obstacle_pointcloud, "x");
  // PointCloud2ConstIterator<float> iter_wy(trans_obstacle_pointcloud, "y");
  //  #pragma omp parallel for
  // for (int i =0; i < cloud_size; i++)
  // {
  //   const double angle = atan2(*(iter_y+i), *(iter_x+i));
  //   int angle_bin_index = (angle - min_angle) / angle_increment;
  //   BinInfo bininfo(std::hypot(*(iter_y+i), *(iter_x+i)), *(iter_wx+i), *(iter_wy+i));
  //   #pragma omp critical
  //   {
  //   obstacle_pointcloud_angle_bins.at(angle_bin_index)
  //     .push_back(bininfo);
  //   }
  // }



  RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"for obstacle_pointcloud time is %f",stop_watch_ptr_->toc("processing_time_debuger", true));
  // Sort by distance
  for (auto & obstacle_pointcloud_angle_bin : obstacle_pointcloud_angle_bins) {
    std::sort(
      obstacle_pointcloud_angle_bin.begin(), obstacle_pointcloud_angle_bin.end(),
      [](auto a, auto b) { return a.range < b.range; });
  }
  for (auto & raw_pointcloud_angle_bin : raw_pointcloud_angle_bins) {
    std::sort(raw_pointcloud_angle_bin.begin(), raw_pointcloud_angle_bin.end(), [](auto a, auto b) {
      return a.range < b.range;
    });
  }
  RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"for obstacle_pointcloud_angle_bin  time is %f",stop_watch_ptr_->toc("processing_time_debuger", true));
  // First step: Initialize cells to the final point with freespace
  constexpr double distance_margin = 1.0;
 
  #pragma omp parallel for
  
  for (size_t bin_index = 0; bin_index < obstacle_pointcloud_angle_bins.size(); ++bin_index) {
    auto & obstacle_pointcloud_angle_bin = obstacle_pointcloud_angle_bins.at(bin_index);
    auto & raw_pointcloud_angle_bin = raw_pointcloud_angle_bins.at(bin_index);
    // std::cout << "omp_get_num_threads" << omp_get_num_threads() << std::endl;
    BinInfo end_distance;
    if (raw_pointcloud_angle_bin.empty() && obstacle_pointcloud_angle_bin.empty()) {
      continue;
    } else if (raw_pointcloud_angle_bin.empty()) {
      end_distance = obstacle_pointcloud_angle_bin.back();
    } else if (obstacle_pointcloud_angle_bin.empty()) {
      end_distance = raw_pointcloud_angle_bin.back();
    } else {
      end_distance = obstacle_pointcloud_angle_bin.back().range + distance_margin <
                         raw_pointcloud_angle_bin.back().range
                       ? raw_pointcloud_angle_bin.back()
                       : obstacle_pointcloud_angle_bin.back();
    }
    raytrace(
      robot_pose.position.x, robot_pose.position.y, end_distance.wx, end_distance.wy,
      occupancy_cost_value::FREE_SPACE);
  }
  RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"Initialize cells  time is %f",stop_watch_ptr_->toc("processing_time_debuger", true));
  // Second step: Add uknown cell
  #pragma omp parallel for
  for (size_t bin_index = 0; bin_index < obstacle_pointcloud_angle_bins.size(); ++bin_index) {
    auto & obstacle_pointcloud_angle_bin = obstacle_pointcloud_angle_bins.at(bin_index);
    auto & raw_pointcloud_angle_bin = raw_pointcloud_angle_bins.at(bin_index);
    auto raw_distance_iter = raw_pointcloud_angle_bin.begin();
    for (size_t dist_index = 0; dist_index < obstacle_pointcloud_angle_bin.size(); ++dist_index) {
      // Calculate next raw point from obstacle point
      while (raw_distance_iter != raw_pointcloud_angle_bin.end()) {
        if (
          raw_distance_iter->range <
          obstacle_pointcloud_angle_bin.at(dist_index).range + distance_margin)
          raw_distance_iter++;
        else
          break;
      }

      // There is no point far than the obstacle point.
      const bool no_freespace_point = (raw_distance_iter == raw_pointcloud_angle_bin.end());

      if (dist_index + 1 == obstacle_pointcloud_angle_bin.size()) {
        const auto & source = obstacle_pointcloud_angle_bin.at(dist_index);
        if (!no_freespace_point) {
          const auto & target = *raw_distance_iter;
          raytrace(
            source.wx, source.wy, target.wx, target.wy, occupancy_cost_value::NO_INFORMATION);
          setCellValue(target.wx, target.wy, occupancy_cost_value::FREE_SPACE);
        }
        continue;
      }

      auto next_obstacle_point_distance = std::abs(
        obstacle_pointcloud_angle_bin.at(dist_index + 1).range -
        obstacle_pointcloud_angle_bin.at(dist_index).range);
      if (next_obstacle_point_distance <= distance_margin) {
        continue;
      } else if (no_freespace_point) {
        const auto & source = obstacle_pointcloud_angle_bin.at(dist_index);
        const auto & target = obstacle_pointcloud_angle_bin.at(dist_index + 1);
        raytrace(source.wx, source.wy, target.wx, target.wy, occupancy_cost_value::NO_INFORMATION);
        continue;
      }

      auto next_raw_distance =
        std::abs(obstacle_pointcloud_angle_bin.at(dist_index).range - raw_distance_iter->range);
      if (next_raw_distance < next_obstacle_point_distance) {
        const auto & source = obstacle_pointcloud_angle_bin.at(dist_index);
        const auto & target = *raw_distance_iter;
        raytrace(source.wx, source.wy, target.wx, target.wy, occupancy_cost_value::NO_INFORMATION);
        setCellValue(target.wx, target.wy, occupancy_cost_value::FREE_SPACE);
        continue;
      } else {
        const auto & source = obstacle_pointcloud_angle_bin.at(dist_index);
        const auto & target = obstacle_pointcloud_angle_bin.at(dist_index + 1);
        raytrace(source.wx, source.wy, target.wx, target.wy, occupancy_cost_value::NO_INFORMATION);
        continue;
      }
    }
  }
  RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"Add uknown cell time is %f",stop_watch_ptr_->toc("processing_time_debuger", true));
  // Third step: Overwrite occupied cell
  #pragma omp parallel for
  for (size_t bin_index = 0; bin_index < obstacle_pointcloud_angle_bins.size(); ++bin_index) {
    auto & obstacle_pointcloud_angle_bin = obstacle_pointcloud_angle_bins.at(bin_index);
    for (size_t dist_index = 0; dist_index < obstacle_pointcloud_angle_bin.size(); ++dist_index) {
      const auto & source = obstacle_pointcloud_angle_bin.at(dist_index);
      setCellValue(source.wx, source.wy, occupancy_cost_value::LETHAL_OBSTACLE);

      if (dist_index + 1 == obstacle_pointcloud_angle_bin.size()) {
        continue;
      }

      auto next_obstacle_point_distance = std::abs(
        obstacle_pointcloud_angle_bin.at(dist_index + 1).range -
        obstacle_pointcloud_angle_bin.at(dist_index).range);
      if (next_obstacle_point_distance <= distance_margin) {
        const auto & source = obstacle_pointcloud_angle_bin.at(dist_index);
        const auto & target = obstacle_pointcloud_angle_bin.at(dist_index + 1);
        raytrace(source.wx, source.wy, target.wx, target.wy, occupancy_cost_value::LETHAL_OBSTACLE);
        continue;
      }
    }
  }
  RCLCPP_INFO(rclcpp::get_logger("occupancy_grid_map_node_updateWithPointCloud_debuger"),"Overwrite occupied cell time is %f",stop_watch_ptr_->toc("processing_time_debuger", true));
}

void OccupancyGridMap::setCellValue(const double wx, const double wy, const unsigned char cost)
{
  MarkCell marker(costmap_, cost);
  unsigned int mx{};
  unsigned int my{};
  if (!worldToMap(wx, wy, mx, my)) {
    RCLCPP_DEBUG(logger_, "Computing map coords failed");
    return;
  }
  const unsigned int index = getIndex(mx, my);
  marker(index);
}

void OccupancyGridMap::raytrace(
  const double source_x, const double source_y, const double target_x, const double target_y,
  const unsigned char cost)
{
  unsigned int x0{};
  unsigned int y0{};
  const double ox{source_x};
  const double oy{source_y};
  if (!worldToMap(ox, oy, x0, y0)) {
    RCLCPP_DEBUG(
      logger_,
      "The origin for the sensor at (%.2f, %.2f) is out of map bounds. So, the costmap cannot "
      "raytrace for it.",
      ox, oy);
    return;
  }

  // we can pre-compute the endpoints of the map outside of the inner loop... we'll need these later
  const double origin_x = origin_x_, origin_y = origin_y_;
  const double map_end_x = origin_x + size_x_ * resolution_;
  const double map_end_y = origin_y + size_y_ * resolution_;

  double wx = target_x;
  double wy = target_y;

  // now we also need to make sure that the endpoint we're ray-tracing
  // to isn't off the costmap and scale if necessary
  const double a = wx - ox;
  const double b = wy - oy;

  // the minimum value to raytrace from is the origin
  if (wx < origin_x) {
    const double t = (origin_x - ox) / a;
    wx = origin_x;
    wy = oy + b * t;
  }
  if (wy < origin_y) {
    const double t = (origin_y - oy) / b;
    wx = ox + a * t;
    wy = origin_y;
  }

  // the maximum value to raytrace to is the end of the map
  if (wx > map_end_x) {
    const double t = (map_end_x - ox) / a;
    wx = map_end_x - .001;
    wy = oy + b * t;
  }
  if (wy > map_end_y) {
    const double t = (map_end_y - oy) / b;
    wx = ox + a * t;
    wy = map_end_y - .001;
  }

  // now that the vector is scaled correctly... we'll get the map coordinates of its endpoint
  unsigned int x1{};
  unsigned int y1{};

  // check for legality just in case
  if (!worldToMap(wx, wy, x1, y1)) {
    return;
  }

  constexpr unsigned int cell_raytrace_range = 10000;  // large number to ignore range threshold
  MarkCell marker(costmap_, cost);
  raytraceLine(marker, x0, y0, x1, y1, cell_raytrace_range);
}

}  // namespace costmap_2d

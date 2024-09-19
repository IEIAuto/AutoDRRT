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

#ifndef POINTCLOUD_BASED_OCCUPANCY_GRID_MAP__OCCUPANCY_GRID_MAP_FIXED_HPP_
#define POINTCLOUD_BASED_OCCUPANCY_GRID_MAP__OCCUPANCY_GRID_MAP_FIXED_HPP_

#include "pointcloud_based_occupancy_grid_map/occupancy_grid_map_base.hpp"

namespace costmap_2d
{
using geometry_msgs::msg::Pose;
using sensor_msgs::msg::PointCloud2;

class OccupancyGridMapFixedBlindSpot : public OccupancyGridMapInterface
{
public:
  OccupancyGridMapFixedBlindSpot(
    const unsigned int cells_size_x, const unsigned int cells_size_y, const float resolution);

  void updateWithPointCloud(
    const PointCloud2 & raw_pointcloud, const PointCloud2 & obstacle_pointcloud,
    const Pose & robot_pose, const Pose & scan_origin) override;

  using OccupancyGridMapInterface::raytrace;
  using OccupancyGridMapInterface::setCellValue;
  using OccupancyGridMapInterface::updateOrigin;

  void initRosParam(rclcpp::Node & node) override;

private:
  double distance_margin_;
};

}  // namespace costmap_2d

#endif  // POINTCLOUD_BASED_OCCUPANCY_GRID_MAP__OCCUPANCY_GRID_MAP_FIXED_HPP_

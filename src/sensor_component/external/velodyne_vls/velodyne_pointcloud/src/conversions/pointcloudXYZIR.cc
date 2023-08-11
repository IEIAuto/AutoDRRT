/*
 * Copyright 2020 Tier IV, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <velodyne_pointcloud/pointcloudXYZIR.h>

namespace velodyne_pointcloud
{
void PointcloudXYZIR::addPoint(
  const float & x, const float & y, const float & z,
  const uint8_t & return_type, const uint16_t & ring, const uint16_t & azimuth,
  const float & distance, const float & intensity, const double & time_stamp)
{
  (void)azimuth;
  (void)distance;
  (void)time_stamp;
  (void)return_type;

  velodyne_pointcloud::PointXYZIR point;
  point.x = x;
  point.y = y;
  point.z = z;
  point.intensity = intensity;
  point.ring = ring;

  pc->points.push_back(point);
  ++pc->width;
}
}  // namespace velodyne_pointcloud

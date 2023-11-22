// Copyright 2022 The Autoware Contributors
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

#ifndef MAP_HEIGHT_FITTER__MAP_HEIGHT_FITTER_HPP_
#define MAP_HEIGHT_FITTER__MAP_HEIGHT_FITTER_HPP_

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/point.hpp>

#include <memory>
#include <string>

namespace map_height_fitter
{

using geometry_msgs::msg::Point;

class MapHeightFitter final
{
public:
  MapHeightFitter(rclcpp::Node * node);
  ~MapHeightFitter();
  Point fit(const Point & position, const std::string & frame);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace map_height_fitter

#endif  // MAP_HEIGHT_FITTER__MAP_HEIGHT_FITTER_HPP_

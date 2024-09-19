// Copyright 2021 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NODE_V2X_HPP_
#define NODE_V2X_HPP_

#include <nav_msgs/msg/odometry.hpp>
#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>

#include "rclcpp/rclcpp.hpp"

#include <iostream>
#include <sstream>

#include <set>
#include <map>
#include <vector>
#include <string>

#include <cmath>
#include <numeric>
#include <algorithm>
#include <utility>
#include <variant>
#include <any>

#include <typeinfo>
#include <cxxabi.h>
#include <cstdint>

#include <memory>
#include <functional>

#include <mutex>

namespace node_v2x
{

double getime();

class NodeV2X1 : public rclcpp::Node
{

public:

  explicit NodeV2X1(const rclcpp::NodeOptions & options);

private:

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_pose; 

  rclcpp::Subscription<autoware_auto_perception_msgs::msg::PredictedObjects>::SharedPtr sub_object; 

};


class NodeV2X2 : public rclcpp::Node
{

public:

  explicit NodeV2X2(const rclcpp::NodeOptions & options);

private:

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_pose;

  rclcpp::Publisher<autoware_auto_perception_msgs::msg::PredictedObjects>::SharedPtr pub_object;

};

}  // namespace node_V2X

#endif  // NODE_V2X_HPP_

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

#ifndef DEMO_NODE_HPP_
#define DEMO_NODE_HPP_

#include <nav_msgs/msg/odometry.hpp>
#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/header.hpp"
#include <iostream>
#include <sstream>
#include <cstdlib>

#include <vector>
#include <set>
#include <string>

#include <numeric>
#include <cmath>
#include <algorithm>
#include "dma_adapter.hpp"

namespace dma_transfer
{

class DemoNode1 : public rclcpp::Node
{

public:

  explicit DemoNode1(const rclcpp::NodeOptions & options);

private:

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_string; 

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_pose;

  rclcpp::Subscription<autoware_auto_perception_msgs::msg::PredictedObjects>::SharedPtr sub_object; 

  //DmaAdapter<std_msgs::msg::String> dma_trans;
  DmaAdapter<sensor_msgs::msg::Image> dma_trans;

  rclcpp::TimerBase::SharedPtr timer_;
  int count;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_test_;

};


class DemoNode2 : public rclcpp::Node
{

public:

  explicit DemoNode2(const rclcpp::NodeOptions & options);

private:

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_string;  

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_pose;  

  rclcpp::Subscription<autoware_auto_perception_msgs::msg::PredictedObjects>::SharedPtr sub_object; 
  
  //DmaAdapter<std_msgs::msg::String> dma_trans;
  DmaAdapter<sensor_msgs::msg::Image> dma_trans;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_test_;
  
  //rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_test_;

};


}  // namespace dma_transfer

#endif  // DEMO_NODE_HPP_

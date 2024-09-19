// Copyright 2021 TIER IV, Inc.
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

#ifndef VELOCITY_HPP_
#define VELOCITY_HPP_

#include <rclcpp/rclcpp.hpp>
#include <tier4_api_utils/tier4_api_utils.hpp>

#include <tier4_external_api_msgs/srv/pause_driving.hpp>
#include <tier4_external_api_msgs/srv/set_velocity_limit.hpp>
#include <tier4_planning_msgs/msg/velocity_limit.hpp>

namespace internal_api
{
class Velocity : public rclcpp::Node
{
public:
  explicit Velocity(const rclcpp::NodeOptions & options);

private:
  using PauseDriving = tier4_external_api_msgs::srv::PauseDriving;
  using SetVelocityLimit = tier4_external_api_msgs::srv::SetVelocityLimit;
  using VelocityLimit = tier4_planning_msgs::msg::VelocityLimit;

  // ros interface
  tier4_api_utils::Service<PauseDriving>::SharedPtr srv_pause_;
  tier4_api_utils::Service<SetVelocityLimit>::SharedPtr srv_velocity_;
  rclcpp::Publisher<VelocityLimit>::SharedPtr pub_api_velocity_;
  rclcpp::Publisher<VelocityLimit>::SharedPtr pub_planning_velocity_;
  rclcpp::Subscription<VelocityLimit>::SharedPtr sub_planning_velocity_;

  // class constants
  static constexpr double kVelocityEpsilon = 1e-5;

  // class state
  bool is_ready_;
  double velocity_limit_;

  // ros callback
  void setPauseDriving(
    const tier4_external_api_msgs::srv::PauseDriving::Request::SharedPtr request,
    const tier4_external_api_msgs::srv::PauseDriving::Response::SharedPtr response);
  void setVelocityLimit(
    const tier4_external_api_msgs::srv::SetVelocityLimit::Request::SharedPtr request,
    const tier4_external_api_msgs::srv::SetVelocityLimit::Response::SharedPtr response);
  void onVelocityLimit(const tier4_planning_msgs::msg::VelocityLimit::SharedPtr msg);

  // class method
  void publishApiVelocity(double velocity);
  void publishPlanningVelocity(double velocity);
};

}  // namespace internal_api

#endif  // VELOCITY_HPP_

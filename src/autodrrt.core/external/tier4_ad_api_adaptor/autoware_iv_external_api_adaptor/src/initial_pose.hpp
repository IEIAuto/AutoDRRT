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

#ifndef INITIAL_POSE_HPP_
#define INITIAL_POSE_HPP_

#include <component_interface_specs/localization.hpp>
#include <component_interface_utils/rclcpp.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tier4_api_utils/tier4_api_utils.hpp>

#include <tier4_external_api_msgs/srv/initialize_pose.hpp>
#include <tier4_external_api_msgs/srv/initialize_pose_auto.hpp>

namespace external_api
{

class InitialPose : public rclcpp::Node
{
public:
  explicit InitialPose(const rclcpp::NodeOptions & options);

private:
  using InitializePose = tier4_external_api_msgs::srv::InitializePose;
  using InitializePoseAuto = tier4_external_api_msgs::srv::InitializePoseAuto;

  // ros interface
  rclcpp::CallbackGroup::SharedPtr group_;
  tier4_api_utils::Service<InitializePose>::SharedPtr srv_set_initialize_pose_;
  tier4_api_utils::Service<InitializePoseAuto>::SharedPtr srv_set_initialize_pose_auto_;
  component_interface_utils::Client<localization_interface::Initialize>::SharedPtr
    cli_localization_initialize_;

  // ros callback
  void setInitializePose(
    const tier4_external_api_msgs::srv::InitializePose::Request::SharedPtr request,
    const tier4_external_api_msgs::srv::InitializePose::Response::SharedPtr response);
  void setInitializePoseAuto(
    const tier4_external_api_msgs::srv::InitializePoseAuto::Request::SharedPtr request,
    const tier4_external_api_msgs::srv::InitializePoseAuto::Response::SharedPtr response);
};

}  // namespace external_api

#endif  // INITIAL_POSE_HPP_

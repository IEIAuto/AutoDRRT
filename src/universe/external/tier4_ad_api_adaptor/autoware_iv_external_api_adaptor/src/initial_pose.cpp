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

#include "initial_pose.hpp"

#include "converter/response_status.hpp"

#include <array>
#include <memory>

namespace external_api
{

constexpr double initial_pose_timeout = 300;

// clang-format off
const std::array<double, 36> particle_covariance =
{
  1.00, 0.00, 0.00, 0.00, 0.00,  0.00,
  0.00, 1.00, 0.00, 0.00, 0.00,  0.00,
  0.00, 0.00, 0.01, 0.00, 0.00,  0.00,
  0.00, 0.00, 0.00, 0.01, 0.00,  0.00,
  0.00, 0.00, 0.00, 0.00, 0.01,  0.00,
  0.00, 0.00, 0.00, 0.00, 0.00, 10.00,
};
// clang-format on

InitialPose::InitialPose(const rclcpp::NodeOptions & options)
: Node("external_api_initial_pose", options)
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  tier4_api_utils::ServiceProxyNodeInterface proxy(this);

  group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  srv_set_initialize_pose_ = proxy.create_service<InitializePose>(
    "/api/external/set/initialize_pose", std::bind(&InitialPose::setInitializePose, this, _1, _2),
    rmw_qos_profile_services_default, group_);
  srv_set_initialize_pose_auto_ = proxy.create_service<InitializePoseAuto>(
    "/api/external/set/initialize_pose_auto",
    std::bind(&InitialPose::setInitializePoseAuto, this, _1, _2), rmw_qos_profile_services_default,
    group_);

  const auto adaptor = component_interface_utils::NodeAdaptor(this);
  adaptor.init_cli(cli_localization_initialize_);
}

void InitialPose::setInitializePose(
  const tier4_external_api_msgs::srv::InitializePose::Request::SharedPtr request,
  const tier4_external_api_msgs::srv::InitializePose::Response::SharedPtr response)
{
  const auto req = std::make_shared<localization_interface::Initialize::Service::Request>();
  req->pose.push_back(request->pose);
  req->pose.back().pose.covariance = particle_covariance;

  try {
    const auto res = cli_localization_initialize_->call(req, initial_pose_timeout);
    response->status = converter::convert(res->status);
  } catch (const component_interface_utils::ServiceException & error) {
    response->status = tier4_api_utils::response_error(error.what());
  }
}

void InitialPose::setInitializePoseAuto(
  const tier4_external_api_msgs::srv::InitializePoseAuto::Request::SharedPtr,
  const tier4_external_api_msgs::srv::InitializePoseAuto::Response::SharedPtr response)
{
  const auto req = std::make_shared<localization_interface::Initialize::Service::Request>();

  try {
    const auto res = cli_localization_initialize_->call(req, initial_pose_timeout);
    response->status = converter::convert(res->status);
  } catch (const component_interface_utils::ServiceException & error) {
    response->status = tier4_api_utils::response_error(error.what());
  }
}

}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::InitialPose)

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

#include "engage.hpp"

#include <memory>

namespace external_api
{

Engage::Engage(const rclcpp::NodeOptions & options) : Node("external_api_engage", options)
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  tier4_api_utils::ServiceProxyNodeInterface proxy(this);

  group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  srv_engage_ = proxy.create_service<tier4_external_api_msgs::srv::Engage>(
    "/api/external/set/engage", std::bind(&Engage::setEngage, this, _1, _2),
    rmw_qos_profile_services_default, group_);
  cli_engage_ = proxy.create_client<tier4_external_api_msgs::srv::Engage>(
    "/api/autoware/set/engage", rmw_qos_profile_services_default);
  cli_set_operator_ = proxy.create_client<tier4_external_api_msgs::srv::SetOperator>(
    "/api/autoware/set/operator", rmw_qos_profile_services_default);
  pub_engage_status_ = create_publisher<tier4_external_api_msgs::msg::EngageStatus>(
    "/api/external/get/engage", rclcpp::QoS(1));
  sub_engage_status_ = create_subscription<autoware_auto_vehicle_msgs::msg::Engage>(
    "/api/autoware/get/engage", rclcpp::QoS(1), std::bind(&Engage::onEngageStatus, this, _1));
  sub_autoware_state_ = create_subscription<autoware_auto_system_msgs::msg::AutowareState>(
    "/autoware/state", rclcpp::QoS(1), std::bind(&Engage::onAutowareState, this, _1));

  waiting_for_engage_ = false;
  driving_ = false;
  auto_operator_change_ = declare_parameter("auto_operator_change", false);
}

void Engage::setEngage(
  const tier4_external_api_msgs::srv::Engage::Request::SharedPtr request,
  const tier4_external_api_msgs::srv::Engage::Response::SharedPtr response)
{
  if (request->engage && driving_) {
    response->status = tier4_api_utils::response_ignored("It is already engaged.");
    return;
  }

  if (request->engage && !waiting_for_engage_) {
    response->status = tier4_api_utils::response_error("It is not ready to engage.");
    return;
  }

  if (auto_operator_change_) {
    using tier4_external_api_msgs::msg::Operator;
    auto req = std::make_shared<tier4_external_api_msgs::srv::SetOperator::Request>();
    req->mode.mode = request->engage ? Operator::AUTONOMOUS : Operator::DRIVER;
    auto [status, resp] = cli_set_operator_->call(req);
    if (!tier4_api_utils::is_success(status)) {
      response->status = status;
      return;
    }
  }

  auto [status, resp] = cli_engage_->call(request);
  if (!tier4_api_utils::is_success(status)) {
    response->status = status;
    return;
  }
  response->status = resp->status;
}

void Engage::onEngageStatus(const autoware_auto_vehicle_msgs::msg::Engage::SharedPtr message)
{
  auto msg = tier4_external_api_msgs::build<tier4_external_api_msgs::msg::EngageStatus>()
               .stamp(message->stamp)
               .engage(message->engage);
  pub_engage_status_->publish(msg);
}

void Engage::onAutowareState(const autoware_auto_system_msgs::msg::AutowareState::SharedPtr message)
{
  using autoware_auto_system_msgs::msg::AutowareState;
  waiting_for_engage_ = (message->state == AutowareState::WAITING_FOR_ENGAGE);
  driving_ = (message->state == AutowareState::DRIVING);
}

}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::Engage)

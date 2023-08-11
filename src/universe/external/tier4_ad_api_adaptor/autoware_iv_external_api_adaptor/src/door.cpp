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

#include "door.hpp"

namespace external_api
{

Door::Door(const rclcpp::NodeOptions & options) : Node("external_api_door", options)
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  tier4_api_utils::ServiceProxyNodeInterface proxy(this);

  group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  srv_ = proxy.create_service<tier4_external_api_msgs::srv::SetDoor>(
    "/api/external/set/door", std::bind(&Door::setDoor, this, _1, _2),
    rmw_qos_profile_services_default, group_);
  cli_ = proxy.create_client<tier4_external_api_msgs::srv::SetDoor>(
    "/api/vehicle/set/door", rmw_qos_profile_services_default);
  pub_door_status_ = create_publisher<tier4_external_api_msgs::msg::DoorStatus>(
    "/api/external/get/door", rclcpp::QoS(1));
  sub_door_status_ = create_subscription<tier4_api_msgs::msg::DoorStatus>(
    "/vehicle/status/door_status", rclcpp::QoS(1), std::bind(&Door::getDoorStatus, this, _1));
}

void Door::setDoor(
  const tier4_external_api_msgs::srv::SetDoor::Request::SharedPtr request,
  const tier4_external_api_msgs::srv::SetDoor::Response::SharedPtr response)
{
  auto [status, resp] = cli_->call(request);
  if (!tier4_api_utils::is_success(status)) {
    response->status = status;
    return;
  }
  response->status = resp->status;
}

void Door::getDoorStatus(const tier4_api_msgs::msg::DoorStatus::SharedPtr message)
{
  tier4_external_api_msgs::msg::DoorStatus msg;
  msg.stamp = now();
  msg.status = message->status;
  pub_door_status_->publish(msg);
}

}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::Door)

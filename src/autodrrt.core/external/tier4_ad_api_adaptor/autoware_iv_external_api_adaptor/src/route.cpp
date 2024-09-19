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

#include "route.hpp"

#include "converter/response_status.hpp"
#include "converter/routing.hpp"

#include <memory>

namespace external_api
{

Route::Route(const rclcpp::NodeOptions & options) : Node("external_api_route", options)
{
  using std::placeholders::_1;
  using std::placeholders::_2;

  // external
  {
    tier4_api_utils::ServiceProxyNodeInterface proxy(this);
    group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    srv_set_route_ = proxy.create_service<tier4_external_api_msgs::srv::SetRoute>(
      "/api/external/set/route", std::bind(&Route::setRoute, this, _1, _2),
      rmw_qos_profile_services_default, group_);
    srv_clear_route_ = proxy.create_service<tier4_external_api_msgs::srv::ClearRoute>(
      "/api/external/set/clear_route", std::bind(&Route::clearRoute, this, _1, _2),
      rmw_qos_profile_services_default, group_);
    pub_get_route_ = create_publisher<tier4_external_api_msgs::msg::Route>(
      "/api/external/get/route", rclcpp::QoS(1).transient_local());
  }

  // adapi
  {
    const auto adaptor = component_interface_utils::NodeAdaptor(this);
    adaptor.init_cli(cli_clear_route_);
    adaptor.init_cli(cli_set_route_);
    adaptor.init_sub(sub_get_route_, this, &Route::onRoute);
  }
}

void Route::setRoute(
  const tier4_external_api_msgs::srv::SetRoute::Request::SharedPtr request,
  const tier4_external_api_msgs::srv::SetRoute::Response::SharedPtr response)
{
  // clear to overwrite
  {
    const auto req = std::make_shared<tier4_external_api_msgs::srv::ClearRoute::Request>();
    const auto res = std::make_shared<tier4_external_api_msgs::srv::ClearRoute::Response>();
    clearRoute(req, res);
  }

  try {
    const auto req = std::make_shared<autoware_ad_api::routing::SetRoute::Service::Request>();
    *req = converter::convert(*request);
    const auto res = cli_set_route_->call(req);
    response->status = converter::convert(res->status);
  } catch (const component_interface_utils::ServiceException & error) {
    response->status = tier4_api_utils::response_error(error.what());
  }
}

void Route::clearRoute(
  const tier4_external_api_msgs::srv::ClearRoute::Request::SharedPtr,
  const tier4_external_api_msgs::srv::ClearRoute::Response::SharedPtr response)
{
  try {
    const auto req = std::make_shared<autoware_ad_api::routing::ClearRoute::Service::Request>();
    const auto res = cli_clear_route_->call(req);
    response->status = converter::convert(res->status);
  } catch (const component_interface_utils::ServiceException & error) {
    response->status = tier4_api_utils::response_error(error.what());
  }
}

void Route::onRoute(const autoware_ad_api::routing::Route::Message::ConstSharedPtr message)
{
  if (!message->data.empty()) {
    pub_get_route_->publish(converter::convert(*message));
  }
}

}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::Route)

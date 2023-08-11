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

#include "map.hpp"

namespace external_api
{

Map::Map(const rclcpp::NodeOptions & options) : Node("external_api_map", options)
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  tier4_api_utils::ServiceProxyNodeInterface proxy(this);

  group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  srv_lanelet_xml_ = proxy.create_service<tier4_external_api_msgs::srv::GetTextFile>(
    "/api/external/get/map/lanelet/xml", std::bind(&Map::getLaneletXml, this, _1, _2),
    rmw_qos_profile_services_default, group_);
  cli_lanelet_xml_ = proxy.create_client<tier4_external_api_msgs::srv::GetTextFile>(
    "/api/autoware/get/map/lanelet/xml", rmw_qos_profile_services_default);

  pub_map_info_ = create_publisher<tier4_external_api_msgs::msg::MapHash>(
    "/api/external/get/map/info/hash", rclcpp::QoS(1).transient_local());
  sub_map_info_ = create_subscription<tier4_external_api_msgs::msg::MapHash>(
    "/api/autoware/get/map/info/hash", rclcpp::QoS(1).transient_local(),
    std::bind(&Map::getMapHash, this, _1));
}

void Map::getMapHash(const tier4_external_api_msgs::msg::MapHash::SharedPtr message)
{
  pub_map_info_->publish(*message);
}

void Map::getLaneletXml(
  const tier4_external_api_msgs::srv::GetTextFile::Request::SharedPtr request,
  const tier4_external_api_msgs::srv::GetTextFile::Response::SharedPtr response)
{
  auto [status, resp] = cli_lanelet_xml_->call(request);
  if (!tier4_api_utils::is_success(status)) {
    response->status = status;
    return;
  }
  response->file = resp->file;
  response->status = resp->status;
}

}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::Map)

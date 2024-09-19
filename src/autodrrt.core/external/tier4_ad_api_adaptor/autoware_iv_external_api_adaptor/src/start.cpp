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

#include "start.hpp"

namespace external_api
{

Start::Start(const rclcpp::NodeOptions & options) : Node("external_api_start", options)
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  tier4_api_utils::ServiceProxyNodeInterface proxy(this);

  srv_set_request_start_ = proxy.create_service<std_srvs::srv::Trigger>(
    "/api/autoware/set/start_request", std::bind(&Start::setRequestStart, this, _1, _2));

  sub_get_operator_ = create_subscription<tier4_external_api_msgs::msg::Operator>(
    "/api/external/get/operator", rclcpp::QoS(1), std::bind(&Start::getOperator, this, _1));
}

void Start::setRequestStart(
  const std_srvs::srv::Trigger::Request::SharedPtr,
  const std_srvs::srv::Trigger::Response::SharedPtr response)
{
  using namespace std::literals::chrono_literals;

  if (operator_ && operator_->mode == Operator::AUTONOMOUS) {
    rclcpp::Rate rate(5000ms);
    rate.sleep();
  }
  response->success = true;
}

void Start::getOperator(const tier4_external_api_msgs::msg::Operator::ConstSharedPtr message)
{
  operator_ = message;
}

}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::Start)

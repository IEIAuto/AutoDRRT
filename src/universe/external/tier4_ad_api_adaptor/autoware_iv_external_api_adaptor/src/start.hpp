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

#ifndef START_HPP_
#define START_HPP_

#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tier4_api_utils/tier4_api_utils.hpp"

#include "tier4_external_api_msgs/msg/operator.hpp"

namespace external_api
{

class Start : public rclcpp::Node
{
public:
  explicit Start(const rclcpp::NodeOptions & options);

private:
  using Trigger = std_srvs::srv::Trigger;
  using Operator = tier4_external_api_msgs::msg::Operator;

  // ros interface
  tier4_api_utils::Service<Trigger>::SharedPtr srv_set_request_start_;
  rclcpp::Subscription<Operator>::SharedPtr sub_get_operator_;

  // ros callback
  void setRequestStart(
    const std_srvs::srv::Trigger::Request::SharedPtr request,
    const std_srvs::srv::Trigger::Response::SharedPtr response);
  void getOperator(const tier4_external_api_msgs::msg::Operator::ConstSharedPtr message);

  // class state
  tier4_external_api_msgs::msg::Operator::ConstSharedPtr operator_;
};

}  // namespace external_api

#endif  // START_HPP_

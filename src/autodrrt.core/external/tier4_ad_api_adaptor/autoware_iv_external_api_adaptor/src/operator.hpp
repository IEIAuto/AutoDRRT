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

#ifndef OPERATOR_HPP_
#define OPERATOR_HPP_

#include "rclcpp/rclcpp.hpp"
#include "tier4_api_utils/tier4_api_utils.hpp"

#include "tier4_external_api_msgs/msg/observer.hpp"
#include "tier4_external_api_msgs/msg/operator.hpp"
#include "tier4_external_api_msgs/srv/set_observer.hpp"
#include "tier4_external_api_msgs/srv/set_operator.hpp"

namespace external_api
{

class Operator : public rclcpp::Node
{
public:
  explicit Operator(const rclcpp::NodeOptions & options);

private:
  using SetOperator = tier4_external_api_msgs::srv::SetOperator;
  using SetObserver = tier4_external_api_msgs::srv::SetObserver;
  using GetOperator = tier4_external_api_msgs::msg::Operator;
  using GetObserver = tier4_external_api_msgs::msg::Observer;

  // ros interface
  rclcpp::CallbackGroup::SharedPtr group_;
  tier4_api_utils::Service<SetOperator>::SharedPtr srv_set_operator_;
  tier4_api_utils::Service<SetObserver>::SharedPtr srv_set_observer_;
  tier4_api_utils::Client<SetOperator>::SharedPtr cli_set_operator_;
  tier4_api_utils::Client<SetObserver>::SharedPtr cli_set_observer_;
  rclcpp::Publisher<GetOperator>::SharedPtr pub_get_operator_;
  rclcpp::Publisher<GetObserver>::SharedPtr pub_get_observer_;
  rclcpp::Subscription<GetOperator>::SharedPtr sub_get_operator_;
  rclcpp::Subscription<GetObserver>::SharedPtr sub_get_observer_;

  // ros callback
  void setOperator(
    const tier4_external_api_msgs::srv::SetOperator::Request::SharedPtr request,
    const tier4_external_api_msgs::srv::SetOperator::Response::SharedPtr response);
  void setObserver(
    const tier4_external_api_msgs::srv::SetObserver::Request::SharedPtr request,
    const tier4_external_api_msgs::srv::SetObserver::Response::SharedPtr response);
  void onOperator(const tier4_external_api_msgs::msg::Operator::ConstSharedPtr message);
  void onObserver(const tier4_external_api_msgs::msg::Observer::ConstSharedPtr message);
};

}  // namespace external_api

#endif  // OPERATOR_HPP_

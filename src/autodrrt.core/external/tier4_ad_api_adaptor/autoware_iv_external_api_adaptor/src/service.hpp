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

#ifndef SERVICE_HPP_
#define SERVICE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "tier4_api_utils/tier4_api_utils.hpp"

#include "tier4_external_api_msgs/msg/service.hpp"
#include "tier4_external_api_msgs/srv/set_service.hpp"

namespace external_api
{

class Service : public rclcpp::Node
{
public:
  explicit Service(const rclcpp::NodeOptions & options);

private:
  using SetService = tier4_external_api_msgs::srv::SetService;
  using ServiceMsg = tier4_external_api_msgs::msg::Service;

  // ros interface
  tier4_api_utils::Service<SetService>::SharedPtr srv_set_service_;
  rclcpp::Publisher<ServiceMsg>::SharedPtr pub_get_service_;

  // ros callback
  void setService(
    const tier4_external_api_msgs::srv::SetService::Request::SharedPtr request,
    const tier4_external_api_msgs::srv::SetService::Response::SharedPtr response);
};

}  // namespace external_api

#endif  // SERVICE_HPP_

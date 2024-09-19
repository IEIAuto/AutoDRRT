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

#ifndef CPU_USAGE_HPP_
#define CPU_USAGE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "tier4_api_utils/tier4_api_utils.hpp"

#include "tier4_external_api_msgs/msg/cpu_status.hpp"
#include "tier4_external_api_msgs/msg/cpu_usage.hpp"

namespace external_api
{

class CpuUsage : public rclcpp::Node
{
public:
  explicit CpuUsage(const rclcpp::NodeOptions & options);

private:
  rclcpp::Publisher<tier4_external_api_msgs::msg::CpuUsage>::SharedPtr pub_cpu_usage_;
  rclcpp::Subscription<tier4_external_api_msgs::msg::CpuUsage>::SharedPtr sub_cpu_usage_;
};

}  // namespace external_api

#endif  // CPU_USAGE_HPP_

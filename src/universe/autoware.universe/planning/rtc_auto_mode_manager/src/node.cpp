// Copyright 2022 TIER IV, Inc.
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

#include "rtc_auto_mode_manager/node.hpp"

#include <algorithm>

namespace rtc_auto_mode_manager
{

RTCAutoModeManagerNode::RTCAutoModeManagerNode(const rclcpp::NodeOptions & node_options)
: Node("rtc_auto_mode_manager_node", node_options)
{
  const std::vector<std::string> module_list =
    declare_parameter("module_list", std::vector<std::string>());
  const std::vector<std::string> default_enable_list =
    declare_parameter("default_enable_list", std::vector<std::string>());

  for (const auto & module_name : module_list) {
    const bool enabled =
      std::count(default_enable_list.begin(), default_enable_list.end(), module_name) != 0;
    managers_.push_back(std::make_shared<RTCAutoModeManagerInterface>(this, module_name, enabled));
  }
  statuses_pub_ = create_publisher<AutoModeStatusArray>("output/auto_mode_statuses", 1);
  using std::chrono_literals::operator""ms;
  timer_ = rclcpp::create_timer(this, get_clock(), 100ms, [this] {
    AutoModeStatusArray auto_mode_statuses;
    for (const auto & m : managers_) {
      auto_mode_statuses.stamp = get_clock()->now();
      auto_mode_statuses.statuses.emplace_back(m->getAutoModeStatus());
    }
    statuses_pub_->publish(auto_mode_statuses);
    auto_mode_statuses.statuses.clear();
  });
}

}  // namespace rtc_auto_mode_manager

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(rtc_auto_mode_manager::RTCAutoModeManagerNode)

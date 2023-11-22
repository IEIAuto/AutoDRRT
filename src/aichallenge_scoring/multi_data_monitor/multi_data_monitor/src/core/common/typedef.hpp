// Copyright 2022 Takagi, Isamu
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

#ifndef CORE__COMMON__TYPEDEF_HPP_
#define CORE__COMMON__TYPEDEF_HPP_

#include "common/container.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace multi_data_monitor
{

struct StreamData;
struct FilterData;
struct WidgetData;
struct DesignData;
using StreamLink = std::shared_ptr<StreamData>;
using WidgetLink = std::shared_ptr<WidgetData>;
using DesignLink = std::shared_ptr<DesignData>;
using FilterLink = std::shared_ptr<FilterData>;
using StreamList = std::vector<StreamLink>;
using WidgetList = std::vector<WidgetLink>;
using DesignList = std::vector<DesignLink>;
using FilterList = std::vector<FilterLink>;

class BasicStream;
class BasicFilter;
class BasicWidget;
using Stream = std::shared_ptr<BasicStream>;
using Filter = std::shared_ptr<BasicFilter>;
using Widget = std::shared_ptr<BasicWidget>;
using StreamMaps = HashMap<StreamLink, Stream>;
using FilterMaps = HashMap<FilterLink, Filter>;
using WidgetMaps = HashMap<WidgetLink, Widget>;

}  // namespace multi_data_monitor

namespace generic_type_utility
{

class GenericMessage;
class GenericProperty;

}  // namespace generic_type_utility

namespace rclcpp
{

class Node;
class TimerBase;
class GenericSubscription;
class QoS;

}  // namespace rclcpp

namespace multi_data_monitor::ros
{

using Node = std::shared_ptr<rclcpp::Node>;
using Timer = std::shared_ptr<rclcpp::TimerBase>;
using Subscription = std::shared_ptr<rclcpp::GenericSubscription>;

}  // namespace multi_data_monitor::ros

#endif  // CORE__COMMON__TYPEDEF_HPP_

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

#include "stream_runner.hpp"

namespace multi_data_monitor
{

void StreamRunner::create(const ConfigData & config)
{
  return create(config, WidgetMaps());
}

void StreamRunner::create(const ConfigData & config, const WidgetMaps & widgets)
{
  const auto filters = filter_loader_.create(config.filters);
  stream_loader_.create(config.streams, filters, widgets);
  rclcpp_runner_.set_topics(stream_loader_.topics());
}

void StreamRunner::start(ros::Node node)
{
  rclcpp_runner_.start(node);
}

void StreamRunner::shutdown()
{
  rclcpp_runner_.shutdown();
  filter_loader_.release();
  stream_loader_.release();
}

}  // namespace multi_data_monitor

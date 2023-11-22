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

#include "widget_runner.hpp"
#include "stream/basic.hpp"
#include <multi_data_monitor/widget.hpp>

namespace multi_data_monitor
{

void WidgetRunner::create(const ConfigData & config)
{
  const auto widgets = widget_loader_.create(config.widgets, config.designs);
  stream_runner_.create(config, widgets);
}

void WidgetRunner::start(ros::Node node)
{
  stream_runner_.start(node);
}

void WidgetRunner::shutdown()
{
  stream_runner_.shutdown();
  widget_loader_.release();
}

QWidget * WidgetRunner::take_root_widget()
{
  return widget_loader_.take_root_widget();
}

}  // namespace multi_data_monitor

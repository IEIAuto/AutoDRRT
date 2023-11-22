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

#include "panel.hpp"
#include <multi_data_monitor/widget.hpp>

namespace multi_data_monitor
{

PanelStream::PanelStream(Widget widget)
{
  widget_ = widget;
}

void PanelStream::setting(YAML::Node yaml)
{
  (void)yaml;
}

void PanelStream::message(const Packet & packet)
{
  if (widget_)
  {
    widget_->call_apply(packet);
  }
}

}  // namespace multi_data_monitor

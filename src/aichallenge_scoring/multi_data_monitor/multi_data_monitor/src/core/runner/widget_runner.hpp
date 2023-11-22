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

#ifndef CORE__RUNNER__WIDGET_RUNNER_HPP_
#define CORE__RUNNER__WIDGET_RUNNER_HPP_

#include "common/typedef.hpp"
#include "config/types.hpp"
#include "loader/widget_loader.hpp"
#include "runner/stream_runner.hpp"

namespace multi_data_monitor
{

class WidgetRunner final
{
public:
  void create(const ConfigData & config);
  void start(ros::Node node);
  void shutdown();
  QWidget * take_root_widget();

private:
  WidgetLoader widget_loader_;
  StreamRunner stream_runner_;
};

}  // namespace multi_data_monitor

#endif  // CORE__RUNNER__WIDGET_RUNNER_HPP_

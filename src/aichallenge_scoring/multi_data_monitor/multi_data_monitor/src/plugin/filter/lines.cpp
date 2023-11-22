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

#include <multi_data_monitor/action.hpp>
#include <algorithm>
#include <string>

namespace multi_data_monitor
{

class Lines : public multi_data_monitor::Action
{
private:
  int lines_;

public:
  void Initialize(const YAML::Node & yaml) { lines_ = yaml["lines"].as<int>(); }
  MonitorValues Apply(const MonitorValues & input) override
  {
    const auto value = input.value.as<std::string>();
    const auto count = std::count(value.begin(), value.end(), '\n');
    const auto lines = std::string(std::max(0L, lines_ - count - 1), '\n');
    return {YAML::Node(value + lines), input.attrs};
  }
};

}  // namespace multi_data_monitor

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(multi_data_monitor::Lines, multi_data_monitor::BasicFilter)

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

#ifndef CORE__PARSER__CONSTRUCTION_HPP_
#define CORE__PARSER__CONSTRUCTION_HPP_

#include "config/types.hpp"

namespace multi_data_monitor
{

class ParseBasicObject
{
public:
  ConfigData execute(ConfigFile & file);

private:
  void parse_filter_root(YAML::Node yaml);
  void parse_stream_root(YAML::Node yaml);
  void parse_widget_root(YAML::Node yaml);
  void parse_subscription(YAML::Node yaml);
  FilterLink parse_filter_yaml(YAML::Node yaml);
  FilterLink parse_filter_link(YAML::Node yaml);
  FilterLink parse_filter_dict(YAML::Node yaml);
  FilterLink parse_filter_list(YAML::Node yaml);
  StreamLink parse_stream_yaml(YAML::Node yaml);
  StreamLink parse_stream_link(YAML::Node yaml);
  StreamLink parse_stream_dict(YAML::Node yaml);
  WidgetLink parse_widget_yaml(YAML::Node yaml);
  WidgetLink parse_widget_link(YAML::Node yaml);
  WidgetLink parse_widget_dict(YAML::Node yaml);
  ConfigData data_;
};

}  // namespace multi_data_monitor

#endif  // CORE__PARSER__CONSTRUCTION_HPP_

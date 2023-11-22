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
#include <fmt/format.h>
#include <string>

namespace multi_data_monitor
{

class Format : public multi_data_monitor::Action
{
private:
  std::string format_;
  enum class Type { Double, String, Uint64, Sint64, Unknown } type_;

public:
  void Initialize(const YAML::Node & yaml)
  {
    const auto type = yaml["type"].as<std::string>();
    format_ = yaml["format"].as<std::string>();
    type_ = Type::Unknown;
    // clang-format off
    if (type == "double") { type_ = Type::Double; }
    if (type == "string") { type_ = Type::String; }
    if (type == "uint64") { type_ = Type::Uint64; }
    if (type == "sint64") { type_ = Type::Sint64; }
    // clang-format on

    // TODO(Takagi, Isamu): warning
  }
  MonitorValues Apply(const MonitorValues & input) override
  {
    YAML::Node value;
    switch (type_)
    {
      case Type::Double:
        value = fmt::format(format_, input.value.as<double>());
        break;
      case Type::Uint64:
        value = fmt::format(format_, input.value.as<uint64_t>());
        break;
      case Type::Sint64:
        value = fmt::format(format_, input.value.as<int64_t>());
        break;
      case Type::String:
        value = fmt::format(format_, input.value.as<std::string>());
        break;
      default:
        value = input.value;
        break;
    }
    return {value, input.attrs};
  }
};

}  // namespace multi_data_monitor

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(multi_data_monitor::Format, multi_data_monitor::BasicFilter)

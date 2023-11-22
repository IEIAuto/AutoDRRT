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

#ifndef PLUGIN__FILTER__COMPONENTS__SETTER_HPP_
#define PLUGIN__FILTER__COMPONENTS__SETTER_HPP_

#include <multi_data_monitor/packet.hpp>
#include <yaml-cpp/yaml.h>
#include <optional>
#include <string>
#include <unordered_map>

namespace multi_data_monitor
{

class SetAction
{
public:
  explicit SetAction(YAML::Node & yaml)
  {
    const auto value = yaml["value"];
    if (value)
    {
      yaml.remove("value");
      value_ = value;
    }

    const auto attrs = yaml["attrs"];
    if (attrs)
    {
      yaml.remove("attrs");
      for (const auto & pair : attrs)
      {
        const auto name = pair.first.as<std::string>();
        const auto attr = pair.second.as<std::string>();
        attrs_[name] = attr;
      }
    }
  }

  Packet apply(const Packet & packet) const
  {
    std::unordered_map<std::string, std::string> attrs;
    attrs.insert(attrs_.begin(), attrs_.end());
    attrs.insert(packet.attrs.begin(), packet.attrs.end());
    return {value_.value_or(packet.value), attrs};
  }

private:
  std::optional<YAML::Node> value_;
  std::unordered_map<std::string, std::string> attrs_;
};

}  // namespace multi_data_monitor

#endif  // PLUGIN__FILTER__COMPONENTS__SETTER_HPP_

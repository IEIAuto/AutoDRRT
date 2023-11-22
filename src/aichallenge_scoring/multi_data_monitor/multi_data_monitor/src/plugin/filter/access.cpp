// Copyright 2023 Takagi, Isamu
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

#include <multi_data_monitor/filter.hpp>
#include <string>

namespace multi_data_monitor
{

class Access : public BasicFilter
{
public:
  void setup(YAML::Node yaml) override;
  Packet apply(const Packet & packet) override;

private:
  std::vector<std::string> fields_;
  YAML::Node value_;
};

void Access::setup(YAML::Node yaml)
{
  switch (yaml["field"].Type())
  {
    case YAML::NodeType::Scalar:
      fields_.push_back(yaml["field"].as<std::string>());
      break;

    case YAML::NodeType::Sequence:
      fields_ = yaml["field"].as<std::vector<std::string>>();
      break;

    default:
      // TODO(Takagi, Isamu): warning
      break;
  }
  value_ = yaml["fails"];
}

Packet Access::apply(const Packet & packet)
{
  YAML::Node value = packet.value;
  for (const auto & field : fields_)
  {
    if (!value[field])
    {
      value.reset(YAML::Clone(value_));
      break;
    }
    value.reset(value[field]);
  }
  return {value, packet.attrs};
}

}  // namespace multi_data_monitor

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(multi_data_monitor::Access, multi_data_monitor::BasicFilter)

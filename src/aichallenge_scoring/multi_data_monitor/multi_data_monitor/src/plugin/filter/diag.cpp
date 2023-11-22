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

class DiagFind : public BasicFilter
{
public:
  void setup(YAML::Node yaml) override;
  Packet apply(const Packet & packet) override;

private:
  std::string name_;
};

void DiagFind::setup(YAML::Node yaml)
{
  name_ = yaml["name"].as<std::string>();
}

Packet DiagFind::apply(const Packet & packet)
{
  for (const auto & status : packet.value)
  {
    if (name_ == status["name"].as<std::string>())
    {
      return {status, packet.attrs};
    }
  }
  return {YAML::Node(), packet.attrs};
}

}  // namespace multi_data_monitor

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(multi_data_monitor::DiagFind, multi_data_monitor::BasicFilter)

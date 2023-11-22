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

#include <multi_data_monitor/filter.hpp>
#include <cmath>
#include <string>

namespace multi_data_monitor
{

class Units : public BasicFilter
{
public:
  void setup(YAML::Node yaml) override;
  Packet apply(const Packet & packet) override;

private:
  double coefficient_;
};

void Units::setup(YAML::Node yaml)
{
  const auto mode = yaml["mode"].as<std::string>();
  coefficient_ = 1.0;

  // clang-format off
  if (mode == "mps_to_kph") { coefficient_ = 1.0 * 3.6; return; }
  if (mode == "kph_to_mps") { coefficient_ = 1.0 / 3.6; return; }
  if (mode == "deg_to_rad") { coefficient_ = M_PI / 180.0; return; }
  if (mode == "rad_to_deg") { coefficient_ = 180.0 / M_PI; return; }
  // clang-format on

  // TODO(Takagi, Isamu): error handling
}

Packet Units::apply(const Packet & packet)
{
  const auto value = coefficient_ * packet.value.as<double>();
  return {YAML::Node(value), packet.attrs};
}

}  // namespace multi_data_monitor

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(multi_data_monitor::Units, multi_data_monitor::BasicFilter)

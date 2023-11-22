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

#include "components/conditions.hpp"
#include "components/setter.hpp"
#include <multi_data_monitor/filter.hpp>

namespace multi_data_monitor
{

class SetIfAction : private SetAction, Condition
{
public:
  SetIfAction(const std::string & type, YAML::Node & yaml) : SetAction(yaml), Condition(type, yaml) {}
  using Condition::eval;
  using SetAction::apply;
};

class SetIf : public BasicFilter
{
public:
  void setup(YAML::Node yaml) override;
  Packet apply(const Packet & packet) override;

private:
  std::unique_ptr<SetIfAction> action_;
};

void SetIf::setup(YAML::Node yaml)
{
  const auto type = yaml["type"].as<std::string>();
  yaml.remove("type");
  action_ = std::make_unique<SetIfAction>(type, yaml);
}

Packet SetIf::apply(const Packet & packet)
{
  return action_->eval(packet.value) ? action_->apply(packet) : packet;
}

class SetFirstIf : public BasicFilter
{
public:
  void setup(YAML::Node yaml) override;
  Packet apply(const Packet & packet) override;

private:
  std::vector<std::unique_ptr<SetIfAction>> actions_;
};

void SetFirstIf::setup(YAML::Node yaml)
{
  const auto type = yaml["type"].as<std::string>();
  const auto list = yaml["list"];
  yaml.remove("type");
  yaml.remove("list");

  if (list.IsSequence())
  {
    for (auto item : list)
    {
      actions_.push_back(std::make_unique<SetIfAction>(type, item));
    }
  }
}

Packet SetFirstIf::apply(const Packet & packet)
{
  for (const auto & action : actions_)
  {
    if (action->eval(packet.value))
    {
      return action->apply(packet);
    }
  }
  return packet;
}

}  // namespace multi_data_monitor

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(multi_data_monitor::SetIf, multi_data_monitor::BasicFilter)
PLUGINLIB_EXPORT_CLASS(multi_data_monitor::SetFirstIf, multi_data_monitor::BasicFilter)

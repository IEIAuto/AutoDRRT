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

#include "topic.hpp"
#include "common/exceptions.hpp"
#include "common/util.hpp"
#include "common/yaml.hpp"
#include <generic_type_utility/generic_message.hpp>
#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace
{

std::string convert_qos(const rclcpp::QoS & qos)
{
  // clang-format off
  static const std::unordered_map<rmw_qos_reliability_policy_t, std::string> reliabilities =
  {
    {RMW_QOS_POLICY_RELIABILITY_SYSTEM_DEFAULT, "d"},
    {RMW_QOS_POLICY_RELIABILITY_RELIABLE,       "r"},
    {RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,    "b"}
  };
  static const std::unordered_map<rmw_qos_durability_policy_t, std::string> durabilities =
  {
    {RMW_QOS_POLICY_DURABILITY_SYSTEM_DEFAULT,  "d"},
    {RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL, "t"},
    {RMW_QOS_POLICY_DURABILITY_VOLATILE,        "v"}
  };
  // clang-format on

  const auto get_str = [](const auto & map, const auto & qos) { return map.count(qos) ? map.at(qos) : "x"; };
  const auto profile = qos.get_rmw_qos_profile();
  const auto reliability = get_str(reliabilities, profile.reliability);
  const auto durability = get_str(durabilities, profile.durability);
  return reliability + durability + std::to_string(profile.depth);
}

rclcpp::QoS convert_qos(const std::string text)
{
  using multi_data_monitor::ConfigError;

  // clang-format off
  static const std::unordered_map<std::string, rmw_qos_reliability_policy_t> reliabilities =
  {
    {"d", RMW_QOS_POLICY_RELIABILITY_SYSTEM_DEFAULT},
    {"r", RMW_QOS_POLICY_RELIABILITY_RELIABLE},
    {"b", RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT}
  };
  static const std::unordered_map<std::string, rmw_qos_durability_policy_t> durabilities =
  {
    {"d", RMW_QOS_POLICY_DURABILITY_SYSTEM_DEFAULT},
    {"t", RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL},
    {"v", RMW_QOS_POLICY_DURABILITY_VOLATILE}
  };
  // clang-format on

  if (text.length() < 3)
  {
    throw ConfigError("topic qos must be at least 3 characters");
  }

  const auto create_qos = [](const std::string & depth)
  {
    try
    {
      return rclcpp::QoS(std::stoi(depth));
    }
    catch (const std::exception &)
    {
      throw ConfigError("invalid qos depth: " + depth);
    }
  };

  rclcpp::QoS qos = create_qos(text.substr(2));
  const auto r = text.substr(0, 1);
  const auto d = text.substr(1, 1);

  if (reliabilities.count(r) == 0)
  {
    throw ConfigError("unknown qos reliability: " + r);
  }
  qos.reliability(reliabilities.at(r));

  if (durabilities.count(d) == 0)
  {
    throw ConfigError("unknown qos durability: " + d);
  }
  qos.durability(durabilities.at(d));

  return qos;
}

}  // namespace

namespace multi_data_monitor
{

TopicStream::TopicStream()
{
}

TopicStream::~TopicStream()
{
}

void TopicStream::setting(YAML::Node yaml)
{
  name_ = yaml::take_required(yaml, "name").as<std::string>("");
  type_ = yaml::take_optional(yaml, "type").as<std::string>("");
  qos_ = yaml::take_optional(yaml, "qos").as<std::string>("");
  sub_error_ = false;
}

void TopicStream::message(const Packet & packet)
{
  outputs(packet);
}

void TopicStream::update(ros::Node node)
{
  if (!sub_ && !sub_error_)
  {
    create_subscription(node);
  }
}

void TopicStream::shutdown()
{
  remove_subscription();
}

void TopicStream::create_subscription(ros::Node node)
{
  auto type = type_;
  auto qos = qos_;

  if (type.empty() || qos.empty())
  {
    const auto infos = node->get_publishers_info_by_topic(name_);
    if (infos.empty())
    {
      RCLCPP_DEBUG_STREAM(node->get_logger(), "no topic info: " + name_);
      return;
    }

    std::unordered_set<std::string> types;
    std::unordered_set<std::string> qoses;
    for (const auto & info : infos)
    {
      types.insert(info.topic_type());
      qoses.insert(convert_qos(info.qos_profile()));
    }

    if (type.empty())
    {
      if (types.size() != 1)
      {
        RCLCPP_WARN_STREAM(node->get_logger(), "topic type is not unique: " << util::join(types));
        return;
      }
      type = *types.begin();
    }

    if (qos.empty())
    {
      if (qoses.size() != 1)
      {
        RCLCPP_WARN_STREAM(node->get_logger(), "topic qos is not unique: " << util::join(qoses));
        return;
      }
      qos = *qoses.begin();
    }
  }

  const auto callback = [this, node](const std::shared_ptr<const rclcpp::SerializedMessage> serialized)
  {
    const auto value = generic_->deserialize(*serialized);
    const auto attrs = std::unordered_map<std::string, std::string>();
    try
    {
      message(Packet{value, attrs});
    }
    catch (const std::exception & error)
    {
      RCLCPP_ERROR_STREAM(node->get_logger(), error.what());
    }
  };

  RCLCPP_INFO_STREAM(node->get_logger(), "start subscription: " + name_ + " [" + type + "]");
  generic_ = std::make_unique<generic_type_utility::GenericMessage>(type);
  for (const auto & property : properties_)
  {
    if (!generic_->validate(property))
    {
      sub_error_ = true;
      RCLCPP_ERROR_STREAM(node->get_logger(), "invalid property: " + property.path() + " [" + type + "]");
      return;
    }
  }
  sub_ = node->create_generic_subscription(name_, type, convert_qos(qos), callback);
}

void TopicStream::remove_subscription()
{
  generic_.reset();
  sub_.reset();
}

void TopicStream::validate(const generic_type_utility::GenericProperty & property)
{
  properties_.push_back(property);
}

}  // namespace multi_data_monitor

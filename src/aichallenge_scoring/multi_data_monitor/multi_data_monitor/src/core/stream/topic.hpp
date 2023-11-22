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

#ifndef CORE__STREAM__TOPIC_HPP_
#define CORE__STREAM__TOPIC_HPP_

#include "common/typedef.hpp"
#include "stream/basic.hpp"
#include <memory>
#include <string>
#include <vector>

namespace multi_data_monitor
{

struct TopicStream : public InOutStream
{
public:
  TopicStream();
  ~TopicStream() override;
  void setting(YAML::Node yaml) override;
  void message(const Packet & packet) override;
  void update(ros::Node node);
  void shutdown();
  void validate(const generic_type_utility::GenericProperty & property);

private:
  void create_subscription(ros::Node node);
  void remove_subscription();
  std::string name_;
  std::string type_;
  std::string qos_;
  std::unique_ptr<generic_type_utility::GenericMessage> generic_;
  std::vector<generic_type_utility::GenericProperty> properties_;

  bool sub_error_;
  ros::Subscription sub_;
};

}  // namespace multi_data_monitor

#endif  // CORE__STREAM__TOPIC_HPP_

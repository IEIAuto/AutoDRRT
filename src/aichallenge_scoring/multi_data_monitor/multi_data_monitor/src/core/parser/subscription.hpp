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

#ifndef CORE__PARSER__SUBSCRIPTION_HPP_
#define CORE__PARSER__SUBSCRIPTION_HPP_

#include "config/types.hpp"
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace multi_data_monitor
{

class MergeSubscription : public ConfigParserInterface
{
public:
  std::string name() { return "merge-subscription"; }
  ConfigData execute(const ConfigData & input) override;

private:
  void handle_subscription(const StreamLink & input);
  void create_topic(const std::string & name, const std::string & code);
  void create_field(const std::string & name, const std::string & code);

  struct TopicData
  {
    StreamLink node;
    std::unordered_set<std::string> types;
    std::unordered_set<std::string> qoses;
  };
  struct FieldData
  {
    StreamLink node;
    std::unordered_set<std::string> types;
  };

  ConfigData output_;
  std::unordered_map<std::string, TopicData> topics_;
  std::unordered_map<std::string, FieldData> fields_;
};

}  // namespace multi_data_monitor

#endif  // CORE__PARSER__SUBSCRIPTION_HPP_

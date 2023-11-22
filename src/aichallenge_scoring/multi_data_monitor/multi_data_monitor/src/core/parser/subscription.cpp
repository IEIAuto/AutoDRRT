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

#include "subscription.hpp"
#include "common/exceptions.hpp"
#include "common/util.hpp"
#include "common/yaml.hpp"
#include <string>

namespace multi_data_monitor
{

ConfigData MergeSubscription::execute(const ConfigData & input)
{
  output_ = input;

  for (const auto & stream : input.streams)
  {
    if (stream->klass == builtin::subscription)
    {
      handle_subscription(stream);
    }
  }

  // TODO(Takagi, Isamu): refactor

  for (const auto & [name, data] : topics_)
  {
    if (2 <= data.qoses.size())
    {
      const auto list = "[" + util::join(data.qoses) + "]";
      throw ConfigError("topic qos is not unique: " + list + " for " + name);
    }
    if (1 == data.qoses.size())
    {
      data.node->yaml["qos"] = *data.qoses.begin();
    }
  }

  for (const auto & [name, data] : topics_)
  {
    if (2 <= data.types.size())
    {
      const auto list = "[" + util::join(data.types) + "]";
      throw ConfigError("topic type is not unique: " + list + " for " + name);
    }
    if (1 == data.types.size())
    {
      data.node->yaml["type"] = *data.types.begin();
    }
  }

  for (const auto & [name, data] : fields_)
  {
    if (2 <= data.types.size())
    {
      const auto list = "[" + util::join(data.types) + "]";
      throw ConfigError("field type is not unique: " + list + " for " + name);
    }
    if (1 == data.types.size())
    {
      data.node->yaml["type"] = *data.types.begin();
    }
  }

  return output_;
}

void MergeSubscription::handle_subscription(const StreamLink & input)
{
  const auto topic_name = yaml::take_required(input->yaml, "topic").as<std::string>();
  const auto field_name = yaml::take_required(input->yaml, "field").as<std::string>();
  const auto field_code = topic_name + ": " + field_name;
  create_topic(topic_name, topic_name);
  create_field(field_name, field_code);
  TopicData & topic_data = topics_[topic_name];
  FieldData & field_data = fields_[field_code];

  const auto topic_type = yaml::take_optional(input->yaml, "topic-type");
  if (topic_type)
  {
    topic_data.types.insert(topic_type.as<std::string>());
  }

  const auto field_type = yaml::take_optional(input->yaml, "field-type");
  if (field_type)
  {
    field_data.types.insert(field_type.as<std::string>());
  }

  const auto qos = yaml::take_optional(input->yaml, "qos");
  if (qos)
  {
    topic_data.qoses.insert(qos.as<std::string>());
  }

  StreamLink topic = topic_data.node;
  StreamLink field = field_data.node;
  input->refer = field;
  field->items = StreamList{topic};
  yaml::check_empty(input->yaml);
}

void MergeSubscription::create_topic(const std::string & name, const std::string & code)
{
  if (topics_.count(code) == 0)
  {
    const auto node = output_.create_stream("@topic");
    node->system = true;
    node->yaml["name"] = name;
    topics_[code].node = node;
  }
}

void MergeSubscription::create_field(const std::string & name, const std::string & code)
{
  if (fields_.count(code) == 0)
  {
    const auto node = output_.create_stream("@field");
    node->system = true;
    node->yaml["name"] = name;
    fields_[code].node = node;
  }
}

}  // namespace multi_data_monitor

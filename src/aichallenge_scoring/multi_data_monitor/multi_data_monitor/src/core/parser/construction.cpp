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

#include "construction.hpp"
#include "common/exceptions.hpp"
#include "common/yaml.hpp"
#include <functional>
#include <string>

namespace multi_data_monitor
{

using RootFunc = void (ParseBasicObject::*)(YAML::Node);

void parse(YAML::Node & yaml, const std::string & name, ParseBasicObject * self, RootFunc func)
{
  const auto nodes = yaml::take_optional(yaml, name);
  if (!nodes.IsDefined())
  {
    return;
  }
  if (!nodes.IsSequence())
  {
    throw ConfigError("config section '" + name + "' is not a sequence");
  }
  for (const auto & node : nodes)
  {
    std::invoke(func, self, node);
  }
}

ConfigData ParseBasicObject::execute(ConfigFile & file)
{
  parse(file.yaml, "subscriptions", this, &ParseBasicObject::parse_subscription);
  parse(file.yaml, "filters", this, &ParseBasicObject::parse_filter_root);
  parse(file.yaml, "streams", this, &ParseBasicObject::parse_stream_root);
  parse(file.yaml, "widgets", this, &ParseBasicObject::parse_widget_root);
  return data_;
}

void ParseBasicObject::parse_filter_root(YAML::Node yaml)
{
  parse_filter_yaml(yaml);
}

void ParseBasicObject::parse_stream_root(YAML::Node yaml)
{
  parse_stream_yaml(yaml);
}

void ParseBasicObject::parse_widget_root(YAML::Node yaml)
{
  parse_widget_yaml(yaml);
}

void ParseBasicObject::parse_subscription(YAML::Node topic)
{
  if (!topic.IsMap())
  {
    // TODO(Takagi, Isamu): error message
    throw ConfigError("is not map");
  }

  const auto fields = yaml::take_required(topic, "fields");
  if (!fields.IsSequence())
  {
    // TODO(Takagi, Isamu): error message
    throw ConfigError("topic fields is not sequence");
  }

  const auto topic_name = yaml::take_required(topic, "name");
  const auto topic_type = yaml::take_optional(topic, "type");
  const auto topic_qos = yaml::take_optional(topic, "qos");
  yaml::check_empty(topic);

  for (YAML::Node field : fields)
  {
    const auto label = yaml::take_optional(field, "label").as<std::string>("");
    const auto field_name = yaml::take_required(field, "name");
    const auto field_type = yaml::take_optional(field, "type");
    yaml::check_empty(field);

    YAML::Node yaml;
    yaml["topic"] = topic_name;
    yaml["field"] = field_name;
    yaml["topic-type"] = topic_type;
    yaml["field-type"] = field_type;
    yaml["qos"] = topic_qos;

    const auto stream = data_.create_stream(builtin::subscription, label, yaml);
    stream->system = true;
  }
}

FilterLink ParseBasicObject::parse_filter_yaml(YAML::Node yaml)
{
  if (yaml.IsScalar())
  {
    return parse_filter_link(yaml);
  }
  if (yaml.IsMap())
  {
    return parse_filter_dict(yaml);
  }
  if (yaml.IsSequence())
  {
    return parse_filter_list(yaml);
  }
  // TODO(Takagi, Isamu): error message
  throw ConfigError("unexpected filter format");
}

StreamLink ParseBasicObject::parse_stream_yaml(YAML::Node yaml)
{
  if (yaml.IsScalar())
  {
    return parse_stream_link(yaml);
  }
  if (yaml.IsMap())
  {
    return parse_stream_dict(yaml);
  }
  // TODO(Takagi, Isamu): error message
  throw ConfigError("unexpected stream format");
}

WidgetLink ParseBasicObject::parse_widget_yaml(YAML::Node yaml)
{
  if (yaml.IsScalar())
  {
    return parse_widget_link(yaml);
  }
  if (yaml.IsMap())
  {
    return parse_widget_dict(yaml);
  }
  // TODO(Takagi, Isamu): error message
  throw ConfigError("unexpected widget format");
}

FilterLink ParseBasicObject::parse_filter_link(YAML::Node yaml)
{
  FilterLink filter = data_.create_filter(builtin::relay);
  filter->system = true;
  filter->yaml["refer"] = yaml.as<std::string>();
  return filter;
}

StreamLink ParseBasicObject::parse_stream_link(YAML::Node yaml)
{
  StreamLink stream = data_.create_stream(builtin::relay);
  stream->system = true;
  stream->yaml["refer"] = yaml.as<std::string>();
  return stream;
}

WidgetLink ParseBasicObject::parse_widget_link(YAML::Node yaml)
{
  WidgetLink widget = data_.create_widget(builtin::relay);
  widget->system = true;
  widget->yaml["refer"] = yaml.as<std::string>();
  return widget;
}

FilterLink ParseBasicObject::parse_filter_dict(YAML::Node yaml)
{
  const auto klass = yaml::take_required(yaml, "class").as<std::string>("");
  const auto label = yaml::take_optional(yaml, "label").as<std::string>("");

  FilterLink filter = data_.create_filter(klass, label, yaml);
  if (klass == builtin::function)
  {
    YAML::Node rules = yaml::take_required(yaml, "rules");
    if (!rules.IsSequence())
    {
      YAML::Node array;
      array.push_back(rules);
      rules.reset(array);
    }
    for (const auto & rule : rules)
    {
      filter->items.push_back(parse_filter_yaml(rule));
    }
  }
  return filter;
}

StreamLink ParseBasicObject::parse_stream_dict(YAML::Node yaml)
{
  const auto klass = yaml::take_required(yaml, "class").as<std::string>("");
  const auto label = yaml::take_optional(yaml, "label").as<std::string>("");
  const auto input = yaml::take_optional(yaml, "input");
  const auto rules = yaml::take_optional(yaml, "rules");

  StreamLink stream = data_.create_stream(klass, label, yaml);
  if (input)
  {
    stream->items = StreamList{parse_stream_yaml(input)};
  }
  if (rules)
  {
    // TODO(Takagi, Isamu): check if class is apply
    stream->apply = parse_filter_yaml(rules);
  }
  return stream;
}

WidgetLink ParseBasicObject::parse_widget_dict(YAML::Node yaml)
{
  const auto klass = yaml::take_required(yaml, "class").as<std::string>("");
  const auto label = yaml::take_optional(yaml, "label").as<std::string>("");
  const auto items = yaml::take_optional(yaml, "items");
  const auto input = yaml::take_optional(yaml, "input");
  const auto rules = yaml::take_optional(yaml, "rules");

  WidgetLink widget = data_.create_widget(klass, label, yaml);
  if (items)
  {
    if (!items.IsSequence())
    {
      throw ConfigError("widget property 'items' is not a sequence");
    }
    for (const auto & item : items)
    {
      widget->items.push_back(parse_widget_yaml(item));
    }
  }

  if (input)
  {
    StreamLink panel = data_.create_stream(builtin::panel);
    panel->system = true;
    panel->panel = widget;
    panel->items = StreamList{parse_stream_yaml(input)};

    if (rules)
    {
      StreamLink apply = data_.create_stream(builtin::apply);
      apply->system = true;
      apply->apply = parse_filter_yaml(rules);
      apply->items = panel->items;
      panel->items = StreamList{apply};
    }
  }
  return widget;
}

FilterLink ParseBasicObject::parse_filter_list(YAML::Node yaml)
{
  FilterLink filter = data_.create_filter(builtin::function);
  for (const auto & item : yaml)
  {
    filter->items.push_back(parse_filter_yaml(item));
  }
  return filter;
}

}  // namespace multi_data_monitor

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

#include "yaml.hpp"
#include "exceptions.hpp"
#include <string>
#include <vector>

namespace multi_data_monitor::yaml
{

YAML::Node take_optional(YAML::Node & yaml, const std::string & name)
{
  const auto node = yaml[name];
  yaml.remove(name);
  return node;
}

YAML::Node take_required(YAML::Node & yaml, const std::string & name)
{
  const auto node = yaml[name];
  if (!node)
  {
    // TODO(Takagi, Isamu): add debug info
    throw ConfigError("required key: " + name);
  }
  yaml.remove(name);
  return node;
}

void check_empty(YAML::Node & yaml)
{
  std::vector<std::string> fields;
  for (const auto pair : yaml)
  {
    fields.push_back(pair.first.as<std::string>());
  }
  if (!fields.empty())
  {
    // TODO(Takagi, Isamu): use join
    std::string text;
    for (const auto & field : fields)
    {
      text += " " + field;
    }
    throw ConfigError("not empty:" + text);
  }
}

}  // namespace multi_data_monitor::yaml

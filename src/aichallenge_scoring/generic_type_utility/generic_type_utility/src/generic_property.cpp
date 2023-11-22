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

#include "generic_type_utility/generic_property.hpp"

namespace generic_type_utility
{

GenericProperty::GenericProperty(const std::string & path)
{
  std::string text = path;
  if (!text.empty() && text.front() != '.' && text.front() != '@')
  {
    text = "." + text;
  }

  std::vector<size_t> indices;
  {
    size_t pos = 0;
    while (pos = text.find_first_of(".@", pos), pos != std::string::npos)
    {
      indices.push_back(pos++);
    }
    indices.push_back(text.size());
  }

  for (size_t i = 1; i < indices.size(); ++i)
  {
    const auto pos = indices[i - 1];
    const auto len = indices[i] - indices[i - 1];
    if (len < 2) throw std::invalid_argument("token size is too short: " + path);

    const auto token = text.substr(pos + 1, len - 1);
    if (text[pos] == '.') elements_.push_back({token, std::nullopt});
    if (text[pos] == '@') elements_.push_back({std::nullopt, std::stoi(token)});
  }
  elements_.push_back({std::nullopt, std::nullopt});  // Add sentinel.
}

const std::string GenericProperty::path() const
{
  std::string result;
  for (const auto & element : elements_)
  {
    if (element.field) result += "." + element.field.value();
    if (element.index) result += "@" + std::to_string(element.index.value());
  }
  return result;
}

YAML::Node GenericProperty::apply(const YAML::Node & yaml) const
{
  YAML::Node node = yaml;
  for (const auto & element : elements_)
  {
    if (element.field) node.reset(node[element.field.value()]);
    if (element.index) node.reset(node[element.index.value()]);
  }
  return node;
}

}  // namespace generic_type_utility

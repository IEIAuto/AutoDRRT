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

#include "stylesheet.hpp"
#include "common/exceptions.hpp"
#include "common/path.hpp"
#include "common/yaml.hpp"
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace multi_data_monitor
{

DesignList ParseStyleSheet::execute(ConfigFile & file)
{
  const auto stylesheets = yaml::take_optional(file.yaml, "stylesheets");
  if (!stylesheets.IsDefined())
  {
    return DesignList();
  }
  if (!stylesheets.IsSequence())
  {
    throw ConfigError("config section 'stylesheets' is not a sequence");
  }

  DesignList result;
  for (const auto stylesheet : stylesheets)
  {
    result.push_back(parse_stylesheet(stylesheet));
  }
  return result;
}

std::string load_file(const std::string & input)
{
  const auto path = std::filesystem::path(path::resolve(input));
  if (!std::filesystem::exists(path))
  {
    throw FilePathError("stylesheet file not found '" + path.string() + "'");
  }

  std::ifstream ifs(path);
  if (!ifs)
  {
    throw RuntimeError("stylesheet file could not be read '" + path.string() + "'");
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return buffer.str();
}

DesignLink ParseStyleSheet::parse_stylesheet(YAML::Node yaml)
{
  if (!yaml.IsMap())
  {
    throw ConfigError("unexpected stream format");
  }

  DesignLink design = std::make_shared<DesignData>();
  design->klass = yaml::take_optional(yaml, "target").as<std::string>("");
  design->stylesheet = load_file(yaml::take_required(yaml, "path").as<std::string>());
  return design;
}

}  // namespace multi_data_monitor

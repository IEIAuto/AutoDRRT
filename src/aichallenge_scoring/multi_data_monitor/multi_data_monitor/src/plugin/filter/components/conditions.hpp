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

#ifndef PLUGIN__FILTER__COMPONENTS__CONDITIONS_HPP_
#define PLUGIN__FILTER__COMPONENTS__CONDITIONS_HPP_

#include <yaml-cpp/yaml.h>
#include <memory>
#include <string>

namespace multi_data_monitor
{

class Condition
{
public:
  Condition(const std::string & type, YAML::Node & yaml);
  ~Condition();
  bool eval(const YAML::Node & yaml) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace multi_data_monitor

#endif  // PLUGIN__FILTER__COMPONENTS__CONDITIONS_HPP_

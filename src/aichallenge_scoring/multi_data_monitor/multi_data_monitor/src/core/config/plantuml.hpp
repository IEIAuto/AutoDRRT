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

#ifndef CORE__CONFIG__PLANTUML_HPP_
#define CORE__CONFIG__PLANTUML_HPP_

#include "config/types.hpp"
#include <string>

namespace multi_data_monitor::plantuml
{

class Diagram
{
public:
  std::string convert(const ConfigData & data) const;
  void write(const ConfigData & data, const std::string & path) const;
};

}  // namespace multi_data_monitor::plantuml

#endif  // CORE__CONFIG__PLANTUML_HPP_

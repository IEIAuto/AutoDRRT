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

#include "print.hpp"
#include "common/yaml.hpp"
#include <iostream>
#include <string>

namespace multi_data_monitor
{

void PrintStream::setting(YAML::Node yaml)
{
  title_ = yaml::take_optional(yaml, "title").as<std::string>("");
}

void PrintStream::message(const Packet & packet)
{
  if (!title_.empty())
  {
    std::cout << title_ << std::endl;
  }
  std::cout << packet.value << std::endl;
}

}  // namespace multi_data_monitor

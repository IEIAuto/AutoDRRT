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

#include "check_system_class.hpp"
#include "common/exceptions.hpp"

namespace multi_data_monitor
{

ConfigData CheckSystemClass::execute(const ConfigData & input)
{
  for (const auto & stream : input.streams)
  {
    const auto & klass = stream->klass;
    if (!klass.empty() && klass[0] == '@' && !stream->system)
    {
      throw ConfigError("this class name is reserved: " + klass);
    }
  }
  return input;
}

}  // namespace multi_data_monitor

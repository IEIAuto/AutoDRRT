// Copyright 2023 Takagi, Isamu
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

#include "common/exceptions.hpp"
#include <multi_data_monitor/filter.hpp>

namespace multi_data_monitor
{

void BasicFilter::call_setup(YAML::Node yaml)
{
  // TODO(Takagi, Isamu): more plugin info
  try
  {
    setup(yaml);
  }
  catch (const std::exception & error)
  {
    throw PluginError(error.what() + std::string(" from ") + typeid(*this).name());
  }
}

Packet BasicFilter::call_apply(const Packet & packet)
{
  // TODO(Takagi, Isamu): more plugin info
  try
  {
    return apply(packet);
  }
  catch (const std::exception & error)
  {
    throw PluginError(error.what() + std::string(" from ") + typeid(*this).name());
  }
}

}  // namespace multi_data_monitor

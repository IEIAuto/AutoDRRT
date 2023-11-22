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

#include "function.hpp"
#include <vector>

namespace multi_data_monitor
{

FunctionFilter::FunctionFilter(const std::vector<Filter> & filters)
{
  filters_ = filters;
}

void FunctionFilter::setup(YAML::Node yaml)
{
  (void)yaml;
}

Packet FunctionFilter::apply(const Packet & packet)
{
  Packet result = packet;
  for (const auto & filter : filters_)
  {
    result = filter->call_apply(result);
  }
  return result;
}

}  // namespace multi_data_monitor

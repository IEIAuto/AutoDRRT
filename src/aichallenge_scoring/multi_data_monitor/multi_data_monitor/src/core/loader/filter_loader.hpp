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

#ifndef CORE__LOADER__FILTER_LOADER_HPP_
#define CORE__LOADER__FILTER_LOADER_HPP_

#include "common/typedef.hpp"
#include <multi_data_monitor/filter.hpp>
#include <pluginlib/class_loader.hpp>
#include <vector>

namespace multi_data_monitor
{

class FilterLoader final
{
public:
  FilterLoader();
  FilterMaps create(const FilterList & configs);
  void release();

private:
  Filter create_filter(const FilterLink & config, const FilterMaps & mapping);

  // The plugin loader must be first for release order.
  pluginlib::ClassLoader<BasicFilter> plugins_;
  std::vector<Filter> filters_;
};

}  // namespace multi_data_monitor

#endif  // CORE__LOADER__FILTER_LOADER_HPP_

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

#include "filter_loader.hpp"
#include "common/exceptions.hpp"
#include "config/types.hpp"
#include "filter/function.hpp"
#include <memory>
#include <string>
#include <unordered_map>

// TODO(Takagi, Isamu): merge filter loader
namespace
{
std::string get_full_plugin_name(const std::string & klass)
{
  if (klass.find("::") != std::string::npos)
  {
    return klass;
  }
  return multi_data_monitor::plugin::name::package + std::string("::") + klass;
}

}  // namespace

namespace multi_data_monitor
{

FilterLoader::FilterLoader() : plugins_(plugin::name::package, plugin::name::filter)
{
}

FilterMaps FilterLoader::create(const FilterList & configs)
{
  FilterMaps mapping;
  for (const auto & config : configs)
  {
    const auto filter = create_filter(config, mapping);
    mapping[config] = filters_.emplace_back(filter);

    // TODO(Takagi, Isamu): more plugin info
    try
    {
      filter->call_setup(config->yaml);
    }
    catch (const std::exception & error)
    {
      throw PluginError(config->klass + ".setup: " + error.what());
    }
  }
  return mapping;
}

Filter FilterLoader::create_filter(const FilterLink & config, const FilterMaps & mapping)
{
  if (config->klass == builtin::function)
  {
    return std::make_shared<FunctionFilter>(mapping.at(config->items));
  }

  // Search in default plugins if namespace is omitted.
  std::string klass = get_full_plugin_name(config->klass);
  if (!plugins_.isClassAvailable(klass))
  {
    throw ConfigError("unknown filter type: " + config->klass);
  }
  return plugins_.createSharedInstance(klass);
}

void FilterLoader::release()
{
  // TODO(Takagi, Isamu): check use count
  // Release shared_ptr to unload plugins.
  filters_.clear();
}

}  // namespace multi_data_monitor

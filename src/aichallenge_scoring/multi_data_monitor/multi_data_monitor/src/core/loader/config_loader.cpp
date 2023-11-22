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

#include "config_loader.hpp"
#include "parser/check_system_class.hpp"
#include "parser/construction.hpp"
#include "parser/file.hpp"
#include "parser/resolve_relation.hpp"
#include "parser/stylesheet.hpp"
#include "parser/subscription.hpp"

namespace multi_data_monitor
{

ConfigData ConfigLoader::Execute(const std::string & path)
{
  return ConfigLoader().execute(path);
}

ConfigLoader::ConfigLoader()
{
  parsers_.push_back(std::make_shared<MergeSubscription>());
  parsers_.push_back(std::make_shared<CheckSystemClass>());
  parsers_.push_back(std::make_shared<ConnectRelation>());
  parsers_.push_back(std::make_shared<ResolveRelation>());
  parsers_.push_back(std::make_shared<ReleaseRelation>());
  parsers_.push_back(std::make_shared<NormalizeRelation>());
}

void ConfigLoader::set_step_hook(HookFunction hook)
{
  step_hook_ = hook;
}

void ConfigLoader::set_last_hook(HookFunction hook)
{
  last_hook_ = hook;
}

ConfigData ConfigLoader::execute(const std::string & path) const
{
  auto file = ConfigFileLoader().execute(path);
  auto data = ParseBasicObject().execute(file);

  if (step_hook_)
  {
    step_hook_(0, "construct-node", data);
  }
  for (size_t i = 0; i < parsers_.size(); ++i)
  {
    data = parsers_[i]->execute(data);
    if (step_hook_)
    {
      step_hook_(i + 1, parsers_[i]->name(), data);
    }
  }
  if (last_hook_)
  {
    last_hook_(0, "", data);
  }

  data.designs = ParseStyleSheet().execute(file);
  return data;
}

}  // namespace multi_data_monitor

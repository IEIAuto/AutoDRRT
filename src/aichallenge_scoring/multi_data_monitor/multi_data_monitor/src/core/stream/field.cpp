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

#include "field.hpp"
#include "common/yaml.hpp"
#include <string>

namespace multi_data_monitor
{

void FieldStream::setting(YAML::Node yaml)
{
  const auto name = yaml::take_required(yaml, "name").as<std::string>("");
  const auto type = yaml::take_optional(yaml, "type").as<std::string>("");
  property_ = generic_type_utility::GenericProperty(name);
}

void FieldStream::message(const Packet & packet)
{
  outputs({property_.apply(packet.value), packet.attrs});
}

void FieldStream::validate(const std::shared_ptr<TopicStream> & topic)
{
  topic->validate(property_);
}

}  // namespace multi_data_monitor

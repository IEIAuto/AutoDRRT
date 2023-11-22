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

#include "generic_type_utility/generic_message.hpp"

namespace generic_type_utility
{

GenericMessage::GenericMessage(const std::string & type_name) : introspection_(type_name), serialization_(type_name)
{
}

bool GenericMessage::validate(const GenericProperty & property) const
{
  return introspection_.validate(property);
}

YAML::Node GenericMessage::deserialize(const rclcpp::SerializedMessage & serialized) const
{
  const auto message = introspection_.create_message();
  serialization_.deserialize(serialized, *message);
  return introspection_.make_yaml(*message);
}

}  // namespace generic_type_utility

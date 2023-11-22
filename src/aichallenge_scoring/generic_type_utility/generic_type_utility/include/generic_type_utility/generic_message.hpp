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

#ifndef GENERIC_TYPE_UTILITY__GENERIC_MESSAGE_HPP_
#define GENERIC_TYPE_UTILITY__GENERIC_MESSAGE_HPP_

#include "generic_type_utility/generic_property.hpp"
#include "generic_type_utility/ros2/introspection.hpp"
#include "generic_type_utility/ros2/serialization.hpp"
#include <string>

namespace generic_type_utility
{

class GenericMessage
{
public:
  explicit GenericMessage(const std::string & type);
  bool validate(const GenericProperty & property) const;
  YAML::Node deserialize(const rclcpp::SerializedMessage & serialized) const;

private:
  RosIntrospection introspection_;
  RosSerialization serialization_;
};

}  // namespace generic_type_utility

#endif  // GENERIC_TYPE_UTILITY__GENERIC_MESSAGE_HPP_

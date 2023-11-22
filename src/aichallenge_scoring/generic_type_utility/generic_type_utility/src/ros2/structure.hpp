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

#ifndef ROS2__STRUCTURE_HPP_
#define ROS2__STRUCTURE_HPP_

#include "generic_type_utility/generic_property.hpp"
#include <rosidl_typesupport_introspection_cpp/message_introspection.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace generic_type_utility
{

class RosTypeNode
{
public:
  using IntrospectionClass = rosidl_typesupport_introspection_cpp::MessageMembers;
  using IntrospectionField = rosidl_typesupport_introspection_cpp::MessageMember;
  RosTypeNode(const IntrospectionField * field, const rosidl_message_type_support_t * handle);
  const IntrospectionClass * get_introspection_class();
  const std::string get_class_name();
  const std::string get_field_name();
  bool validate(const GenericProperty::Iterator & property) const;
  YAML::Node make_yaml(const void * memory);

private:
  bool validate_field(const GenericProperty::Iterator & property) const;
  bool validate_index(const GenericProperty::Iterator & property) const;
  YAML::Node make_yaml_class(const void * memory);
  YAML::Node make_yaml_array(const void * memory);
  YAML::Node make_yaml_value(const void * memory);

  const IntrospectionField * field_;
  const IntrospectionClass * klass_;
  std::vector<std::unique_ptr<RosTypeNode>> nodes_;
  std::unordered_map<std::string, RosTypeNode *> nodes_map_;
};

}  // namespace generic_type_utility

#endif  // ROS2__STRUCTURE_HPP_

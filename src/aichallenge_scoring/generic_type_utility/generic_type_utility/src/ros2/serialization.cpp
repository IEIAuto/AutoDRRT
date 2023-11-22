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

#include "generic_type_utility/ros2/serialization.hpp"
#include <rclcpp/serialization.hpp>
#include <rclcpp/typesupport_helpers.hpp>

namespace generic_type_utility
{

class RosSerialization::Impl
{
public:
  explicit Impl(const std::string & type_name);

  std::shared_ptr<rcpputils::SharedLibrary> library_;
  std::unique_ptr<rclcpp::SerializationBase> serialization_;
};

RosSerialization::Impl::Impl(const std::string & type_name)
{
  constexpr char identifier[] = "rosidl_typesupport_cpp";
  library_ = rclcpp::get_typesupport_library(type_name, identifier);

  const auto handle = rclcpp::get_typesupport_handle(type_name, identifier, *library_);
  serialization_ = std::make_unique<rclcpp::SerializationBase>(handle);
}

RosSerialization::RosSerialization(const std::string & type_name)
{
  impl_ = std::make_unique<Impl>(type_name);
}

RosSerialization::~RosSerialization()
{
}

void RosSerialization::deserialize(const rclcpp::SerializedMessage & serialized, RosMessage & message) const
{
  impl_->serialization_->deserialize_message(&serialized, message.memory.get());
}

}  // namespace generic_type_utility

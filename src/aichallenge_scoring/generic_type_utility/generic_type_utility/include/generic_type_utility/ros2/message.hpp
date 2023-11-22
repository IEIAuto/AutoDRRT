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

#ifndef GENERIC_TYPE_UTILITY__ROS2__MESSAGE_HPP_
#define GENERIC_TYPE_UTILITY__ROS2__MESSAGE_HPP_

#include <memory>

// clang-format off
namespace rclcpp { class SerializedMessage; }
namespace rcpputils { class SharedLibrary; }
// clang-format on

namespace generic_type_utility
{

class RosMessageDeleter final
{
public:
  using FunctionPointer = void (*)(void *);
  using FunctionLibrary = std::shared_ptr<rcpputils::SharedLibrary>;
  RosMessageDeleter();
  RosMessageDeleter(FunctionPointer function, FunctionLibrary library);
  void operator()(void * memory) const noexcept;
  static void * Allocate(size_t size);

private:
  FunctionPointer function_;
  FunctionLibrary library_;  // To keep the delete function loaded.
};

struct RosMessage final
{
  std::unique_ptr<void, RosMessageDeleter> memory;
};

}  // namespace generic_type_utility

#endif  // GENERIC_TYPE_UTILITY__ROS2__MESSAGE_HPP_

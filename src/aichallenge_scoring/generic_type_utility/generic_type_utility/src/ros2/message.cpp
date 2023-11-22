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

#include "generic_type_utility/ros2/message.hpp"
#include <cstdlib>

namespace generic_type_utility
{

RosMessageDeleter::RosMessageDeleter()
{
  function_ = nullptr;
  library_ = nullptr;
}

RosMessageDeleter::RosMessageDeleter(FunctionPointer function, FunctionLibrary library)
{
  function_ = function;
  library_ = library;
}

void RosMessageDeleter::operator()(void * memory) const noexcept
{
  function_(memory);
  std::free(memory);
}

void * RosMessageDeleter::Allocate(size_t size)
{
  return std::malloc(size);
}

}  // namespace generic_type_utility

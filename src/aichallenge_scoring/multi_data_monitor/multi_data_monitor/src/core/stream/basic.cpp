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

#include "basic.hpp"
#include "common/exceptions.hpp"

namespace multi_data_monitor
{

void BasicStream::connect([[maybe_unused]] Stream stream)
{
  throw ConfigError("connect to input stream");
}

void InOutStream::connect(Stream stream)
{
  outputs_.push_back(stream);
}

void InOutStream::outputs(const Packet & packet)
{
  for (Stream & output : outputs_)
  {
    output->message(packet);
  }
}

}  // namespace multi_data_monitor

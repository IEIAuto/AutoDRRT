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

#ifndef CORE__STREAM__BASIC_HPP_
#define CORE__STREAM__BASIC_HPP_

#include "common/typedef.hpp"
#include <multi_data_monitor/packet.hpp>
#include <memory>
#include <vector>

namespace multi_data_monitor
{

struct BasicStream
{
public:
  // TODO(Takagi, Isamu): Remove debug code.
  // virtual ~BasicStream() = default;
  static inline int created = 0;
  static inline int removed = 0;
  BasicStream() { ++created; }
  virtual ~BasicStream() { ++removed; }

  virtual void setting(YAML::Node yaml) = 0;
  virtual void connect(Stream stream);
  virtual void message(const Packet & packet) = 0;
};

struct InOutStream : public BasicStream
{
public:
  void connect(Stream stream) override;

protected:
  void outputs(const Packet & packet);

private:
  std::vector<Stream> outputs_;
};

}  // namespace multi_data_monitor

#endif  // CORE__STREAM__BASIC_HPP_

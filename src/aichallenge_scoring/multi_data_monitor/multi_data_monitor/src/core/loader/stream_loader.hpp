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

#ifndef CORE__LOADER__STREAM_LOADER_HPP_
#define CORE__LOADER__STREAM_LOADER_HPP_

#include "common/typedef.hpp"
#include <memory>
#include <vector>

namespace multi_data_monitor
{

class TopicStream;

class StreamLoader final
{
public:
  StreamLoader();
  StreamMaps create(const StreamList & configs, const FilterMaps & filters);
  StreamMaps create(const StreamList & configs, const FilterMaps & filters, const WidgetMaps & widgets);
  void release();
  const auto & topics() { return topics_; }

private:
  std::vector<std::shared_ptr<BasicStream>> streams_;
  std::vector<std::shared_ptr<TopicStream>> topics_;
};

}  // namespace multi_data_monitor

#endif  // CORE__LOADER__STREAM_LOADER_HPP_

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

#ifndef CORE__COMMON__CONTAINER_HPP_
#define CORE__COMMON__CONTAINER_HPP_

#include <unordered_map>
#include <vector>

namespace multi_data_monitor
{

template <class K, class V>
class HashMap : private std::unordered_map<K, V>
{
public:
  using std::unordered_map<K, V>::operator[];
  using std::unordered_map<K, V>::at;
  using std::unordered_map<K, V>::begin;
  using std::unordered_map<K, V>::end;

  std::vector<V> at(const std::vector<K> & keys) const
  {
    std::vector<V> values;
    for (const auto & key : keys)
    {
      values.push_back(this->at(key));
    }
    return values;
  }

  V get(const K & key, const V & value) const
  {
    // Need this pointer for count function.
    return this->count(key) ? this->at(key) : value;
  }
};

}  // namespace multi_data_monitor

#endif  // CORE__COMMON__CONTAINER_HPP_

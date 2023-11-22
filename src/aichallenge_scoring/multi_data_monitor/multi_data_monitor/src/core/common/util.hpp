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

#ifndef CORE__COMMON__UTIL_HPP_
#define CORE__COMMON__UTIL_HPP_

#include <algorithm>
#include <string>
#include <vector>

namespace multi_data_monitor::util
{

template <class Iterable>
std::string join(const Iterable & iterable, const std::string & delimiter = ", ")
{
  const auto head = iterable.begin();
  const auto tail = iterable.end();
  std::string result;
  for (auto iter = head; iter != tail; ++iter)
  {
    result += (iter != head ? delimiter : "") + (*iter);
  }
  return result;
}

template <class T>
std::vector<T> filter(const std::vector<T> & input, const std::function<bool(const T &)> function)
{
  std::vector<T> result;
  for (const auto & data : input)
  {
    if (function(data)) result.push_back(data);
  }
  return result;
}

template <class R, class Iterable, class Function>
std::vector<R> map(const Iterable & input, Function && function)
{
  std::vector<R> result;
  std::transform(std::begin(input), std::end(input), std::back_inserter(result), function);
  return result;
}

}  // namespace multi_data_monitor::util

#endif  // CORE__COMMON__UTIL_HPP_

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

#ifndef CORE__COMMON__GRAPH_HPP_
#define CORE__COMMON__GRAPH_HPP_

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace multi_data_monitor::graph
{

template <class T>
using Nodes = std::vector<T>;

template <class T>
using Graph = std::unordered_map<T, std::vector<T>>;

template <class T>
bool is_tree(const Graph<T> & graph)
{
  std::unordered_map<T, size_t> degrees;
  std::vector<T> nodes;
  for (const auto & [node, links] : graph)
  {
    for (const auto & link : links)
    {
      ++degrees[link];
    }
    nodes.push_back(node);
  }

  std::unordered_map<size_t, size_t> histogram;
  for (const auto & node : nodes)
  {
    ++histogram[degrees[node]];
  }
  return (histogram[0] == 1) && (histogram[1] + 1 == nodes.size());
}

template <class T>
Nodes<T> topological_sort(const Graph<T> & graph)
{
  std::unordered_map<T, int> degrees;
  std::vector<T> nodes;
  std::vector<T> buffer;
  std::vector<T> result;

  for (const auto & [node, links] : graph)
  {
    for (const auto & link : links)
    {
      ++degrees[link];
    }
    nodes.push_back(node);
  }

  for (const auto & node : nodes)
  {
    if (degrees[node] == 0)
    {
      buffer.push_back(node);
    }
  }

  while (!buffer.empty())
  {
    const auto node = buffer.back();
    buffer.pop_back();
    for (const auto & link : graph.at(node))
    {
      if (--degrees[link] == 0)
      {
        buffer.push_back(link);
      }
    }
    result.push_back(node);
  }

  std::reverse(result.begin(), result.end());
  return result;
}

}  // namespace multi_data_monitor::graph

#endif  // CORE__COMMON__GRAPH_HPP_

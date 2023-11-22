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

#include "resolve_relation.hpp"
#include "common/exceptions.hpp"
#include "common/graph.hpp"
#include "common/util.hpp"
#include "common/yaml.hpp"
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace multi_data_monitor
{

template <class T>
void connect_labels(const std::vector<T> & nodes)
{
  std::unordered_map<std::string, T> labels;
  for (const auto & node : nodes)
  {
    const auto & label = node->label;
    if (!label.empty())
    {
      if (labels.count(label))
      {
        throw ConfigError::LabelConflict(label, T::element_type::TypeName);
      }
      labels[label] = node;
    }
  }
  for (const auto & node : nodes)
  {
    if (node->klass == builtin::relay)
    {
      const auto refer = yaml::take_required(node->yaml, "refer").template as<std::string>();
      if (labels.count(refer) == 0)
      {
        throw ConfigError::LabelNotFound(refer, T::element_type::TypeName);
      }
      node->refer = labels.at(refer);
    }
  }
}

template <class T>
T resolve_node(const T & node, std::unordered_set<T> & visit)
{
  if (!node->refer)
  {
    return node;
  }
  if (visit.count(node))
  {
    // TODO(Takagi, Isamu): TypeName and LabelNames
    throw LabelCirculation("label loop is detected");
  }
  visit.insert(node);
  node->refer = resolve_node(node->refer, visit);
  return node->refer;
}

template <class T>
void resolve_labels(const std::vector<T> & nodes)
{
  for (const auto & node : nodes)
  {
    for (auto & item : node->items)
    {
      std::unordered_set<T> visit;
      item = resolve_node(item, visit);
    }
  }
}

template <class T>
void resolve_filter(const std::vector<T> & nodes)
{
  for (const auto & node : nodes)
  {
    if (node->apply)
    {
      std::unordered_set<FilterLink> visit;
      node->apply = resolve_node(node->apply, visit);
    }
  }
}

template <class T>
std::vector<T> filter_unused_nodes(const std::vector<T> & nodes, const std::unordered_set<std::string> & targets)
{
  std::unordered_set<T> used;
  for (const auto & node : nodes)
  {
    for (const auto & item : node->items)
    {
      used.insert(item);
    }
  }

  std::vector<T> result;
  for (const auto & node : nodes)
  {
    if (targets.count(node->klass))
    {
      if (used.count(node) == 0) continue;
      // TODO(Takagi, Isamu): TypeName
      throw LogicError("ReleaseRelation: unintended reverse reference");
    }
    result.push_back(node);
  }
  return result;
}

template <class T>
std::vector<T> normalize_graph(const std::vector<T> & nodes, bool tree)
{
  graph::Graph<T> graph;
  for (const auto & node : nodes)
  {
    graph[node] = node->items;
  }

  const auto result = graph::topological_sort(graph);
  if (result.size() != graph.size())
  {
    // TODO(Takagi, Isamu): TypeName
    throw GraphCirculation("graph loop is detected");
  }
  if (tree && !graph::is_tree(graph))
  {
    // TODO(Takagi, Isamu): TypeName
    throw GraphIsNotTree("graph is not tree");
  }
  return result;
}

ConfigData ConnectRelation::execute(const ConfigData & input)
{
  connect_labels(input.streams);
  connect_labels(input.filters);
  connect_labels(input.widgets);
  return input;
}

ConfigData ResolveRelation::execute(const ConfigData & input)
{
  resolve_filter(input.streams);
  resolve_labels(input.streams);
  resolve_labels(input.filters);
  resolve_labels(input.widgets);
  return input;
}

ConfigData ReleaseRelation::execute(const ConfigData & input)
{
  ConfigData output = input;
  output.streams = filter_unused_nodes(input.streams, {builtin::relay, builtin::subscription});
  output.filters = filter_unused_nodes(input.filters, {builtin::relay});
  output.widgets = filter_unused_nodes(input.widgets, {builtin::relay});
  return output;
}

ConfigData NormalizeRelation::execute(const ConfigData & input)
{
  ConfigData output = input;
  output.streams = normalize_graph(input.streams, false);
  output.filters = normalize_graph(input.filters, false);
  output.widgets = normalize_graph(input.widgets, true);
  return output;
}

}  // namespace multi_data_monitor

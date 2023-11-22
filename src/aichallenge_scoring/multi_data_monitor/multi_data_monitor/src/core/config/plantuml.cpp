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

#include "plantuml.hpp"
#include <fstream>
#include <sstream>
#include <vector>

namespace multi_data_monitor::plantuml
{

template <class T>
void dump_node(std::ostringstream & ss, const T & data, const std::string & color)
{
  const auto label = data->label.empty() ? "" : " [" + data->label + "]";
  ss << "card " << data << " " << color << " [" << std::endl;
  ss << data->klass << label << std::endl;
  if (data->yaml.size())
  {
    ss << "---" << std::endl;
    ss << YAML::Dump(data->yaml) << std::endl;
  }
  ss << "]" << std::endl;
}

std::string Diagram::convert(const ConfigData & data) const
{
  std::ostringstream ss;
  ss << "@startuml debug" << std::endl;

  for (const auto & stream : data.streams)
  {
    dump_node(ss, stream, "#AAFFFF");
  }
  for (const auto & filter : data.filters)
  {
    dump_node(ss, filter, "#FFFFAA");
  }
  for (const auto & widget : data.widgets)
  {
    dump_node(ss, widget, "#FFAAFF");
  }

  for (const auto & stream : data.streams)
  {
    if (stream->panel)
    {
      ss << stream->panel << " --> " << stream << std::endl;
    }
    if (stream->apply)
    {
      ss << stream << " --> " << stream->apply << std::endl;
    }
    if (stream->refer)
    {
      ss << stream << " --> " << stream->refer << " #line.dashed" << std::endl;
    }
    for (const auto & item : stream->items)
    {
      ss << stream << " --> " << item << std::endl;
    }
  }

  for (const auto & filter : data.filters)
  {
    if (filter->refer)
    {
      ss << filter << " --> " << filter->refer << " #line.dashed" << std::endl;
    }
    for (const auto & item : filter->items)
    {
      ss << filter << " --> " << item << std::endl;
    }
  }

  for (const auto & widget : data.widgets)
  {
    if (widget->refer)
    {
      ss << widget << " --> " << widget->refer << " #line.dashed" << std::endl;
    }
    for (const auto & item : widget->items)
    {
      ss << widget << " --> " << item << std::endl;
    }
  }

  ss << "@enduml" << std::endl;
  return ss.str();
}

void Diagram::write(const ConfigData & data, const std::string & path) const
{
  std::ofstream ofs(path);
  ofs << convert(data) << std::endl;
}

}  // namespace multi_data_monitor::plantuml

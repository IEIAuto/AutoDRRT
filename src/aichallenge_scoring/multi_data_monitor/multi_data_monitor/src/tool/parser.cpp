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

#include "config/plantuml.hpp"
#include "loader/config_loader.hpp"
#include "loader/stream_loader.hpp"
#include <iostream>
using namespace multi_data_monitor;  // NOLINT

ConfigData load(const std::string & path, const std::set<std::string> & option)
{
  const auto step_hook = [](int step, const std::string & name, const ConfigData & data)
  {
    const auto diagram = plantuml::Diagram();
    const auto filename = std::to_string(step) + "-" + name;
    const auto filepath = "step" + filename + ".plantuml";
    std::cout << "write " << filepath << std::endl;
    diagram.write(data, filepath);
  };

  const auto last_hook = [](int, const std::string &, const ConfigData & data)
  {
    const auto diagram = plantuml::Diagram();
    const auto filepath = "multi-data-monitor.plantuml";
    std::cout << "write " << filepath << std::endl;
    diagram.write(data, filepath);
  };

  ConfigLoader loader;
  if (option.count("--plantuml"))
  {
    loader.set_last_hook(last_hook);
  }
  if (option.count("--plantuml-all"))
  {
    loader.set_step_hook(step_hook);
  }
  return loader.execute(path);
}

int main(int argc, char ** argv)
{
  if (argc < 3)
  {
    std::cerr << "usage: command <scheme> <path> [options]" << std::endl;
    return 1;
  }

  const auto scheme = std::string(argv[1]);
  const auto config = std::string(argv[2]);
  const auto option = std::set<std::string>(argv + 3, argv + argc);
  load(scheme + "://" + config, option);
  std::cout << "Config file parsed." << std::endl;
}

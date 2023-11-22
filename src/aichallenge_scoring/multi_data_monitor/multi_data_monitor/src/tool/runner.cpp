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

#include "loader/config_loader.hpp"
#include "runner/widget_runner.hpp"
#include <rclcpp/rclcpp.hpp>
#include <iostream>
using namespace multi_data_monitor;  // NOLINT

int main(int argc, char ** argv)
{
  if (argc != 3)
  {
    std::cerr << "usage: command <scheme> <config-file-path>" << std::endl;
    return 1;
  }

  StreamRunner runner;
  {
    const auto scheme = std::string(argv[1]);
    const auto config = std::string(argv[2]);
    runner.create(ConfigLoader::Execute(scheme + "://" + config));
  }

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("runner");
  runner.start(node);
  {
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    executor.remove_node(node);
  }
  runner.shutdown();
  rclcpp::shutdown();
}

// Copyright 2015-2019 Autoware Foundation
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

#include "ndt_scan_matcher/ndt_scan_matcher_core.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto ndt_scan_matcher = std::make_shared<NDTScanMatcher>();
  rclcpp::executors::MultiThreadedExecutor exec;
  //  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(ndt_scan_matcher);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}

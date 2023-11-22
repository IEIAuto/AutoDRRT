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

#include "common/exceptions.hpp"
#include "loader/config_loader.hpp"
#include <gtest/gtest.h>

std::string resource(const std::string & path)
{
  static const std::string scheme = "file://";
  return scheme + TEST_YAML + path;
}

void load_config(const std::string & path)
{
  multi_data_monitor::ConfigLoader::Execute(resource(path));
}

TEST(GraphStructure, StreamLabelCirculation)
{
  EXPECT_THROW(load_config("stream-label-circulation.yaml"), multi_data_monitor::LabelCirculation);
}

TEST(GraphStructure, StreamGraphCirculation)
{
  EXPECT_THROW(load_config("stream-graph-circulation.yaml"), multi_data_monitor::GraphCirculation);
}

TEST(GraphStructure, WidgetLabelCirculation)
{
  EXPECT_THROW(load_config("widget-label-circulation.yaml"), multi_data_monitor::LabelCirculation);
}

TEST(GraphStructure, WidgetGraphCirculation)
{
  EXPECT_THROW(load_config("widget-graph-circulation.yaml"), multi_data_monitor::GraphCirculation);
}

TEST(GraphStructure, WidgetGraphItNotTree)
{
  EXPECT_THROW(load_config("widget-graph-is-not-tree.yaml"), multi_data_monitor::GraphIsNotTree);
}

class RclcppEnvironment : public testing::Environment
{
public:
  RclcppEnvironment(int argc, char ** argv)
  {
    argc = argc;
    argv = argv;
  }
  void SetUp() override
  {
    // rclcpp::init();
  }
  void TearDown() override
  {
    // rclcpp::shutdown();
  }

private:
  int argc;
  char ** argv;
};

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  // testing::AddGlobalTestEnvironment();
  const auto result = RUN_ALL_TESTS();
  return result;
}

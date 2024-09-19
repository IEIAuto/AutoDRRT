// Copyright 2021 Apex.AI, Inc.
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

/// \copyright Copyright 2021 Apex.AI, Inc.
/// All rights reserved.

#include <point_cloud_msg_wrapper/default_field_generators.hpp>
#include <point_cloud_msg_wrapper/field_generators.hpp>

#include <gtest/gtest.h>

#include <string>
#include <algorithm>

namespace
{
struct PointXI
{
  std::int64_t id;
  float x;
};

}  // namespace

TEST(FieldGeneratorsTest, init) {
  sensor_msgs::msg::PointCloud2::_fields_type fields;
  point_cloud_msg_wrapper::field_y_generator::push_back_field_if_needed<PointXI>(fields);
  ASSERT_TRUE(fields.empty());
  point_cloud_msg_wrapper::field_x_generator::push_back_field_if_needed<PointXI>(fields);
  ASSERT_EQ(1UL, fields.size());
  EXPECT_EQ("x", fields.back().name);
  EXPECT_EQ(sensor_msgs::msg::PointField::FLOAT32, fields.back().datatype);
  EXPECT_EQ(1U, fields.back().count);
  EXPECT_EQ(8U, fields.back().offset);

  point_cloud_msg_wrapper::field_id_generator::push_back_field_if_needed<PointXI>(fields);
  ASSERT_EQ(2UL, fields.size());
  EXPECT_EQ("id", fields.back().name);
  EXPECT_EQ(sensor_msgs::msg::PointField::INT32, fields.back().datatype);
  EXPECT_EQ(2U, fields.back().count);
  EXPECT_EQ(0U, fields.back().offset);
}

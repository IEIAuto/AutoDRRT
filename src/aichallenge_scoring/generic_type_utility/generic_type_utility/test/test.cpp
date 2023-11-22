// Copyright 2021 Takagi, Isamu
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

#include "generic_type_utility/generic_message.hpp"
#include <rclcpp/rclcpp.hpp>
#include <gtest/gtest.h>
using namespace generic_type_utility;  // NOLINT(build/namespaces)

template <class T>
rclcpp::SerializedMessage create_serialized(const T & data)
{
  rclcpp::SerializedMessage serialized;
  rcutils_uint8_array_t & buffer = serialized.get_rcl_serialized_message();
  serialized.reserve(data.size());
  buffer.buffer_length = data.size();
  std::copy(data.begin(), data.end(), buffer.buffer);
  return serialized;
}

TEST(generic_type_utility, test1)
{
  // clang-format off
  const auto data = std::array<uint8_t, 28> ({
    0x00, 0x01, 0x00, 0x00,
    0x30, 0x41, 0xab, 0x00,
    0x85, 0xe1, 0xa7, 0x02,
    0x05, 0x00, 0x00, 0x00,
    0x54, 0x65, 0x73, 0x74,
    0x00, 0x00, 0x00, 0x00
  });
  // clang-format on

  const auto support = GenericMessage("std_msgs/msg/Header");
  const auto p1 = GenericProperty("stamp.sec");
  const auto p2 = GenericProperty("stamp.nanosec");
  const auto p3 = GenericProperty("frame_id");

  const auto yaml = support.deserialize(create_serialized(data));
  EXPECT_EQ(p1.apply(yaml).as<int32_t>(), 11223344);
  EXPECT_EQ(p2.apply(yaml).as<uint32_t>(), 44556677u);
  EXPECT_EQ(p3.apply(yaml).as<std::string>(), "Test");
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

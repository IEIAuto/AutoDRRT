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

#include "generic_type_utility/generic_message.hpp"
#include <rclcpp/rclcpp.hpp>
#include <iostream>

using generic_type_utility::GenericMessage;
using generic_type_utility::GenericProperty;

class SampleNode : public rclcpp::Node
{
public:
  SampleNode() : rclcpp::Node("generic_type_utility")
  {
    const std::string type_name = "std_msgs/msg/Header";
    message_ = std::make_unique<GenericMessage>(type_name);
    property_ = std::make_unique<GenericProperty>("stamp.sec");

    const auto callback = [this](const std::shared_ptr<rclcpp::SerializedMessage> serialized)
    {
      const auto yaml = message_->deserialize(*serialized);
      const auto node = property_->apply(yaml);
      std::cout << node.as<int>() << std::endl;
    };
    sub_ = create_generic_subscription("/message", type_name, rclcpp::QoS(1), callback);
  }

private:
  std::unique_ptr<GenericMessage> message_;
  std::unique_ptr<GenericProperty> property_;
  rclcpp::GenericSubscription::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SampleNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
}

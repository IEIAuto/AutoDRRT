// Copyright 2021 TIER IV, Inc.
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

#include "demo_node.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string>
#include <pthread.h>

#include <nav_msgs/msg/odometry.hpp>
#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>

#include <typeinfo>
#include <cxxabi.h>
#include "dma_adapter.hpp"
#include <chrono>

namespace dma_transfer
{    
    
    //void string_callback(const std_msgs::msg::String& msg)
    void string_callback(const sensor_msgs::msg::Image& msg)
    {
        
        std::cout << "DMA MESSAGE RECEIVED!\n"  << std::endl;
        
    }

    DemoNode2::DemoNode2(const rclcpp::NodeOptions &options)
        : Node("demo_node_recv", options)
    {
        //const uintptr_t addr_ = 0x2b28000000;
	const uintptr_t addr_ = 0x1571f6000;
        dma_trans.set_physical_address(addr_);
        // dma_trans.init_fd();
        
        rclcpp::QoS qos(rclcpp::KeepLast(10));
        qos.reliable();
        qos.deadline(rclcpp::Duration(1, 0));
        dma_trans.create_subscription_via_dma(this, "/test_dma", string_callback);

        // sub_test_ = this->create_subscription<std_msgs::msg::String>("/test_dds", 1,
        //     [this](const std_msgs::msg::String::ConstSharedPtr msg) {
        //         rclcpp::Time msgTimeStamp = msg->stamp;
        //         rclcpp::Time currentTimeStamp = this->now();
        //         rclcpp::Duration timeDiff = currentTimeStamp - msgTimeStamp;
        //         std::cout << "==dds trans time is :" << std::to_string(timeDiff.nanoseconds()/1000) << " us==" << std::endl;
        //     }
        // );
    }

} // namespace dma_transfer

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(dma_transfer::DemoNode2)

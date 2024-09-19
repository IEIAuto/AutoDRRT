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

#include "opencv2/opencv.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>

#include <typeinfo>
#include <cxxabi.h>
#include "dma_adapter.hpp"
#include <chrono>

namespace dma_transfer
{

    // void serialization_test(const std_msgs::msg::String::ConstSharedPtr msg)
    // {       

    //     std::vector<uint8_t> serialized_data;
    //     DmaAdapter<std_msgs::msg::String>::serialize(*msg, serialized_data);

    //     printf("send data:\n");
    //     for (int i = 0; i < serialized_data.size(); i ++)
    //     {
    //         printf("%d", serialized_data[i]);
    //     }

    //     DmaAdapter<std_msgs::msg::String>::dma_write(serialized_data);
    //     DmaAdapter<std_msgs::msg::String>::dma_read(serialized_data.size());
        
    //     std_msgs::msg::String received_data;
    //     DmaAdapter<std_msgs::msg::String>::deserialize(serialized_data,received_data);
    //     printf("received data: %s\n", received_data.data.c_str());
       
    // }
    

    DemoNode1::DemoNode1(const rclcpp::NodeOptions &options)
        : Node("demo_node", options)
    {

        rclcpp::QoS qos(rclcpp::KeepLast(10));
        qos.reliable();
        qos.deadline(rclcpp::Duration(1, 0));
        count = 0;
        // dma_trans.create_subscription_via_dma(this,"/test_dma", string_callback);
        const uintptr_t addr_ = 0x2b28000000;
        dma_trans.set_physical_address(addr_);
        // sub_string = this->create_subscription<std_msgs::msg::String>("/chatter", rclcpp::QoS(1),
        
        // [this](const std_msgs::msg::String::ConstSharedPtr msg) {
        //     //code
        //     std::cout << "I heard sth. from string" << std::endl;
        //     // serialization_test(msg);
            
        //     dma_trans.publish_via_dma(this, *msg, "/test_dma");

        // });
 
        // pub_test_ = this->create_publisher<std_msgs::msg::String>("/test_dds", rclcpp::QoS(1));

	cv::Mat cv_image = cv::imread("./dump/test.jpg", cv::IMREAD_COLOR);
        if (cv_image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load image");
            return;
        }
        auto image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cv_image).toImageMsg();

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000),
            [this, image_msg]() {
            int desiredLength = 4000;
            auto msg = std_msgs::msg::String();
            std::string str(desiredLength, 'A');
            msg.data = "I send the " + std::to_string(count++) + "th messages " + str;
            std::cout << "I send the " << std::to_string(count - 1) << "th messages " << std::endl;
            //dma_trans.publish_via_dma(this, msg, "/test_dma");
            dma_trans.publish_via_dma(this, *image_msg, "/test_dma");
            // msg.stamp = this->now();
            // pub_test_->publish(msg);
        }
        );
    }

} // namespace dma_transfer

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(dma_transfer::DemoNode1)

// Copyright 2016 Open Source Robotics Foundation, Inc.
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

#include "composition/listener_component.hpp"

#include <iostream>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "composition/gpu_share.hpp"

namespace gpudirect
{

// Create a Listener "component" that subclasses the generic rclcpp::Node base class.
// Components get built into shared libraries and as such do not write their own main functions.
// The process using the component's shared library will instantiate the class as a ROS node.
Listener::Listener(const rclcpp::NodeOptions & options)
: Node("listener", options)
{
  // Create a callback function for when messages are received.
  // Variations of this function also exist using, for example, UniquePtr for zero-copy transport.
  
  
  auto callback =
    [this](std_msgs::msg::String::ConstSharedPtr msg) -> void
    {

      auto start_time = std::chrono::high_resolution_clock::now();

      RCLCPP_INFO(this->get_logger(), "I heard: [%s]", msg->data.c_str());
      std::flush(std::cout);
    
      void *gpu_ptr = SharedGPUMemory::get_instance().get_memory();
      
      if (gpu_ptr != nullptr) {
        RCLCPP_INFO(this->get_logger(), "Node B accessed shared GPU memory.");

        // 创建主机内存来接收 GPU 内存中的数据
        
        int data_size = 8*1024*1024;
        int *host_data = new int[data_size];

        // 从 GPU 内存拷贝数据到主机
        cudaMemcpy(host_data, gpu_ptr, data_size, cudaMemcpyDeviceToHost);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        RCLCPP_INFO(rclcpp::get_logger("======cost read time======"), "duration.count(): %lld us", static_cast<long long>(duration.count()));

        // 打印从 GPU 共享内存中读取的数据
        // for (int i = 0; i < 10; i++) {
        //   RCLCPP_INFO(this->get_logger(), "Data at index %d: %d", i, host_data[i]);
        // }
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to access shared GPU memory.");
      }


    };

  // Create a subscription to the "chatter" topic which can be matched with one or more
  // compatible ROS publishers.
  // Note that not all publishers on the same topic with the same type will be compatible:
  // they must have compatible Quality of Service policies.
  sub_ = create_subscription<std_msgs::msg::String>("chatter", 10, callback);
}

}  

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(gpudirect::Listener)

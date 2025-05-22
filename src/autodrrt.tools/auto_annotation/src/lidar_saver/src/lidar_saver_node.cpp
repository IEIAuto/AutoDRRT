/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */



#include <iostream>
#include <vector>
#include <string>



#include <chrono>
#include <functional>
#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/header.hpp"
#include <fstream>
#include <ctime>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

class LidarSaver : public rclcpp::Node
{
public:
    ~LidarSaver(){
    }
    LidarSaver() : Node("time_cal")
    { 
        // RCLCPP_INFO(this->get_logger(), "LidarSaver started");
        // image_pub_ = this->create_publisher<std_msgs::msg::Header>("/time_cal/images_info", rclcpp::QoS(1));
        // pcl_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/time_cal/pcl", rclcpp::QoS(1));
        // objects_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>("/time_cal/objects", rclcpp::QoS(1));
        // camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/time_cal/camera_info", rclcpp::QoS(1));
        // raw_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/time_cal/image_raw", rclcpp::QoS(1));
        lidar_info_pub_ = this->create_publisher<std_msgs::msg::Header>("/message_sync/out/pointcloud_info", rclcpp::SensorDataQoS().reliable());


        /* save to bin */
        // pcl_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        //     "/message_sync/out/pointcloud", rclcpp::SensorDataQoS(),
        //     [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        //         std::srand(std::time(0));
        //         int random_number = std::rand() % 90000 + 10000;
        //         std::string filename = "./lidar_bins/" + msg->header.frame_id + "/" + msg->header.frame_id + "_" + std::to_string(msg->header.stamp.sec) + std::to_string(random_number) + ".bin";
        //         const uint8_t* data_ptr = msg->data.data();
        //         size_t data_size = msg->data.size();
        //         std::ofstream file(filename, std::ios::binary);
        //         file.write(reinterpret_cast<const char*>(data_ptr), data_size);
        //         file.close();
        //         std_msgs::msg::Header tmp_header;
        //         tmp_header.stamp = msg->header.stamp;
        //         tmp_header.frame_id = filename;
        //         lidar_info_pub_->publish(tmp_header);
        // });



     /* save to pcd */
        pcl_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/message_sync/out/pointcloud", rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                std::srand(std::time(0));
                int random_number = std::rand() % 90000 + 10000;
                std::string filename = "./lidar_pcds/" + msg->header.frame_id + "/" + msg->header.frame_id + "_" + std::to_string(msg->header.stamp.sec) + std::to_string(random_number) + ".pcd";
                pcl::PointCloud<pcl::PointXYZI> cloud;
                pcl::fromROSMsg(*msg, cloud);
                pcl::io::savePCDFileASCII(filename,cloud);
                std_msgs::msg::Header tmp_header;
                tmp_header.stamp = msg->header.stamp;
                tmp_header.frame_id = filename;
                lidar_info_pub_->publish(tmp_header);
        });


    }

private:
    rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr lidar_info_pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_sub_;
};


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    // std::cout << " start " << std::endl;
    rclcpp::spin(std::make_shared<LidarSaver>());
    rclcpp::shutdown();
    return 0;
}
// #include "rclcpp_components/register_node_macro.hpp"
// RCLCPP_COMPONENTS_REGISTER_NODE(LidarSaver)

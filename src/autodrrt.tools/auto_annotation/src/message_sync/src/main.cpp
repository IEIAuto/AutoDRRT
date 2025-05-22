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
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/header.hpp"
/***********************************swpld added**********************************/
#include "sensor_msgs/msg/image.hpp"



#include "sensor_msgs/msg/point_cloud2.hpp"

#include "sensor_msgs/msg/camera_info.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"

#include <autoware_auto_perception_msgs/msg/detected_object_kinematics.hpp>
#include <autoware_auto_perception_msgs/msg/detected_objects.hpp>
#include <autoware_auto_perception_msgs/msg/object_classification.hpp>
#include <autoware_auto_perception_msgs/msg/shape.hpp>

#include <algorithm>
#include <dirent.h>
#include <mutex>



using Label = autoware_auto_perception_msgs::msg::ObjectClassification;

class MessageSync : public rclcpp::Node
{
public:
    ~MessageSync(){
    }
    MessageSync() : Node("message_sync")
    { 
        // RCLCPP_INFO(this->get_logger(), "MessageSync started");
        // image_pub_ = this->create_publisher<std_msgs::msg::Header>("/message_sync/images_info", rclcpp::QoS(1));
        pcl_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/message_sync/pcl", rclcpp::QoS(1));
        objects_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>("/message_sync/objects", rclcpp::QoS(1));
        camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/message_sync/camera_info", rclcpp::QoS(1));
        raw_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/message_sync/image_raw", rclcpp::QoS(1));


        using std::placeholders::_1;
        using std::placeholders::_2;
        using std::placeholders::_3;
        using std::placeholders::_4;
        using std::placeholders::_5;
        using std::placeholders::_6;
        using std::placeholders::_7;
    

        rmw_qos_profile_t image_rmw_qos(rclcpp::SensorDataQoS().get_rmw_qos_profile());
        image_rmw_qos.depth = 20;

        // image_sub_sync.subscribe(this, "/input/images_info", image_rmw_qos);
        pcl_sub_sync.subscribe(this, "/input/pointcloud", image_rmw_qos);
        obj_sub_sync.subscribe(this, "/input/objects", image_rmw_qos);
        camera_info_sub_sync.subscribe(this, "/input/camera_info", image_rmw_qos);
        raw_image_sub_sync.subscribe(this, "/input/image_raw", image_rmw_qos);

        image_rmw_qos.depth = 5;
        image_rmw_qos.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
        
        // image_rmw_qos.depth = 5;
        // image_rmw_qos.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
        // image_sub_sync[0].subscribe(this, "/front/image_raw_time",image_rmw_qos);
        // image_sub_sync[1].subscribe(this, "/front/image_raw_time",image_rmw_qos);
        // image_sub_sync[2].subscribe(this, "/front/image_raw_time",image_rmw_qos);
        // image_sub_sync[3].subscribe(this, "/front/image_raw_time",image_rmw_qos);
        // image_sub_sync[4].subscribe(this, "/front/image_raw_time",image_rmw_qos);
        // image_sub_sync[5].subscribe(this, "/front/image_raw_time",image_rmw_qos);

 
        
        // sync_stream.reset(new message_filters::Synchronizer<approximate_policy_sync>(approximate_policy_sync(200), image_sub_sync, pcl_sub_sync, obj_sub_sync, camera_info_sub_sync));
 
        // sync_stream->registerCallback(std::bind(&MessageSync::sync_stream_callback, this, _1, _2, _3, _4));
  
        sync_stream.reset(new message_filters::Synchronizer<approximate_policy_sync>(approximate_policy_sync(100), raw_image_sub_sync, camera_info_sub_sync, pcl_sub_sync));
 
        sync_stream->registerCallback(std::bind(&MessageSync::sync_stream_callback, this, _1, _2, _3));
    }

private:

 
    // void sync_stream_callback(const std_msgs::msg::Header::ConstSharedPtr &image, const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pcl,\
    // const autoware_auto_perception_msgs::msg::DetectedObjects::ConstSharedPtr &obj, const sensor_msgs::msg::CameraInfo::ConstSharedPtr &cinfo) {
    void sync_stream_callback(const sensor_msgs::msg::Image::ConstSharedPtr &cimage,\
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr &cinfo, const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pcl) {

        std::cout << "I get sth. from Camera" << std::endl;

        raw_image_pub_->publish(*cimage);
        pcl_pub_->publish(*pcl);
        camera_info_pub_->publish(*cinfo);
        // objects_pub_->publish(*obj);
       
    }

    rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub_;
    rclcpp::Publisher<autoware_auto_perception_msgs::msg::DetectedObjects>::SharedPtr objects_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_image_pub_;


    message_filters::Subscriber<std_msgs::msg::Header> image_sub_sync;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pcl_sub_sync;
    message_filters::Subscriber<autoware_auto_perception_msgs::msg::DetectedObjects> obj_sub_sync;
    message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_sync;
    message_filters::Subscriber<sensor_msgs::msg::Image> raw_image_sub_sync;
    

    // using approximate_policy_sync = message_filters::sync_policies::ApproximateTime<std_msgs::msg::Header, sensor_msgs::msg::PointCloud2, autoware_auto_perception_msgs::msg::DetectedObjects, sensor_msgs::msg::CameraInfo>;
    using approximate_policy_sync = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo, sensor_msgs::msg::PointCloud2>;
   
    typedef message_filters::Synchronizer<approximate_policy_sync> Synchron_sub;
    std::unique_ptr<Synchron_sub> sync_stream;

};


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    // std::cout << " start " << std::endl;
    rclcpp::spin(std::make_shared<MessageSync>());
    rclcpp::shutdown();
    return 0;
}
// #include "rclcpp_components/register_node_macro.hpp"
// RCLCPP_COMPONENTS_REGISTER_NODE(MessageSync)

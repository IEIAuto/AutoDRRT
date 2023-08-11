// Copyright 2020 Tier IV, Inc.
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
/*
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: cropbox.cpp
 *
 */

#include "pointcloud_preprocessor/crop_box_filter/crop_box_filter_nodelet.hpp"
#include "pointcloud_preprocessor/crop_box_filter/crop_box_cuda.hpp"
#include "pointcloud_preprocessor/crop_box_filter/crop_box_filter_opt.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <vector>

namespace pointcloud_preprocessor
{
CropBoxFilterComponent::CropBoxFilterComponent(const rclcpp::NodeOptions & options)
: Filter("CropBoxFilter", options)
{
  // initialize debug tool
  {
    using tier4_autoware_utils::DebugPublisher;
    using tier4_autoware_utils::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ = std::make_unique<DebugPublisher>(this, "crop_box_filter");
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");
  }

  // set initial parameters
  {
    auto & p = param_;
    p.min_x = static_cast<float>(declare_parameter("min_x", -1.0));
    p.min_y = static_cast<float>(declare_parameter("min_y", -1.0));
    p.min_z = static_cast<float>(declare_parameter("min_z", -1.0));
    p.max_x = static_cast<float>(declare_parameter("max_x", 1.0));
    p.max_y = static_cast<float>(declare_parameter("max_y", 1.0));
    p.max_z = static_cast<float>(declare_parameter("max_z", 1.0));
    p.negative = static_cast<float>(declare_parameter("negative", false));
  }

  // set additional publishers
  {
    crop_box_polygon_pub_ =
      this->create_publisher<geometry_msgs::msg::PolygonStamped>("~/crop_box_polygon", 10);
  }

  // set parameter service callback
  {
    using std::placeholders::_1;
    set_param_res_ = this->add_on_set_parameters_callback(
      std::bind(&CropBoxFilterComponent::paramCallback, this, _1));
  }
}

void CropBoxFilterComponent::filter(
  const PointCloud2ConstPtr & input, [[maybe_unused]] const IndicesPtr & indices,
  PointCloud2 & output)
{
  
  crop_box_filter_opt::CropBoxParam pram_trans{};
  pram_trans.min_x = param_.min_x;
  pram_trans.min_y = param_.min_y;
  pram_trans.min_z = param_.min_z;
  pram_trans.max_x = param_.max_x;
  pram_trans.max_y = param_.max_y;
  pram_trans.max_z = param_.max_z;
  pram_trans.negative = param_.negative;
  crop_box_filter_opt::filter(input,indices,output,pram_trans);
   // stop_watch_ptr_->toc("processing_time", true);
        //  std::scoped_lock lock(mutex_);
        // const auto data_size = input->data.size();
        // const auto point_step = input->point_step;
        // output.data.resize(input->data.size());
        // Eigen::Vector3f pt(Eigen::Vector3f::Zero());
        // size_t j = 0;
        // int count =0;
        // // If inside the cropbox
        // if (!param_.negative) {
        //   for (size_t i = 0; i + point_step < data_size; i += point_step) {
        //     memcpy(pt.data(), &input->data[i], sizeof(float) * 3);
        //     // if(i == 0)
        //     // {
        //     //    std::cout << "input_xyz is " << pt.x()<<" "<<pt.y()<< " " <<pt.z() << std::endl;;
        //     // }
        //     if (
        //       param_.min_z < pt.z() && pt.z() < param_.max_z && param_.min_y < pt.y() &&
        //       pt.y() < param_.max_y && param_.min_x < pt.x() && pt.x() < param_.max_x) {
        //       memcpy(&output.data[j], &input->data[i], point_step);
        //       j += point_step;
        //     }
        //     else{
        //       //  std::cout << i << " " << pt.x()<<" "<<pt.y()<< " " <<pt.z() << " "<<param_.negative << std::endl;
        //        count++;
        //     }
        //   }
        //   // If outside the cropbox
        // } else {
        //   for (size_t i = 0; i + point_step < data_size; i += point_step) {
        //     memcpy(pt.data(), &input->data[i], sizeof(float) * 3);
        //     if (
        //       param_.min_z > pt.z() || pt.z() > param_.max_z || param_.min_y > pt.y() ||
        //       pt.y() > param_.max_y || param_.min_x > pt.x() || pt.x() > param_.max_x) {
        //       memcpy(&output.data[j], &input->data[i], point_step);
        //       j += point_step;
        //     }
        //     // else{
        //     //   //  std::cout << i << " " << pt.x()<<" "<<pt.y()<< " " <<pt.z() << " "<<param_.negative<< std::endl;
        //     //    count++;
        //     // }
        //   } 
        // }
        // output.data.resize(j);
        // output.header.frame_id = input->header.frame_id;
        // output.height = 1;
        // output.fields = input->fields;
        // output.is_bigendian = input->is_bigendian;
        // output.point_step = input->point_step;
        // output.is_dense = input->is_dense;
        // output.width = static_cast<uint32_t>(output.data.size() / output.height / output.point_step);
        // output.row_step = static_cast<uint32_t>(output.data.size() / output.height);
  //--------------------------original----------------------------//



  publishCropBoxPolygon();
  // RCLCPP_INFO(get_logger(), "publishCropBoxPolygon time is %f",stop_watch_ptr_->toc("processing_time",true));
  // add processing time for debug
  if (debug_publisher_) {
    const double cyclic_time_ms = stop_watch_ptr_->toc("cyclic_time", true);
    const double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
    debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/cyclic_time_ms", cyclic_time_ms);
    debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/processing_time_ms", processing_time_ms);
  }
 
}

void CropBoxFilterComponent::publishCropBoxPolygon()
{
  auto generatePoint = [](double x, double y, double z) {
    geometry_msgs::msg::Point32 point;
    point.x = x;
    point.y = y;
    point.z = z;
    return point;
  };

  const double x1 = param_.max_x;
  const double x2 = param_.min_x;
  const double x3 = param_.min_x;
  const double x4 = param_.max_x;

  const double y1 = param_.max_y;
  const double y2 = param_.max_y;
  const double y3 = param_.min_y;
  const double y4 = param_.min_y;

  const double z1 = param_.min_z;
  const double z2 = param_.max_z;

  geometry_msgs::msg::PolygonStamped polygon_msg;
  polygon_msg.header.frame_id = tf_input_frame_;
  polygon_msg.header.stamp = get_clock()->now();
  polygon_msg.polygon.points.push_back(generatePoint(x1, y1, z1));
  polygon_msg.polygon.points.push_back(generatePoint(x2, y2, z1));
  polygon_msg.polygon.points.push_back(generatePoint(x3, y3, z1));
  polygon_msg.polygon.points.push_back(generatePoint(x4, y4, z1));
  polygon_msg.polygon.points.push_back(generatePoint(x1, y1, z1));

  polygon_msg.polygon.points.push_back(generatePoint(x1, y1, z2));

  polygon_msg.polygon.points.push_back(generatePoint(x2, y2, z2));
  polygon_msg.polygon.points.push_back(generatePoint(x2, y2, z1));
  polygon_msg.polygon.points.push_back(generatePoint(x2, y2, z2));

  polygon_msg.polygon.points.push_back(generatePoint(x3, y3, z2));
  polygon_msg.polygon.points.push_back(generatePoint(x3, y3, z1));
  polygon_msg.polygon.points.push_back(generatePoint(x3, y3, z2));

  polygon_msg.polygon.points.push_back(generatePoint(x4, y4, z2));
  polygon_msg.polygon.points.push_back(generatePoint(x4, y4, z1));
  polygon_msg.polygon.points.push_back(generatePoint(x4, y4, z2));

  polygon_msg.polygon.points.push_back(generatePoint(x1, y1, z2));

  crop_box_polygon_pub_->publish(polygon_msg);
}

rcl_interfaces::msg::SetParametersResult CropBoxFilterComponent::paramCallback(
  const std::vector<rclcpp::Parameter> & p)
{
  std::scoped_lock lock(mutex_);

  CropBoxParam new_param{};

  if (
    get_param(p, "min_x", new_param.min_x) && get_param(p, "min_y", new_param.min_y) &&
    get_param(p, "min_z", new_param.min_z) && get_param(p, "max_x", new_param.max_x) &&
    get_param(p, "max_y", new_param.max_y) && get_param(p, "max_z", new_param.max_z) &&
    get_param(p, "negative", new_param.negative)) {
    if (
      param_.min_x != new_param.min_x || param_.max_x != new_param.max_x ||
      param_.min_y != new_param.min_y || param_.max_y != new_param.max_y ||
      param_.min_z != new_param.min_z || param_.max_z != new_param.max_z ||
      param_.negative != new_param.negative) {
      RCLCPP_DEBUG(
        get_logger(), "[%s::paramCallback] Setting the minimum point to: %f %f %f.", get_name(),
        new_param.min_x, new_param.min_y, new_param.min_z);
      RCLCPP_DEBUG(
        get_logger(), "[%s::paramCallback] Setting the minimum point to: %f %f %f.", get_name(),
        new_param.max_x, new_param.max_y, new_param.max_z);
      RCLCPP_DEBUG(
        get_logger(), "[%s::paramCallback] Setting the filter negative flag to: %s.", get_name(),
        new_param.negative ? "true" : "false");
      param_ = new_param;
    }
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  return result;
}

}  // namespace pointcloud_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_preprocessor::CropBoxFilterComponent)

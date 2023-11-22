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

#include "pointcloud_preprocessor/passthrough_filter/passthrough_filter_uint16_nodelet.hpp"

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/segment_differences.h>

#include <string>
#include <vector>

#include <pcl/conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

namespace pointcloud_preprocessor
{
PassThroughFilterUInt16Component::PassThroughFilterUInt16Component(
  const rclcpp::NodeOptions & options)
: Filter("PassThroughFilterUInt16", options)
{
  // set initial parameters
  {
    int filter_min = static_cast<int>(declare_parameter("filter_limit_min", 0));
    int filter_max = static_cast<int>(declare_parameter("filter_limit_max", 127));
    impl_.setFilterLimits(filter_min, filter_max);

    impl_.setFilterFieldName(
      static_cast<std::string>(declare_parameter("filter_field_name", "ring")));
    impl_.setKeepOrganized(static_cast<bool>(declare_parameter("keep_organized", false)));
    impl_.setFilterLimitsNegative(
      static_cast<bool>(declare_parameter("filter_limit_negative", false)));
  }

  using std::placeholders::_1;
  set_param_res_ = this->add_on_set_parameters_callback(
    std::bind(&PassThroughFilterUInt16Component::paramCallback, this, _1));
}


void PassThroughFilterUInt16Component::filter(
  const PointCloud2ConstPtr & input, const IndicesPtr & indices, PointCloud2 & output)
{
  std::scoped_lock lock(mutex_);
  
  rclcpp::Time start_time = this->get_clock()->now();

  const std::vector<int> &unused_indices = *indices;

  // pcl::PCLPointCloud2::Ptr pcl_input(new pcl::PCLPointCloud2);
  // pcl_conversions::toPCL(*(input), *(pcl_input));
  // impl_.setInputCloud(pcl_input);
  // impl_.setIndices(indices);
  // pcl::PCLPointCloud2 pcl_output;
  // impl_.filter(pcl_output);
  // pcl_conversions::moveFromPCL(pcl_output, output);
  pcl::PCLPointCloud2::Ptr pcl_input(new pcl::PCLPointCloud2);
  pcl_conversions::toPCL(*input, *pcl_input);

  // 转换pcl::PCLPointCloud2到pcl::PointCloud<pcl::PointXYZI>
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::fromPCLPointCloud2(*pcl_input, cloud);

  // 创建一个索引数组，用于保存满足条件的点
  std::vector<int> point_indices;

  int total_points = 0;
  int filtered_points = 0;

  // 遍历点云，找到满足强度条件的点的索引
  for (size_t i = 0; i < cloud.size(); ++i) {
      
      total_points++;

      if (cloud.points[i].intensity > 10) { // 设置强度阈值
          point_indices.push_back(i);
          filtered_points++;
      }
  }
  
   

  // 创建一个新的点云，只包含满足条件的点
  pcl::PointCloud<pcl::PointXYZI> filtered_cloud;
  pcl::copyPointCloud(cloud, point_indices, filtered_cloud);

  // 如果需要将结果转换回sensor_msgs/PointCloud2消息，可以执行以下操作
  pcl::PCLPointCloud2 pcl_output;
  pcl::toPCLPointCloud2(filtered_cloud, pcl_output);
  // sensor_msgs::PointCloud2 filtered_cloud_msg;
  pcl_conversions::fromPCL(pcl_output, output);

  rclcpp::Time end_time = this->get_clock()->now();

  rclcpp::Duration duration = end_time - start_time;  // 计算时间差

  double elapsed_time_ms = duration.seconds() * 1000.0;  // 转换为毫秒

  

}

rcl_interfaces::msg::SetParametersResult PassThroughFilterUInt16Component::paramCallback(
  const std::vector<rclcpp::Parameter> & p)
{
  std::scoped_lock lock(mutex_);

  std::uint16_t filter_min, filter_max;
  impl_.getFilterLimits(filter_min, filter_max);

  // Check the current values for filter min-max
  if (get_param(p, "filter_limit_min", filter_min)) {
    RCLCPP_DEBUG(
      get_logger(), "Setting the minimum filtering value a point will be considered from to: %d.",
      filter_min);
    // Set the filter min-max if different
    impl_.setFilterLimits(filter_min, filter_max);
  }
  // Check the current values for filter min-max
  if (get_param(p, "filter_limit_max", filter_max)) {
    RCLCPP_DEBUG(
      get_logger(), "Setting the maximum filtering value a point will be considered from to: %d.",
      filter_max);
    // Set the filter min-max if different
    impl_.setFilterLimits(filter_min, filter_max);
  }

  // Check the current value for the filter field
  std::string filter_field_name;
  get_param(p, "filter_field_name", filter_field_name);
  if (impl_.getFilterFieldName() != filter_field_name) {
    // Set the filter field if different
    impl_.setFilterFieldName(filter_field_name);
    RCLCPP_DEBUG(get_logger(), "Setting the filter field name to: %s.", filter_field_name.c_str());
  }

  // Check the current value for keep_organized
  bool keep_organized{get_parameter("keep_organized").as_bool()};
  get_param(p, "keep_organized", keep_organized);
  if (impl_.getKeepOrganized() != keep_organized) {
    RCLCPP_DEBUG(
      get_logger(), "Setting the filter keep_organized value to: %s.",
      keep_organized ? "true" : "false");
    // Call the virtual method in the child
    impl_.setKeepOrganized(keep_organized);
  }

  // Check the current value for the negative flag
  bool filter_limit_negative{get_parameter("filter_limit_negative").as_bool()};
  get_param(p, "filter_limit_negative", filter_limit_negative);
  if (impl_.getFilterLimitsNegative() != filter_limit_negative) {
    RCLCPP_DEBUG(
      get_logger(), "Setting the filter negative flag to: %s.",
      filter_limit_negative ? "true" : "false");
    // Call the virtual method in the child
    impl_.setFilterLimitsNegative(filter_limit_negative);
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  return result;
}
}  // namespace pointcloud_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_preprocessor::PassThroughFilterUInt16Component)

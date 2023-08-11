// Copyright 2020-2022 Arm Ltd., TierIV
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

#include <common/types.hpp>
#include <lidar_apollo_segmentation_tvm/lidar_apollo_segmentation_tvm.hpp>
#include <lidar_apollo_segmentation_tvm_nodes/lidar_apollo_segmentation_tvm_node.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <memory>

using autoware::common::types::bool8_t;
using autoware::common::types::float32_t;
using autoware::common::types::float64_t;
using autoware::perception::lidar_apollo_segmentation_tvm::ApolloLidarSegmentation;

namespace autoware
{
namespace perception
{
namespace lidar_apollo_segmentation_tvm_nodes
{

ApolloLidarSegmentationNode::ApolloLidarSegmentationNode(const rclcpp::NodeOptions & options)
: Node("lidar_apollo_segmentation_tvm", options),
  m_cloud_sub_ptr{create_subscription<sensor_msgs::msg::PointCloud2>(
    "points_in", rclcpp::SensorDataQoS().keep_last(1),
    [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { pointCloudCallback(msg); })},
  m_detected_pub_ptr{create_publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
    "objects_out", rclcpp::QoS{1})},
  m_detector_ptr{std::make_shared<lidar_apollo_segmentation_tvm::ApolloLidarSegmentation>(
    declare_parameter("range", rclcpp::ParameterValue{70}).get<int32_t>(),
    declare_parameter("score_threshold", rclcpp::ParameterValue{0.8}).get<float32_t>(),
    declare_parameter("use_intensity_feature", rclcpp::ParameterValue{true}).get<bool8_t>(),
    declare_parameter("use_constant_feature", rclcpp::ParameterValue{false}).get<bool8_t>(),
    declare_parameter("z_offset", rclcpp::ParameterValue{0.0}).get<float32_t>(),
    declare_parameter("min_height", rclcpp::ParameterValue{-5.0}).get<float32_t>(),
    declare_parameter("max_height", rclcpp::ParameterValue{5.0}).get<float32_t>(),
    declare_parameter("objectness_thresh", rclcpp::ParameterValue{0.5}).get<float32_t>(),
    declare_parameter("min_pts_num", rclcpp::ParameterValue{3}).get<int32_t>(),
    declare_parameter("height_thresh", rclcpp::ParameterValue{0.5}).get<float32_t>())}
{
  // Log unexpected versions of the neural network.
  auto version_status = m_detector_ptr->version_check();
  if (version_status != tvm_utility::Version::OK) {
    auto network_name = m_detector_ptr->network_name();
    if (version_status == tvm_utility::Version::Unknown) {
      RCLCPP_INFO(
        get_logger(), "The '%s' network doesn't provide a version number.", network_name.c_str());
    } else if (version_status == tvm_utility::Version::Untested) {
      RCLCPP_WARN(
        get_logger(), "The version of the '%s' network is untested.", network_name.c_str());
    } else if (version_status == tvm_utility::Version::Unsupported) {
      RCLCPP_ERROR(
        get_logger(), "The version of the '%s' network is unsupported.", network_name.c_str());
    }
  }
}

void ApolloLidarSegmentationNode::pointCloudCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr & msg)
{
  std::shared_ptr<const tier4_perception_msgs::msg::DetectedObjectsWithFeature> output_msg;
  try {
    output_msg = m_detector_ptr->detectDynamicObjects(*msg);
  } catch (const std::exception & e) {
    RCLCPP_WARN(get_logger(), e.what());
    return;
  }
  m_detected_pub_ptr->publish(*output_msg);
}
}  // namespace lidar_apollo_segmentation_tvm_nodes
}  // namespace perception
}  // namespace autoware

RCLCPP_COMPONENTS_REGISTER_NODE(
  autoware::perception::lidar_apollo_segmentation_tvm_nodes::ApolloLidarSegmentationNode)

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

#ifndef OBJECT_ASSOCIATION_MERGER__NODE_HPP_
#define OBJECT_ASSOCIATION_MERGER__NODE_HPP_

#include "object_association_merger/data_association/data_association.hpp"

#include <rclcpp/rclcpp.hpp>

#include <autoware_auto_perception_msgs/msg/detected_objects.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/convert.h>
#include <tf2/transform_datatypes.h>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace object_association
{
class ObjectAssociationMergerNode : public rclcpp::Node
{
public:
  explicit ObjectAssociationMergerNode(const rclcpp::NodeOptions & node_options);
  enum class PriorityMode : int { Object0 = 0, Object1 = 1, Confidence = 2 };

private:
  void objectsCallback(
    const autoware_auto_perception_msgs::msg::DetectedObjects::ConstSharedPtr & input_object0_msg,
    const autoware_auto_perception_msgs::msg::DetectedObjects::ConstSharedPtr & input_object1_msg);

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<autoware_auto_perception_msgs::msg::DetectedObjects>::SharedPtr
    merged_object_pub_;
  message_filters::Subscriber<autoware_auto_perception_msgs::msg::DetectedObjects> object0_sub_;
  message_filters::Subscriber<autoware_auto_perception_msgs::msg::DetectedObjects> object1_sub_;
  typedef message_filters::sync_policies::ApproximateTime<
    autoware_auto_perception_msgs::msg::DetectedObjects,
    autoware_auto_perception_msgs::msg::DetectedObjects>
    SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;
  Sync sync_;
  std::unique_ptr<DataAssociation> data_association_;
  std::string base_link_frame_id_;  // associated with the base_link frame

  PriorityMode priority_mode_;
  bool remove_overlapped_unknown_objects_;
  struct
  {
    double precision_threshold;
    double recall_threshold;
    double generalized_iou_threshold;
    std::map<int /*class label*/, double /*distance_threshold*/> distance_threshold_map;
  } overlapped_judge_param_;
};
}  // namespace object_association

#endif  // OBJECT_ASSOCIATION_MERGER__NODE_HPP_

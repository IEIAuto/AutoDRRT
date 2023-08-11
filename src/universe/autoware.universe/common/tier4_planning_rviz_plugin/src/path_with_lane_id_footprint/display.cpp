// Copyright 2021 Tier IV, Inc. All rights reserved.
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

#include <memory>

#define EIGEN_MPL2_ONLY
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <path_with_lane_id_footprint/display.hpp>

namespace rviz_plugins
{
AutowarePathWithLaneIdFootprintDisplay::AutowarePathWithLaneIdFootprintDisplay()
{
  property_path_footprint_view_ = new rviz_common::properties::BoolProperty(
    "View PathWithLaneId Footprint", true, "", this, SLOT(updateVisualization()), this);
  property_path_footprint_alpha_ = new rviz_common::properties::FloatProperty(
    "Alpha", 1.0, "", property_path_footprint_view_, SLOT(updateVisualization()), this);
  property_path_footprint_alpha_->setMin(0.0);
  property_path_footprint_alpha_->setMax(1.0);
  property_path_footprint_color_ = new rviz_common::properties::ColorProperty(
    "Color", QColor(230, 230, 50), "", property_path_footprint_view_, SLOT(updateVisualization()),
    this);
  property_vehicle_length_ = new rviz_common::properties::FloatProperty(
    "Vehicle Length", 4.77, "", property_path_footprint_view_, SLOT(updateVehicleInfo()), this);
  property_vehicle_width_ = new rviz_common::properties::FloatProperty(
    "Vehicle Width", 1.83, "", property_path_footprint_view_, SLOT(updateVehicleInfo()), this);
  property_rear_overhang_ = new rviz_common::properties::FloatProperty(
    "Rear Overhang", 1.03, "", property_path_footprint_view_, SLOT(updateVehicleInfo()), this);
  property_vehicle_length_->setMin(0.0);
  property_vehicle_width_->setMin(0.0);
  property_rear_overhang_->setMin(0.0);

  property_lane_id_view_ = new rviz_common::properties::BoolProperty(
    "View LaneId", true, "", this, SLOT(updateVisualization()), this);
  property_lane_id_scale_ = new rviz_common::properties::FloatProperty(
    "Scale", 0.1, "", property_lane_id_view_, SLOT(updateVisualization()), this);

  updateVehicleInfo();
}

AutowarePathWithLaneIdFootprintDisplay::~AutowarePathWithLaneIdFootprintDisplay()
{
  if (initialized()) {
    scene_manager_->destroyManualObject(path_footprint_manual_object_);
  }
}

void AutowarePathWithLaneIdFootprintDisplay::onInitialize()
{
  MFDClass::onInitialize();

  path_footprint_manual_object_ = scene_manager_->createManualObject();
  path_footprint_manual_object_->setDynamic(true);
  scene_node_->attachObject(path_footprint_manual_object_);
}

void AutowarePathWithLaneIdFootprintDisplay::reset()
{
  MFDClass::reset();
  path_footprint_manual_object_->clear();

  for (const auto & e : lane_id_obj_ptrs_) {
    scene_node_->removeChild(e.first.get());
  }
  lane_id_obj_ptrs_.clear();
  lane_id_obj_ptrs_.shrink_to_fit();
}

bool AutowarePathWithLaneIdFootprintDisplay::validateFloats(
  const autoware_auto_planning_msgs::msg::PathWithLaneId::ConstSharedPtr & msg_ptr)
{
  for (auto && path_point : msg_ptr->points) {
    if (!rviz_common::validateFloats(path_point.point.pose)) {
      return false;
    }
  }
  return true;
}

void AutowarePathWithLaneIdFootprintDisplay::allocateLaneIdObjects(const std::size_t size)
{
  if (size > lane_id_obj_ptrs_.size()) {
    for (std::size_t i = lane_id_obj_ptrs_.size(); i < size; i++) {
      std::unique_ptr<Ogre::SceneNode> node_ptr;
      node_ptr.reset(scene_node_->createChildSceneNode());
      auto text_ptr =
        std::make_unique<rviz_rendering::MovableText>("not initialized", "Liberation Sans", 0.1);
      text_ptr->setVisible(false);
      text_ptr->setTextAlignment(
        rviz_rendering::MovableText::H_CENTER, rviz_rendering::MovableText::V_ABOVE);
      node_ptr->attachObject(text_ptr.get());
      lane_id_obj_ptrs_.push_back(std::make_pair(std::move(node_ptr), std::move(text_ptr)));
    }
  } else {
    for (std::size_t i = lane_id_obj_ptrs_.size() - 1; i >= size; i--) {
      scene_node_->removeChild(lane_id_obj_ptrs_.at(i).first.get());
    }
    lane_id_obj_ptrs_.resize(size);
  }
}

void AutowarePathWithLaneIdFootprintDisplay::processMessage(
  const autoware_auto_planning_msgs::msg::PathWithLaneId::ConstSharedPtr msg_ptr)
{
  if (!validateFloats(msg_ptr)) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "Topic",
      "Message contained invalid floating point values (nans or infs)");
    return;
  }

  // This doesn't work in the constructor.
  if (!vehicle_info_) {
    try {
      vehicle_info_ = std::make_shared<VehicleInfo>(
        VehicleInfoUtil(*rviz_ros_node_.lock()->get_raw_node()).getVehicleInfo());
      updateVehicleInfo();
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(
        rviz_ros_node_.lock()->get_raw_node()->get_logger(),
        *rviz_ros_node_.lock()->get_raw_node()->get_clock(), 5000, "Failed to get vehicle_info: %s",
        e.what());
    }
  }

  Ogre::Vector3 position;
  Ogre::Quaternion orientation;
  if (!context_->getFrameManager()->getTransform(msg_ptr->header, position, orientation)) {
    RCLCPP_DEBUG(
      rviz_ros_node_.lock()->get_raw_node()->get_logger(),
      "Error transforming from frame '%s' to frame '%s'", msg_ptr->header.frame_id.c_str(),
      qPrintable(fixed_frame_));
  }

  scene_node_->setPosition(position);
  scene_node_->setOrientation(orientation);

  path_footprint_manual_object_->clear();

  Ogre::MaterialPtr material = Ogre::MaterialManager::getSingleton().getByName(
    "BaseWhiteNoLighting", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
  material->setSceneBlending(Ogre::SBT_TRANSPARENT_ALPHA);
  material->setDepthWriteEnabled(false);

  if (!msg_ptr->points.empty()) {
    path_footprint_manual_object_->estimateVertexCount(msg_ptr->points.size() * 4 * 2);
    path_footprint_manual_object_->begin(
      "BaseWhiteNoLighting", Ogre::RenderOperation::OT_LINE_LIST);

    allocateLaneIdObjects(msg_ptr->points.size());

    for (size_t point_idx = 0; point_idx < msg_ptr->points.size(); point_idx++) {
      const auto & path_point = msg_ptr->points.at(point_idx);
      /*
       * Footprint
       */
      if (property_path_footprint_view_->getBool()) {
        Ogre::ColourValue color;
        color = rviz_common::properties::qtToOgre(property_path_footprint_color_->getColor());
        color.a = property_path_footprint_alpha_->getFloat();

        const auto info = vehicle_footprint_info_;
        const float top = info->length - info->rear_overhang;
        const float bottom = -info->rear_overhang;
        const float left = -info->width / 2.0;
        const float right = info->width / 2.0;

        const std::array<float, 4> lon_offset_vec{top, top, bottom, bottom};
        const std::array<float, 4> lat_offset_vec{left, right, right, left};

        for (int f_idx = 0; f_idx < 4; ++f_idx) {
          const Eigen::Quaternionf quat(
            path_point.point.pose.orientation.w, path_point.point.pose.orientation.x,
            path_point.point.pose.orientation.y, path_point.point.pose.orientation.z);

          {
            const Eigen::Vector3f offset_vec{
              lon_offset_vec.at(f_idx), lat_offset_vec.at(f_idx), 0.0};
            const auto offset_to_edge = quat * offset_vec;
            path_footprint_manual_object_->position(
              path_point.point.pose.position.x + offset_to_edge.x(),
              path_point.point.pose.position.y + offset_to_edge.y(),
              path_point.point.pose.position.z);
            path_footprint_manual_object_->colour(color);
          }
          {
            const Eigen::Vector3f offset_vec{
              lon_offset_vec.at((f_idx + 1) % 4), lat_offset_vec.at((f_idx + 1) % 4), 0.0};
            const auto offset_to_edge = quat * offset_vec;
            path_footprint_manual_object_->position(
              path_point.point.pose.position.x + offset_to_edge.x(),
              path_point.point.pose.position.y + offset_to_edge.y(),
              path_point.point.pose.position.z);
            path_footprint_manual_object_->colour(color);
          }
        }
      }

      // LaneId
      if (property_lane_id_view_->getBool()) {
        Ogre::Vector3 position;
        position.x = path_point.point.pose.position.x;
        position.y = path_point.point.pose.position.y;
        position.z = path_point.point.pose.position.z;
        auto & node_ptr = lane_id_obj_ptrs_.at(point_idx).first;
        node_ptr->setPosition(position);

        const auto & text_ptr = lane_id_obj_ptrs_.at(point_idx).second;
        std::string lane_ids_str = "";
        for (const auto & e : path_point.lane_ids) {
          lane_ids_str += std::to_string(e) + ", ";
        }
        text_ptr->setCaption(lane_ids_str);
        text_ptr->setCharacterHeight(property_lane_id_scale_->getFloat());
        text_ptr->setVisible(true);
      } else {
        const auto & text_ptr = lane_id_obj_ptrs_.at(point_idx).second;
        text_ptr->setVisible(false);
      }
    }

    path_footprint_manual_object_->end();
  }
  last_msg_ptr_ = msg_ptr;
}

void AutowarePathWithLaneIdFootprintDisplay::updateVisualization()
{
  if (last_msg_ptr_ != nullptr) {
    processMessage(last_msg_ptr_);
  }
}

void AutowarePathWithLaneIdFootprintDisplay::updateVehicleInfo()
{
  if (vehicle_info_) {
    vehicle_footprint_info_ = std::make_shared<VehicleFootprintInfo>(
      vehicle_info_->vehicle_length_m, vehicle_info_->vehicle_width_m,
      vehicle_info_->rear_overhang_m);
  } else {
    const float length{property_vehicle_length_->getFloat()};
    const float width{property_vehicle_width_->getFloat()};
    const float rear_overhang{property_rear_overhang_->getFloat()};

    vehicle_footprint_info_ = std::make_shared<VehicleFootprintInfo>(length, width, rear_overhang);
  }
}

}  // namespace rviz_plugins

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rviz_plugins::AutowarePathWithLaneIdFootprintDisplay, rviz_common::Display)

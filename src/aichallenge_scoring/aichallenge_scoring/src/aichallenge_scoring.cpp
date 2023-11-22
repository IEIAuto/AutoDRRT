// Copyright 2023 TIER IV, Inc.
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

#include "aichallenge_scoring/aichallenge_scoring.hpp"

#include <lanelet2_extension/utility/utilities.hpp>
#include <lanelet2_core/geometry/Lanelet.h>
#include <lanelet2_core/geometry/LineString.h>

#include <rclcpp/rclcpp.hpp>

namespace aichallenge_scoring {
  AIChallengeScoringNode::AIChallengeScoringNode(const rclcpp::NodeOptions & node_options) 
  : Node("aichallenge_scoring", node_options),
    vehicle_info_(vehicle_info_util::VehicleInfoUtil(*this).getVehicleInfo())
  {
    using std::placeholders::_1;

    task3_start_distance_ = declare_parameter<double>("task3_start_distance");
    task3_end_distance_ = declare_parameter<double>("task3_end_distance");
    task1_start_distance_ = declare_parameter<double>("task1_start_distance");
    task1_end_distance_ = declare_parameter<double>("task1_end_distance");

    // Subscriberstic
    sub_odom_ = create_subscription<Odometry>("/localization/kinematic_state", rclcpp::QoS(1), std::bind(&AIChallengeScoringNode::onOdom, this, _1));
    sub_map_ = create_subscription<HADMapBin>("/map/vector_map", rclcpp::QoS{1}.transient_local(), std::bind(&AIChallengeScoringNode::onMap, this, _1));

    // Publishers
    pub_footprint_marker_ = create_publisher<visualization_msgs::msg::Marker>("output/footprint_marker", rclcpp::QoS(1));
    pub_score_ = create_publisher<aichallenge_scoring_msgs::msg::Score>("output/score", rclcpp::QoS{1}.transient_local());

    // Timer
    using std::chrono_literals::operator""ms;
    timer_ = rclcpp::create_timer(this, get_clock(), 200ms, std::bind(&AIChallengeScoringNode::onTimer, this));

    task3_duration_ = 0.0;
    has_finished_task1_ = false;
    has_finished_task2_ = false;
    has_finished_task3_ = false;
    is_doing_task3_ = false;
    has_started_driving_ = false;
    has_exceeded_speed_limit_ = false;
    odometry_ = nullptr;
    stop_watch_ptr_.tic("task3_duration");
  }

  void AIChallengeScoringNode::onOdom(const Odometry::SharedPtr msg) {
    odometry_ = msg;
  }

  void AIChallengeScoringNode::onMap(const HADMapBin::ConstSharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "[AIChallengeScoringNode]: Start loading lanelet");
    lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
    lanelet::utils::conversion::fromBinMsg(*msg, lanelet_map_ptr_);
    RCLCPP_INFO(this->get_logger(), "[AIChallengeScoringNode]: Map is loaded");
  }

  bool AIChallengeScoringNode::isDataReady() {
    if (lanelet_map_ptr_ == nullptr) {
      RCLCPP_WARN(get_logger(), "Map not received yet");
      return false;
    }

    if (!odometry_) {
      RCLCPP_WARN(get_logger(), "Vehicle odom not received yet");
      return false;
    }

    return true;
  }

  void AIChallengeScoringNode::onTimer() {
    static const std::string total_duration_timer_name = "total_duration";

    if (!isDataReady()) {
      return;
    }

    // Get closest lanelet to current vehicle pose
    if (!lanelet::utils::query::getClosestLanelet(
        lanelet::utils::query::laneletLayer(lanelet_map_ptr_), 
        odometry_->pose.pose, &closest_lanelet_)) {
      RCLCPP_INFO(this->get_logger(), "No closest lanelet");
      return;
    }

    // Calculate the distance along the center line linestring to the vehicle position point
    lanelet::BasicPoint2d vehicle_pos(odometry_->pose.pose.position.x, odometry_->pose.pose.position.y);
    const auto arc_coordinates = lanelet::geometry::toArcCoordinates(lanelet::utils::to2D(closest_lanelet_.centerline().basicLineString()), vehicle_pos);

    auto isStopped = [](const double odom, const double stop_speed) -> bool { 
      if (std::abs(odom) < stop_speed)
        return true;
      else
        return false;
    };

    // Retrieve distance_score
    const auto distance_score = arc_coordinates.length;
    const auto linear_twist = odometry_->twist.twist.linear.x;
    const double stop_speed = 0.01;
    const bool is_stopped = isStopped(linear_twist, stop_speed);

    // Check start of self-driving
    if (!is_stopped && !has_started_driving_) {
      RCLCPP_INFO(this->get_logger(), "Self-driving started");

      has_started_driving_ = true;
      stop_watch_ptr_.tic(total_duration_timer_name);
    }

    // Update total duration
    auto total_duration = 0.0;
    if (has_started_driving_) {
      total_duration = stop_watch_ptr_.toc(total_duration_timer_name, false);
    }

    // Ignore Timeout
    const auto timeout_time = 50000000.0 * 60.0;
    auto is_timeout = total_duration > timeout_time;

    // Check speed limit
    const auto speed_limit = 7.0 * 1000.0 / 60.0 / 60.0;
    if (linear_twist > speed_limit && !has_exceeded_speed_limit_) {
      RCLCPP_INFO(this->get_logger(), "Vehicle speed exceeds the speed limit");

      has_exceeded_speed_limit_ = true;
    }

    // task1 completed or not 
     if (distance_score >= task1_start_distance_ && distance_score <= task1_end_distance_) {
        if (is_stopped) {
          has_finished_task1_ = true;
        }
    }
    
    // Skip task2

    // Start of Task 3: Start timer
    if (distance_score >= task3_start_distance_ && distance_score <= task3_end_distance_ && !is_doing_task3_) {
      RCLCPP_INFO(this->get_logger(), "Task3 started");

      task3_duration_ = stop_watch_ptr_.toc("task3_duration", true);
      is_doing_task3_ = true;
    }

    // End of Task 3: Stop timer
    if (distance_score > task3_end_distance_ && is_stopped && is_doing_task3_) {
      task3_duration_ = stop_watch_ptr_.toc("task3_duration", false);
      is_doing_task3_ = false;
      has_finished_task3_ = true;
    }

    // Prepare message to publish
    aichallenge_scoring_msgs::msg::Score score_msg;
    score_msg.distance_score = std::min(distance_score, task3_end_distance_);
    if (is_doing_task3_) {
      score_msg.task3_duration = stop_watch_ptr_.toc("task3_duration", false);
    } else if (has_finished_task3_) {
      score_msg.task3_duration = task3_duration_;
    } else {
      score_msg.task3_duration = 0.0;
    }
    score_msg.is_distance_score_maxed_out = distance_score >= task3_end_distance_;
    score_msg.is_stopped = is_stopped;
    score_msg.is_doing_task3 = is_doing_task3_;
    score_msg.has_finished_task1 = has_finished_task1_;
    score_msg.has_finished_task2 = true;
    score_msg.has_finished_task3 = has_finished_task3_;
    score_msg.total_duration = total_duration;
    score_msg.is_timeout = is_timeout;
    score_msg.has_exceeded_speed_limit = has_exceeded_speed_limit_;

    // Check if vehicle is inside lane
    createVehicleFootprint(vehicle_info_);
    const auto is_outside_lane = isOutsideLaneFromVehicleFootprint(closest_lanelet_);
    visualizeVehicleFootprint(is_outside_lane);
    score_msg.is_outside_lane = is_outside_lane;

    pub_score_->publish(score_msg);
  }

  void AIChallengeScoringNode::createVehicleFootprint(const vehicle_info_util::VehicleInfo & vehicle_info) {
    // Create vehicle footprint
    const double x_front = vehicle_info.front_overhang_m + vehicle_info.wheel_base_m;
    const double x_rear = -(vehicle_info.rear_overhang_m);
    const double y_left = vehicle_info.wheel_tread_m / 2.0 + vehicle_info.left_overhang_m;
    const double y_right = -(vehicle_info.wheel_tread_m / 2.0 + vehicle_info.right_overhang_m);

    tier4_autoware_utils::LinearRing2d local_vehicle_footprint;
    local_vehicle_footprint.push_back(tier4_autoware_utils::Point2d{x_front, y_left});
    local_vehicle_footprint.push_back(tier4_autoware_utils::Point2d{x_front, y_right});
    local_vehicle_footprint.push_back(tier4_autoware_utils::Point2d{x_rear, y_right});
    local_vehicle_footprint.push_back(tier4_autoware_utils::Point2d{x_rear, y_left});
    local_vehicle_footprint.push_back(tier4_autoware_utils::Point2d{x_front, y_left});

    // Transform the local_vehicle_footprint to map-frame
    vehicle_footprint_ = tier4_autoware_utils::transformVector(local_vehicle_footprint, tier4_autoware_utils::pose2transform(odometry_->pose.pose));
  }

  bool AIChallengeScoringNode::isOutsideLaneFromVehicleFootprint(const lanelet::ConstLanelet & lanelet) {
    // Check if footprint is completely within the lane
    if (!boost::geometry::within(vehicle_footprint_, lanelet.polygon2d().basicPolygon())) {
      // Vehicle is not completely within the lane
      return true;
    }

    // Vehicle is completely within the lane
    return false;
  }

  void AIChallengeScoringNode::visualizeVehicleFootprint(bool is_outside_lane) {

    // Create marker for visualizing vehicle footprint
    visualization_msgs::msg::Marker marker;
    std_msgs::msg::ColorRGBA cl;
    
    if (is_outside_lane) {
      cl.r = 1.0;
      cl.g = 0.0;
      cl.b = 0.0;
      cl.a = 0.99;
    } else {
      cl.r = 0.0;
      cl.g = 1.0;
      cl.b = 0.0;
      cl.a = 0.99;
    }
    
    marker.header.frame_id = "map";
    marker.header.stamp = rclcpp::Time();
    marker.frame_locked = false;
    marker.id = 0;
    marker.ns = "aichallenge_scoring";
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.lifetime = rclcpp::Duration(0, 0);
    marker.pose.position.x = 0.0;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = 0.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.1;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;
    marker.color = cl;

    for (const auto & point : vehicle_footprint_) {
      marker.points.push_back(tier4_autoware_utils::toMsg(point.to_3d()));
    }
    
    pub_footprint_marker_->publish(marker);
  }

} // namespace aichallenge_scoring

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(aichallenge_scoring::AIChallengeScoringNode)

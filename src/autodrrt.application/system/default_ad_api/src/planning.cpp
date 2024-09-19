// Copyright 2022 TIER IV, Inc.
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

#include "planning.hpp"

#include <motion_utils/trajectory/trajectory.hpp>

#include <autoware_adapi_v1_msgs/msg/planning_behavior.hpp>

#include <memory>
#include <string>
#include <vector>

namespace default_ad_api
{

template <class T>
void concat(std::vector<T> & v1, const std::vector<T> & v2)
{
  v1.insert(v1.end(), v2.begin(), v2.end());
}

template <class T>
std::vector<typename rclcpp::Subscription<T>::SharedPtr> init_factors(
  rclcpp::Node * node, std::vector<typename T::ConstSharedPtr> & factors,
  const std::vector<std::string> & topics)
{
  const auto callback = [&factors](const int index) {
    return [&factors, index](const typename T::ConstSharedPtr msg) { factors[index] = msg; };
  };

  std::vector<typename rclcpp::Subscription<T>::SharedPtr> subs;
  for (size_t index = 0; index < topics.size(); ++index) {
    subs.push_back(node->create_subscription<T>(topics[index], rclcpp::QoS(1), callback(index)));
  }
  factors.resize(topics.size());
  return subs;
}

template <class T>
T merge_factors(const rclcpp::Time stamp, const std::vector<typename T::ConstSharedPtr> & factors)
{
  T message;
  message.header.stamp = stamp;
  message.header.frame_id = "map";

  for (const auto & factor : factors) {
    if (factor) {
      concat(message.factors, factor->factors);
    }
  }
  return message;
}

uint16_t convert_velocity_behavior(const std::string & behavior)
{
  using autoware_adapi_v1_msgs::msg::PlanningBehavior;
  using autoware_adapi_v1_msgs::msg::VelocityFactor;

  if (behavior == PlanningBehavior::AVOIDANCE) {
    return VelocityFactor::AVOIDANCE;
  }
  if (behavior == PlanningBehavior::CROSSWALK) {
    return VelocityFactor::CROSSWALK;
  }
  if (behavior == PlanningBehavior::INTERSECTION) {
    return VelocityFactor::INTERSECTION;
  }
  if (behavior == PlanningBehavior::LANE_CHANGE) {
    return VelocityFactor::LANE_CHANGE;
  }
  if (behavior == PlanningBehavior::MERGE) {
    return VelocityFactor::MERGE;
  }
  if (behavior == PlanningBehavior::NO_DRIVABLE_LANE) {
    return VelocityFactor::NO_DRIVABLE_LANE;
  }
  if (behavior == PlanningBehavior::NO_STOPPING_AREA) {
    return VelocityFactor::NO_STOPPING_AREA;
  }
  if (behavior == PlanningBehavior::REAR_CHECK) {
    return VelocityFactor::REAR_CHECK;
  }
  if (behavior == PlanningBehavior::ROUTE_OBSTACLE) {
    return VelocityFactor::ROUTE_OBSTACLE;
  }
  if (behavior == PlanningBehavior::SIDEWALK) {
    return VelocityFactor::SIDEWALK;
  }
  if (behavior == PlanningBehavior::STOP_SIGN) {
    return VelocityFactor::STOP_SIGN;
  }
  if (behavior == PlanningBehavior::SURROUNDING_OBSTACLE) {
    return VelocityFactor::SURROUNDING_OBSTACLE;
  }
  if (behavior == PlanningBehavior::TRAFFIC_SIGNAL) {
    return VelocityFactor::TRAFFIC_SIGNAL;
  }
  if (behavior == PlanningBehavior::USER_DEFINED_DETECTION_AREA) {
    return VelocityFactor::USER_DEFINED_DETECTION_AREA;
  }
  if (behavior == PlanningBehavior::VIRTUAL_TRAFFIC_LIGHT) {
    return VelocityFactor::V2I_GATE_CONTROL_ENTER;
  }
  return VelocityFactor::UNKNOWN;
}

uint16_t convert_steering_behavior(const std::string & behavior)
{
  using autoware_adapi_v1_msgs::msg::PlanningBehavior;
  using autoware_adapi_v1_msgs::msg::SteeringFactor;

  if (behavior == PlanningBehavior::AVOIDANCE) {
    return SteeringFactor::AVOIDANCE_PATH_CHANGE;
  }
  if (behavior == PlanningBehavior::GOAL_PLANNER) {
    return SteeringFactor::GOAL_PLANNER;
  }
  if (behavior == PlanningBehavior::INTERSECTION) {
    return SteeringFactor::INTERSECTION;
  }
  if (behavior == PlanningBehavior::LANE_CHANGE) {
    return SteeringFactor::LANE_CHANGE;
  }
  if (behavior == PlanningBehavior::START_PLANNER) {
    return SteeringFactor::START_PLANNER;
  }
  return SteeringFactor::UNKNOWN;
}

PlanningNode::PlanningNode(const rclcpp::NodeOptions & options) : Node("planning", options)
{
  // TODO(Takagi, Isamu): remove default value
  stop_distance_ = declare_parameter<double>("stop_distance", 1.0);
  stop_duration_ = declare_parameter<double>("stop_duration", 1.0);
  stop_checker_ = std::make_unique<VehicleStopChecker>(this, stop_duration_ + 1.0);

  std::vector<std::string> velocity_factor_topics = {
    "/planning/velocity_factors/blind_spot",
    "/planning/velocity_factors/crosswalk",
    "/planning/velocity_factors/detection_area",
    "/planning/velocity_factors/intersection",
    "/planning/velocity_factors/merge_from_private",
    "/planning/velocity_factors/no_stopping_area",
    "/planning/velocity_factors/obstacle_stop",
    "/planning/velocity_factors/obstacle_cruise",
    "/planning/velocity_factors/occlusion_spot",
    "/planning/velocity_factors/stop_line",
    "/planning/velocity_factors/surround_obstacle",
    "/planning/velocity_factors/traffic_light",
    "/planning/velocity_factors/virtual_traffic_light",
    "/planning/velocity_factors/walkway"};

  std::vector<std::string> steering_factor_topics = {
    "/planning/steering_factor/avoidance", "/planning/steering_factor/intersection",
    "/planning/steering_factor/lane_change", "/planning/steering_factor/start_planner",
    "/planning/steering_factor/goal_planner"};

  sub_velocity_factors_ =
    init_factors<VelocityFactorArray>(this, velocity_factors_, velocity_factor_topics);
  sub_steering_factors_ =
    init_factors<SteeringFactorArray>(this, steering_factors_, steering_factor_topics);

  const auto adaptor = component_interface_utils::NodeAdaptor(this);
  adaptor.init_pub(pub_velocity_factors_);
  adaptor.init_pub(pub_steering_factors_);
  adaptor.init_sub(sub_kinematic_state_, this, &PlanningNode::on_kinematic_state);
  adaptor.init_sub(sub_trajectory_, this, &PlanningNode::on_trajectory);

  const auto rate = rclcpp::Rate(5);
  timer_ = rclcpp::create_timer(this, get_clock(), rate.period(), [this]() { on_timer(); });
}

void PlanningNode::on_trajectory(const Trajectory::ConstSharedPtr msg)
{
  trajectory_ = msg;
}

void PlanningNode::on_kinematic_state(const KinematicState::ConstSharedPtr msg)
{
  kinematic_state_ = msg;

  geometry_msgs::msg::TwistStamped twist;
  twist.header = msg->header;
  twist.twist = msg->twist.twist;
  stop_checker_->addTwist(twist);
}

void PlanningNode::on_timer()
{
  using autoware_adapi_v1_msgs::msg::VelocityFactor;
  auto velocity = merge_factors<VelocityFactorArray>(now(), velocity_factors_);
  auto steering = merge_factors<SteeringFactorArray>(now(), steering_factors_);

  // Set velocity factor type for compatibility.
  for (auto & factor : velocity.factors) {
    factor.type = convert_velocity_behavior(factor.behavior);
  }

  // Set steering factor type for compatibility.
  for (auto & factor : steering.factors) {
    factor.type = convert_steering_behavior(factor.behavior);
  }

  // Set the distance if it is nan.
  if (trajectory_ && kinematic_state_) {
    for (auto & factor : velocity.factors) {
      if (std::isnan(factor.distance)) {
        const auto & curr_point = kinematic_state_->pose.pose.position;
        const auto & stop_point = factor.pose.position;
        const auto & points = trajectory_->points;
        factor.distance = motion_utils::calcSignedArcLength(points, curr_point, stop_point);
      }
    }
  }

  // Set the status if it is unknown.
  const auto is_vehicle_stopped = stop_checker_->isVehicleStopped(stop_duration_);
  for (auto & factor : velocity.factors) {
    if ((factor.status == VelocityFactor::UNKNOWN) && (!std::isnan(factor.distance))) {
      const auto is_stopped = is_vehicle_stopped && (factor.distance < stop_distance_);
      factor.status = is_stopped ? VelocityFactor::STOPPED : VelocityFactor::APPROACHING;
    }
  }

  pub_velocity_factors_->publish(velocity);
  pub_steering_factors_->publish(steering);
}

}  // namespace default_ad_api

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(default_ad_api::PlanningNode)

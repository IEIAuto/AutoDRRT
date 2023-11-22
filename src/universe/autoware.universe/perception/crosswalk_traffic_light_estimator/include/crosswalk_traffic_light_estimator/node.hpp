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

#ifndef CROSSWALK_TRAFFIC_LIGHT_ESTIMATOR__NODE_HPP_
#define CROSSWALK_TRAFFIC_LIGHT_ESTIMATOR__NODE_HPP_

#include <lanelet2_extension/regulatory_elements/autoware_traffic_light.hpp>
#include <lanelet2_extension/utility/query.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <autoware_auto_mapping_msgs/msg/had_map_bin.hpp>
#include <autoware_auto_perception_msgs/msg/traffic_light.hpp>
#include <autoware_auto_perception_msgs/msg/traffic_signal.hpp>
#include <autoware_auto_perception_msgs/msg/traffic_signal_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <tier4_debug_msgs/msg/float64_stamped.hpp>

#include <lanelet2_core/Attribute.h>
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_routing/RoutingGraphContainer.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

#include <memory>
#include <unordered_map>
#include <vector>

namespace traffic_light
{

using autoware_auto_mapping_msgs::msg::HADMapBin;
using autoware_auto_perception_msgs::msg::TrafficLight;
using autoware_auto_perception_msgs::msg::TrafficSignal;
using autoware_auto_perception_msgs::msg::TrafficSignalArray;
using autoware_planning_msgs::msg::LaneletRoute;
using tier4_autoware_utils::DebugPublisher;
using tier4_autoware_utils::StopWatch;
using tier4_debug_msgs::msg::Float64Stamped;
using TrafficLightIdMap = std::unordered_map<lanelet::Id, TrafficSignal>;

class CrosswalkTrafficLightEstimatorNode : public rclcpp::Node
{
public:
  explicit CrosswalkTrafficLightEstimatorNode(const rclcpp::NodeOptions & options);

private:
  rclcpp::Subscription<HADMapBin>::SharedPtr sub_map_;
  rclcpp::Subscription<LaneletRoute>::SharedPtr sub_route_;
  rclcpp::Subscription<TrafficSignalArray>::SharedPtr sub_traffic_light_array_;
  rclcpp::Publisher<TrafficSignalArray>::SharedPtr pub_traffic_light_array_;

  lanelet::LaneletMapPtr lanelet_map_ptr_;
  lanelet::traffic_rules::TrafficRulesPtr traffic_rules_ptr_;
  lanelet::routing::RoutingGraphPtr routing_graph_ptr_;
  std::shared_ptr<const lanelet::routing::RoutingGraphContainer> overall_graphs_ptr_;

  lanelet::ConstLanelets conflicting_crosswalks_;

  void onMap(const HADMapBin::ConstSharedPtr msg);
  void onRoute(const LaneletRoute::ConstSharedPtr msg);
  void onTrafficLightArray(const TrafficSignalArray::ConstSharedPtr msg);

  void updateLastDetectedSignal(const TrafficLightIdMap & traffic_signals);
  void setCrosswalkTrafficSignal(
    const lanelet::ConstLanelet & crosswalk, const uint8_t color, TrafficSignalArray & msg) const;

  lanelet::ConstLanelets getNonRedLanelets(
    const lanelet::ConstLanelets & lanelets, const TrafficLightIdMap & traffic_light_id_map) const;

  uint8_t estimateCrosswalkTrafficSignal(
    const lanelet::ConstLanelet & crosswalk, const lanelet::ConstLanelets & non_red_lanelets) const;

  boost::optional<uint8_t> getHighestConfidenceTrafficSignal(
    const lanelet::ConstLineStringsOrPolygons3d & traffic_lights,
    const TrafficLightIdMap & traffic_light_id_map) const;

  // Node param
  bool use_last_detect_color_;

  // Signal history
  TrafficLightIdMap last_detect_color_;

  // Stop watch
  StopWatch<std::chrono::milliseconds> stop_watch_;

  // Debug
  std::shared_ptr<DebugPublisher> pub_processing_time_;
};

}  // namespace traffic_light

#endif  // CROSSWALK_TRAFFIC_LIGHT_ESTIMATOR__NODE_HPP_

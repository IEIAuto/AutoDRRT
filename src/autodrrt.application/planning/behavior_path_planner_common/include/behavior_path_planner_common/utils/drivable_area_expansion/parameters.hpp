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

#ifndef BEHAVIOR_PATH_PLANNER_COMMON__UTILS__DRIVABLE_AREA_EXPANSION__PARAMETERS_HPP_
#define BEHAVIOR_PATH_PLANNER_COMMON__UTILS__DRIVABLE_AREA_EXPANSION__PARAMETERS_HPP_

#include <rclcpp/node.hpp>
#include <vehicle_info_util/vehicle_info_util.hpp>

#include <string>
#include <vector>

namespace drivable_area_expansion
{

struct DrivableAreaExpansionParameters
{
  static constexpr auto DRIVABLE_AREA_RIGHT_BOUND_OFFSET_PARAM = "drivable_area_right_bound_offset";
  static constexpr auto DRIVABLE_AREA_LEFT_BOUND_OFFSET_PARAM = "drivable_area_left_bound_offset";
  static constexpr auto DRIVABLE_AREA_TYPES_TO_SKIP_PARAM = "drivable_area_types_to_skip";
  static constexpr auto ENABLED_PARAM = "dynamic_expansion.enabled";
  static constexpr auto EGO_EXTRA_FRONT_OVERHANG = "dynamic_expansion.ego.extra_front_overhang";
  static constexpr auto EGO_EXTRA_WHEELBASE = "dynamic_expansion.ego.extra_wheelbase";
  static constexpr auto EGO_EXTRA_WIDTH = "dynamic_expansion.ego.extra_width";
  static constexpr auto DYN_OBJECTS_EXTRA_OFFSET_FRONT =
    "dynamic_expansion.dynamic_objects.extra_footprint_offset.front";
  static constexpr auto DYN_OBJECTS_EXTRA_OFFSET_REAR =
    "dynamic_expansion.dynamic_objects.extra_footprint_offset.rear";
  static constexpr auto DYN_OBJECTS_EXTRA_OFFSET_LEFT =
    "dynamic_expansion.dynamic_objects.extra_footprint_offset.left";
  static constexpr auto DYN_OBJECTS_EXTRA_OFFSET_RIGHT =
    "dynamic_expansion.dynamic_objects.extra_footprint_offset.right";
  static constexpr auto MAX_EXP_DIST_PARAM = "dynamic_expansion.max_expansion_distance";
  static constexpr auto RESAMPLE_INTERVAL_PARAM =
    "dynamic_expansion.path_preprocessing.resample_interval";
  static constexpr auto MAX_PATH_ARC_LENGTH_PARAM =
    "dynamic_expansion.path_preprocessing.max_arc_length";
  static constexpr auto MAX_REUSE_DEVIATION_PARAM =
    "dynamic_expansion.path_preprocessing.reuse_max_deviation";
  static constexpr auto AVOID_DYN_OBJECTS_PARAM = "dynamic_expansion.dynamic_objects.avoid";
  static constexpr auto AVOID_LINESTRING_TYPES_PARAM = "dynamic_expansion.avoid_linestring.types";
  static constexpr auto AVOID_LINESTRING_DIST_PARAM = "dynamic_expansion.avoid_linestring.distance";
  static constexpr auto SMOOTHING_CURVATURE_WINDOW_PARAM =
    "dynamic_expansion.smoothing.curvature_average_window";
  static constexpr auto SMOOTHING_MAX_BOUND_RATE_PARAM =
    "dynamic_expansion.smoothing.max_bound_rate";
  static constexpr auto SMOOTHING_EXTRA_ARC_LENGTH_PARAM =
    "dynamic_expansion.smoothing.extra_arc_length";
  static constexpr auto PRINT_RUNTIME_PARAM = "dynamic_expansion.print_runtime";

  // static expansion
  double drivable_area_right_bound_offset{};
  double drivable_area_left_bound_offset{};
  std::vector<std::string> drivable_area_types_to_skip{};
  // dynamic expansion
  bool enabled = false;
  double avoid_linestring_dist{};
  double extra_front_overhang{};
  double extra_wheelbase{};
  double extra_width{};
  int curvature_average_window{};
  double max_bound_rate{};
  double dynamic_objects_extra_left_offset{};
  double dynamic_objects_extra_right_offset{};
  double dynamic_objects_extra_rear_offset{};
  double dynamic_objects_extra_front_offset{};
  double max_expansion_distance{};
  double max_path_arc_length{};
  double resample_interval{};
  double extra_arc_length{};
  double max_reuse_deviation{};
  bool avoid_dynamic_objects{};
  bool print_runtime{};
  std::vector<std::string> avoid_linestring_types{};
  vehicle_info_util::VehicleInfo vehicle_info;

  DrivableAreaExpansionParameters() = default;
  explicit DrivableAreaExpansionParameters(rclcpp::Node & node) { init(node); }

  void init(rclcpp::Node & node)
  {
    drivable_area_right_bound_offset =
      node.declare_parameter<double>(DRIVABLE_AREA_RIGHT_BOUND_OFFSET_PARAM);
    drivable_area_left_bound_offset =
      node.declare_parameter<double>(DRIVABLE_AREA_LEFT_BOUND_OFFSET_PARAM);
    drivable_area_types_to_skip =
      node.declare_parameter<std::vector<std::string>>(DRIVABLE_AREA_TYPES_TO_SKIP_PARAM);
    enabled = node.declare_parameter<bool>(ENABLED_PARAM);
    max_expansion_distance = node.declare_parameter<double>(MAX_EXP_DIST_PARAM);
    extra_front_overhang = node.declare_parameter<double>(EGO_EXTRA_FRONT_OVERHANG);
    extra_wheelbase = node.declare_parameter<double>(EGO_EXTRA_WHEELBASE);
    extra_width = node.declare_parameter<double>(EGO_EXTRA_WIDTH);
    curvature_average_window = node.declare_parameter<int>(SMOOTHING_CURVATURE_WINDOW_PARAM);
    max_bound_rate = node.declare_parameter<double>(SMOOTHING_MAX_BOUND_RATE_PARAM);
    extra_arc_length = node.declare_parameter<double>(SMOOTHING_EXTRA_ARC_LENGTH_PARAM);
    max_path_arc_length = node.declare_parameter<double>(MAX_PATH_ARC_LENGTH_PARAM);
    resample_interval = node.declare_parameter<double>(RESAMPLE_INTERVAL_PARAM);
    max_reuse_deviation = node.declare_parameter<double>(MAX_REUSE_DEVIATION_PARAM);
    dynamic_objects_extra_front_offset =
      node.declare_parameter<double>(DYN_OBJECTS_EXTRA_OFFSET_FRONT);
    dynamic_objects_extra_rear_offset =
      node.declare_parameter<double>(DYN_OBJECTS_EXTRA_OFFSET_REAR);
    dynamic_objects_extra_left_offset =
      node.declare_parameter<double>(DYN_OBJECTS_EXTRA_OFFSET_LEFT);
    dynamic_objects_extra_right_offset =
      node.declare_parameter<double>(DYN_OBJECTS_EXTRA_OFFSET_RIGHT);
    avoid_linestring_types =
      node.declare_parameter<std::vector<std::string>>(AVOID_LINESTRING_TYPES_PARAM);
    avoid_dynamic_objects = node.declare_parameter<bool>(AVOID_DYN_OBJECTS_PARAM);
    avoid_linestring_dist = node.declare_parameter<double>(AVOID_LINESTRING_DIST_PARAM);
    print_runtime = node.declare_parameter<bool>(PRINT_RUNTIME_PARAM);

    vehicle_info = vehicle_info_util::VehicleInfoUtil(node).getVehicleInfo();
  }
};

}  // namespace drivable_area_expansion
#endif  // BEHAVIOR_PATH_PLANNER_COMMON__UTILS__DRIVABLE_AREA_EXPANSION__PARAMETERS_HPP_

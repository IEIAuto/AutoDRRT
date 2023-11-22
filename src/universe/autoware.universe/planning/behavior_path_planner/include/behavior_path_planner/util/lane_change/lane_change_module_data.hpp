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
#ifndef BEHAVIOR_PATH_PLANNER__UTIL__LANE_CHANGE__LANE_CHANGE_MODULE_DATA_HPP_
#define BEHAVIOR_PATH_PLANNER__UTIL__LANE_CHANGE__LANE_CHANGE_MODULE_DATA_HPP_

#include "lanelet2_core/geometry/Lanelet.h"

#include "autoware_auto_planning_msgs/msg/path_point_with_lane_id.hpp"

#include <string>
#include <vector>

namespace behavior_path_planner
{
struct LaneChangeParameters
{
  // trajectory generation
  double lane_change_prepare_duration{2.0};
  double lane_changing_safety_check_duration{4.0};
  double lane_changing_lateral_jerk{0.5};
  double lane_changing_lateral_acc{0.5};
  double lane_change_finish_judge_buffer{3.0};
  double minimum_lane_change_velocity{5.6};
  double prediction_time_resolution{0.5};
  double maximum_deceleration{1.0};
  int lane_change_sampling_num{10};

  // collision check
  bool enable_collision_check_at_prepare_phase{true};
  double prepare_phase_ignore_target_speed_thresh{0.1};
  bool use_predicted_path_outside_lanelet{false};
  bool use_all_predicted_path{false};

  // abort
  bool enable_cancel_lane_change{true};
  bool enable_abort_lane_change{false};

  double abort_delta_time{3.0};
  double abort_max_lateral_jerk{10.0};

  // drivable area expansion
  double drivable_area_right_bound_offset{0.0};
  double drivable_area_left_bound_offset{0.0};
  std::vector<std::string> drivable_area_types_to_skip{};

  // debug marker
  bool publish_debug_marker{false};
};

enum class LaneChangeStates {
  Normal = 0,
  Cancel,
  Abort,
  Stop,
};

struct LaneChangePhaseInfo
{
  double prepare{0.0};
  double lane_changing{0.0};

  [[nodiscard]] double sum() const { return prepare + lane_changing; }
};

struct LaneChangeTargetObjectIndices
{
  std::vector<size_t> current_lane{};
  std::vector<size_t> target_lane{};
  std::vector<size_t> other_lane{};
};
}  // namespace behavior_path_planner

#endif  // BEHAVIOR_PATH_PLANNER__UTIL__LANE_CHANGE__LANE_CHANGE_MODULE_DATA_HPP_

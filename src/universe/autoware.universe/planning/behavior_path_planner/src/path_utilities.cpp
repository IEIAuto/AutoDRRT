// Copyright 2021 Tier IV, Inc.
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

#include "behavior_path_planner/path_utilities.hpp"

#include "behavior_path_planner/utilities.hpp"

#include <interpolation/spline_interpolation.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <motion_utils/resample/resample.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <tf2/utils.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace behavior_path_planner::util
{
/**
 * @brief calc path arclength on each points from start point to end point.
 */
std::vector<double> calcPathArcLengthArray(
  const PathWithLaneId & path, const size_t start, const size_t end, const double offset)
{
  const auto bounded_start = std::max(start, size_t{0});
  const auto bounded_end = std::min(end, path.points.size());
  std::vector<double> out;
  out.reserve(bounded_end - bounded_start);

  double sum = offset;
  out.push_back(sum);

  for (size_t i = bounded_start + 1; i < bounded_end; ++i) {
    sum +=
      tier4_autoware_utils::calcDistance2d(path.points.at(i).point, path.points.at(i - 1).point);
    out.push_back(sum);
  }
  return out;
}

/**
 * @brief resamplePathWithSpline
 */
PathWithLaneId resamplePathWithSpline(
  const PathWithLaneId & path, const double interval, const bool keep_input_points,
  const std::pair<double, double> target_section)
{
  if (path.points.size() < 2) {
    return path;
  }

  std::vector<autoware_auto_planning_msgs::msg::PathPoint> transformed_path(path.points.size());
  for (size_t i = 0; i < path.points.size(); ++i) {
    transformed_path.at(i) = path.points.at(i).point;
  }

  constexpr double epsilon = 0.2;
  const auto has_almost_same_value = [&](const auto & vec, const auto x) {
    if (vec.empty()) return false;
    const auto has_close = [&](const auto v) { return std::abs(v - x) < epsilon; };
    return std::find_if(vec.begin(), vec.end(), has_close) != vec.end();
  };

  // Get lane ids that are not duplicated
  std::vector<double> s_in;
  std::vector<int64_t> unique_lane_ids;
  const auto s_vec =
    motion_utils::calcSignedArcLengthPartialSum(transformed_path, 0, transformed_path.size());
  for (size_t i = 0; i < path.points.size(); ++i) {
    const double s = s_vec.at(i);
    for (const auto & lane_id : path.points.at(i).lane_ids) {
      if (keep_input_points && !has_almost_same_value(s_in, s)) {
        s_in.push_back(s);
      }
      if (
        std::find(unique_lane_ids.begin(), unique_lane_ids.end(), lane_id) !=
        unique_lane_ids.end()) {
        unique_lane_ids.push_back(lane_id);
        if (!has_almost_same_value(s_in, s)) {
          s_in.push_back(s);
        }
      }
    }
  }

  std::vector<double> s_out = s_in;

  const auto start_s = std::max(target_section.first, 0.0);
  const auto end_s = std::min(target_section.second, s_vec.back());
  for (double s = start_s; s < end_s; s += interval) {
    if (!has_almost_same_value(s_out, s)) {
      s_out.push_back(s);
    }
  }

  // Insert Terminal Point
  if (!has_almost_same_value(s_out, end_s)) {
    s_out.push_back(end_s);
  }

  // Insert Stop Point
  const auto closest_stop_dist = motion_utils::calcDistanceToForwardStopPoint(transformed_path);
  if (closest_stop_dist && !has_almost_same_value(s_out, *closest_stop_dist)) {
    s_out.push_back(*closest_stop_dist);
  }

  if (s_out.empty()) {
    return path;
  }

  std::sort(s_out.begin(), s_out.end());

  return motion_utils::resamplePath(path, s_out);
}

Path toPath(const PathWithLaneId & input)
{
  Path output{};
  output.header = input.header;
  output.left_bound = input.left_bound;
  output.right_bound = input.right_bound;
  output.points.resize(input.points.size());
  for (size_t i = 0; i < input.points.size(); ++i) {
    output.points.at(i) = input.points.at(i).point;
  }
  return output;
}

size_t getIdxByArclength(
  const PathWithLaneId & path, const size_t target_idx, const double signed_arc)
{
  if (path.points.empty()) {
    throw std::runtime_error("[getIdxByArclength] path points must be > 0");
  }

  using tier4_autoware_utils::calcDistance2d;
  double sum_length = 0.0;
  if (signed_arc >= 0.0) {
    for (size_t i = target_idx; i < path.points.size() - 1; ++i) {
      const auto next_i = i + 1;
      sum_length += calcDistance2d(path.points.at(i), path.points.at(next_i));
      if (sum_length > signed_arc) {
        return next_i;
      }
    }
    return path.points.size() - 1;
  } else {
    for (size_t i = target_idx; i > 0; --i) {
      const auto next_i = i - 1;
      sum_length -= calcDistance2d(path.points.at(i), path.points.at(next_i));
      if (sum_length < signed_arc) {
        return next_i;
      }
    }
    return 0;
  }
}

void clipPathLength(
  PathWithLaneId & path, const size_t target_idx, const double forward, const double backward)
{
  if (path.points.size() < 3) {
    return;
  }

  const auto start_idx = util::getIdxByArclength(path, target_idx, -backward);
  const auto end_idx = util::getIdxByArclength(path, target_idx, forward);

  const std::vector<PathPointWithLaneId> clipped_points{
    path.points.begin() + start_idx, path.points.begin() + end_idx + 1};

  path.points = clipped_points;
}

void clipPathLength(
  PathWithLaneId & path, const size_t target_idx, const BehaviorPathPlannerParameters & params)
{
  clipPathLength(path, target_idx, params.forward_path_length, params.backward_path_length);
}

std::pair<TurnIndicatorsCommand, double> getPathTurnSignal(
  const lanelet::ConstLanelets & current_lanes, const ShiftedPath & path,
  const ShiftLine & shift_line, const Pose & pose, const double & velocity,
  const BehaviorPathPlannerParameters & common_parameter)
{
  TurnIndicatorsCommand turn_signal;
  turn_signal.command = TurnIndicatorsCommand::NO_COMMAND;
  const double max_time = std::numeric_limits<double>::max();
  const double max_distance = std::numeric_limits<double>::max();
  if (path.shift_length.empty()) {
    return std::make_pair(turn_signal, max_distance);
  }
  const auto base_link2front = common_parameter.base_link2front;
  const auto vehicle_width = common_parameter.vehicle_width;
  const auto shift_to_outside = vehicle_width / 2;
  const auto turn_signal_shift_length_threshold =
    common_parameter.turn_signal_shift_length_threshold;
  const auto turn_signal_minimum_search_distance =
    common_parameter.turn_signal_minimum_search_distance;
  const auto turn_signal_search_time = common_parameter.turn_signal_search_time;
  constexpr double epsilon = 1e-6;
  const auto arc_position_current_pose = lanelet::utils::getArcCoordinates(current_lanes, pose);

  // Start turn signal when 1 or 2 is satisfied
  //  1. time to shift start point is less than prev_sec
  //  2. distance to shift point is shorter than tl_on_threshold_long

  // Turn signal on when conditions below are satisfied
  //  1. lateral offset is larger than tl_on_threshold_lat for left signal
  //                      smaller than tl_on_threshold_lat for right signal
  //  2. side point at shift start/end point cross the line
  const double distance_to_shift_start =
    std::invoke([&current_lanes, &shift_line, &arc_position_current_pose]() {
      const auto arc_position_shift_start =
        lanelet::utils::getArcCoordinates(current_lanes, shift_line.start);
      return arc_position_shift_start.length - arc_position_current_pose.length;
    });

  const auto time_to_shift_start =
    (std::abs(velocity) < epsilon) ? max_time : distance_to_shift_start / velocity;

  const double diff =
    path.shift_length.at(shift_line.end_idx) - path.shift_length.at(shift_line.start_idx);

  Pose shift_start_point = path.path.points.at(shift_line.start_idx).point.pose;
  Pose shift_end_point = path.path.points.at(shift_line.end_idx).point.pose;
  Pose left_start_point = shift_start_point;
  Pose right_start_point = shift_start_point;
  Pose left_end_point = shift_end_point;
  Pose right_end_point = shift_end_point;
  {
    const double start_yaw = tf2::getYaw(shift_line.start.orientation);
    const double end_yaw = tf2::getYaw(shift_line.end.orientation);
    left_start_point.position.x -= std::sin(start_yaw) * (shift_to_outside);
    left_start_point.position.y += std::cos(start_yaw) * (shift_to_outside);
    right_start_point.position.x -= std::sin(start_yaw) * (-shift_to_outside);
    right_start_point.position.y += std::cos(start_yaw) * (-shift_to_outside);
    left_end_point.position.x -= std::sin(end_yaw) * (shift_to_outside);
    left_end_point.position.y += std::cos(end_yaw) * (shift_to_outside);
    right_end_point.position.x -= std::sin(end_yaw) * (-shift_to_outside);
    right_end_point.position.y += std::cos(end_yaw) * (-shift_to_outside);
  }

  bool left_start_point_is_in_lane = false;
  bool right_start_point_is_in_lane = false;
  bool left_end_point_is_in_lane = false;
  bool right_end_point_is_in_lane = false;
  {
    for (const auto & llt : current_lanes) {
      if (lanelet::utils::isInLanelet(left_start_point, llt, 0.1)) {
        left_start_point_is_in_lane = true;
      }
      if (lanelet::utils::isInLanelet(right_start_point, llt, 0.1)) {
        right_start_point_is_in_lane = true;
      }
      if (lanelet::utils::isInLanelet(left_end_point, llt, 0.1)) {
        left_end_point_is_in_lane = true;
      }
      if (lanelet::utils::isInLanelet(right_end_point, llt, 0.1)) {
        right_end_point_is_in_lane = true;
      }
    }
  }

  const bool cross_line = std::invoke([&]() {
    constexpr bool temporary_set_cross_line_true =
      true;  // due to a bug. See link:
             // https://github.com/autowarefoundation/autoware.universe/pull/748
    if (temporary_set_cross_line_true) {
      return true;
    }
    return (
      left_start_point_is_in_lane != left_end_point_is_in_lane ||
      right_start_point_is_in_lane != right_end_point_is_in_lane);
  });

  if (
    time_to_shift_start < turn_signal_search_time ||
    distance_to_shift_start < turn_signal_minimum_search_distance) {
    if (diff > turn_signal_shift_length_threshold && cross_line) {
      turn_signal.command = TurnIndicatorsCommand::ENABLE_LEFT;
    } else if (diff < -turn_signal_shift_length_threshold && cross_line) {
      turn_signal.command = TurnIndicatorsCommand::ENABLE_RIGHT;
    }
  }

  // calc distance from ego vehicle front to shift end point.
  const double distance_from_vehicle_front =
    std::invoke([&current_lanes, &shift_line, &arc_position_current_pose, &base_link2front]() {
      const auto arc_position_shift_end =
        lanelet::utils::getArcCoordinates(current_lanes, shift_line.end);
      return arc_position_shift_end.length - arc_position_current_pose.length - base_link2front;
    });

  if (distance_from_vehicle_front >= 0.0) {
    return std::make_pair(turn_signal, distance_from_vehicle_front);
  }

  return std::make_pair(turn_signal, max_distance);
}

PathWithLaneId convertWayPointsToPathWithLaneId(
  const freespace_planning_algorithms::PlannerWaypoints & waypoints, const double velocity)
{
  PathWithLaneId path;
  path.header = waypoints.header;
  for (const auto & waypoint : waypoints.waypoints) {
    PathPointWithLaneId point{};
    point.point.pose = waypoint.pose.pose;
    // point.lane_id = // todo
    point.point.longitudinal_velocity_mps = (waypoint.is_back ? -1 : 1) * velocity;
    path.points.push_back(point);
  }
  return path;
}

std::vector<size_t> getReversingIndices(const PathWithLaneId & path)
{
  std::vector<size_t> indices;

  for (size_t i = 0; i < path.points.size() - 1; ++i) {
    if (
      path.points.at(i).point.longitudinal_velocity_mps *
        path.points.at(i + 1).point.longitudinal_velocity_mps <
      0) {
      indices.push_back(i);
    }
  }

  return indices;
}

std::vector<PathWithLaneId> dividePath(
  const PathWithLaneId & path, const std::vector<size_t> indices)
{
  std::vector<PathWithLaneId> divided_paths;

  if (indices.empty()) {
    divided_paths.push_back(path);
    return divided_paths;
  }

  for (size_t i = 0; i < indices.size(); ++i) {
    PathWithLaneId divided_path;
    divided_path.header = path.header;
    if (i == 0) {
      divided_path.points.insert(
        divided_path.points.end(), path.points.begin(), path.points.begin() + indices.at(i) + 1);
    } else {
      // include the point at indices.at(i - 1) and indices.at(i)
      divided_path.points.insert(
        divided_path.points.end(), path.points.begin() + indices.at(i - 1),
        path.points.begin() + indices.at(i) + 1);
    }
    divided_paths.push_back(divided_path);
  }

  PathWithLaneId divided_path;
  divided_path.header = path.header;
  divided_path.points.insert(
    divided_path.points.end(), path.points.begin() + indices.back(), path.points.end());
  divided_paths.push_back(divided_path);

  return divided_paths;
}

void correctDividedPathVelocity(std::vector<PathWithLaneId> & divided_paths)
{
  for (auto & path : divided_paths) {
    const auto is_driving_forward = motion_utils::isDrivingForward(path.points);
    if (!is_driving_forward) {
      continue;
    }

    if (*is_driving_forward) {
      for (auto & point : path.points) {
        point.point.longitudinal_velocity_mps = std::abs(point.point.longitudinal_velocity_mps);
      }
    } else {
      for (auto & point : path.points) {
        point.point.longitudinal_velocity_mps = -std::abs(point.point.longitudinal_velocity_mps);
      }
    }
    path.points.back().point.longitudinal_velocity_mps = 0.0;
  }
}

bool isCloseToPath(const PathWithLaneId & path, const Pose & pose, const double distance_threshold)
{
  for (const auto & point : path.points) {
    const auto & p = point.point.pose.position;
    const double dist = std::hypot(pose.position.x - p.x, pose.position.y - p.y);
    if (dist < distance_threshold) {
      return true;
    }
  }
  return false;
}

// only two points is supported
std::vector<double> splineTwoPoints(
  std::vector<double> base_s, std::vector<double> base_x, const double begin_diff,
  const double end_diff, std::vector<double> new_s)
{
  const double h = base_s.at(1) - base_s.at(0);

  const double c = begin_diff;
  const double d = base_x.at(0);
  const double a = (end_diff * h - 2 * base_x.at(1) + c * h + 2 * d) / std::pow(h, 3);
  const double b = (3 * base_x.at(1) - end_diff * h - 2 * c * h - 3 * d) / std::pow(h, 2);

  std::vector<double> res;
  for (const auto & s : new_s) {
    const double ds = s - base_s.at(0);
    res.push_back(d + (c + (b + a * ds) * ds) * ds);
  }

  return res;
}

std::vector<Pose> interpolatePose(
  const Pose & start_pose, const Pose & end_pose, const double resample_interval)
{
  std::vector<Pose> interpolated_poses{};  // output

  const std::vector<double> base_s{
    0, tier4_autoware_utils::calcDistance2d(start_pose.position, end_pose.position)};
  const std::vector<double> base_x{start_pose.position.x, end_pose.position.x};
  const std::vector<double> base_y{start_pose.position.y, end_pose.position.y};
  std::vector<double> new_s;

  constexpr double eps = 0.3;  // prevent overlapping
  for (double s = eps; s < base_s.back() - eps; s += resample_interval) {
    new_s.push_back(s);
  }

  const std::vector<double> interpolated_x = splineTwoPoints(
    base_s, base_x, std::cos(tf2::getYaw(start_pose.orientation)),
    std::cos(tf2::getYaw(end_pose.orientation)), new_s);
  const std::vector<double> interpolated_y = splineTwoPoints(
    base_s, base_y, std::sin(tf2::getYaw(start_pose.orientation)),
    std::sin(tf2::getYaw(end_pose.orientation)), new_s);
  for (size_t i = 0; i < interpolated_x.size(); ++i) {
    Pose pose{};
    pose = util::lerpByPose(end_pose, start_pose, (base_s.back() - new_s.at(i)) / base_s.back());
    pose.position.x = interpolated_x.at(i);
    pose.position.y = interpolated_y.at(i);
    pose.position.z = end_pose.position.z;
    interpolated_poses.push_back(pose);
  }

  return interpolated_poses;
}

}  // namespace behavior_path_planner::util

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

#include "behavior_path_planner/scene_module/side_shift/side_shift_module.hpp"

#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/util/side_shift/util.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <lanelet2_extension/utility/utilities.hpp>

#include <algorithm>
#include <memory>
#include <string>

namespace behavior_path_planner
{
using geometry_msgs::msg::Point;
using motion_utils::calcSignedArcLength;
using motion_utils::findNearestIndex;
using motion_utils::findNearestSegmentIndex;
using tier4_autoware_utils::calcDistance2d;
using tier4_autoware_utils::getPoint;

SideShiftModule::SideShiftModule(
  const std::string & name, rclcpp::Node & node,
  const std::shared_ptr<SideShiftParameters> & parameters)
: SceneModuleInterface{name, node}, parameters_{parameters}
{
  using std::placeholders::_1;

#ifdef USE_OLD_ARCHITECTURE
  lateral_offset_subscriber_ = node.create_subscription<LateralOffset>(
    "~/input/lateral_offset", 1, std::bind(&SideShiftModule::onLateralOffset, this, _1));
#endif

  // If lateral offset is subscribed, it approves side shift module automatically
  clearWaitingApproval();
}

void SideShiftModule::initVariables()
{
  reference_path_ = PathWithLaneId();
  start_pose_reset_request_ = false;
  requested_lateral_offset_ = 0.0;
  inserted_lateral_offset_ = 0.0;
  inserted_shift_line_ = ShiftLine{};
  shift_status_ = SideShiftStatus{};
  prev_output_ = ShiftedPath{};
  prev_shift_line_ = ShiftLine{};
  path_shifter_ = PathShifter{};
  resetPathCandidate();
  resetPathReference();
}

void SideShiftModule::onEntry()
{
  // write me... (Don't initialize variables, otherwise lateral offset gets zero on entry.)
  start_pose_reset_request_ = false;
#ifdef USE_OLD_ARCHITECTURE
  current_state_ = ModuleStatus::IDLE;
#endif
}

void SideShiftModule::onExit()
{
  // write me...
  initVariables();
  current_state_ = ModuleStatus::SUCCESS;
}

void SideShiftModule::setParameters(const std::shared_ptr<SideShiftParameters> & parameters)
{
  parameters_ = parameters;
}

bool SideShiftModule::isExecutionRequested() const
{
  if (current_state_ == ModuleStatus::RUNNING) {
    return true;
  }

  // If the desired offset has a non-zero value, return true as we want to execute the plan.

  const bool has_request = !isAlmostZero(requested_lateral_offset_);
  RCLCPP_DEBUG_STREAM(
    getLogger(), "ESS::isExecutionRequested() : " << std::boolalpha << has_request);

  return has_request;
}

bool SideShiftModule::isExecutionReady() const
{
  return true;  // TODO(Horibe) is it ok to say "always safe"?
}

bool SideShiftModule::isReadyForNextRequest(
  const double & min_request_time_sec, bool override_requests) const noexcept
{
  rclcpp::Time current_time = clock_->now();
  const auto interval_from_last_request_sec = current_time - last_requested_shift_change_time_;

  if (interval_from_last_request_sec.seconds() >= min_request_time_sec && !override_requests) {
    last_requested_shift_change_time_ = current_time;
    return true;
  }

  return false;
}

ModuleStatus SideShiftModule::updateState()
{
  // Never return the FAILURE. When the desired offset is zero and the vehicle is in the original
  // drivable area,this module can stop the computation and return SUCCESS.

  const auto isOffsetDiffAlmostZero = [this]() noexcept {
    const auto last_sp = path_shifter_.getLastShiftLine();
    if (last_sp) {
      const auto length = std::fabs(last_sp.get().end_shift_length);
      const auto lateral_offset = std::fabs(requested_lateral_offset_);
      const auto offset_diff = lateral_offset - length;
      if (!isAlmostZero(offset_diff)) {
        lateral_offset_change_request_ = true;
        return false;
      }
    }
    return true;
  }();

  const bool no_offset_diff = isOffsetDiffAlmostZero;
  const bool no_request = isAlmostZero(requested_lateral_offset_);

  const auto no_shifted_plan = [&]() {
    if (prev_output_.shift_length.empty()) {
      return true;
    } else {
      const auto max_planned_shift_length = std::max_element(
        prev_output_.shift_length.begin(), prev_output_.shift_length.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
      return std::abs(*max_planned_shift_length) < 0.01;
    }
  }();

  RCLCPP_DEBUG(
    getLogger(), "ESS::updateState() : no_request = %d, no_shifted_plan = %d", no_request,
    no_shifted_plan);

  if (no_request && no_shifted_plan && no_offset_diff) {
    current_state_ = ModuleStatus::SUCCESS;
  } else {
    const auto & current_lanes = util::getCurrentLanes(planner_data_);
    const auto & current_pose = planner_data_->self_odometry->pose.pose;
    const auto & inserted_shift_line_start_pose = inserted_shift_line_.start;
    const auto & inserted_shift_line_end_pose = inserted_shift_line_.end;
    const double self_to_shift_line_start_arc_length =
      behavior_path_planner::util::getSignedDistance(
        current_pose, inserted_shift_line_start_pose, current_lanes);
    const double self_to_shift_line_end_arc_length = behavior_path_planner::util::getSignedDistance(
      current_pose, inserted_shift_line_end_pose, current_lanes);
    if (self_to_shift_line_start_arc_length >= 0) {
      shift_status_ = SideShiftStatus::BEFORE_SHIFT;
    } else if (self_to_shift_line_start_arc_length < 0 && self_to_shift_line_end_arc_length > 0) {
      shift_status_ = SideShiftStatus::SHIFTING;
    } else {
      shift_status_ = SideShiftStatus::AFTER_SHIFT;
    }
    current_state_ = ModuleStatus::RUNNING;
  }

  return current_state_;
}

void SideShiftModule::updateData()
{
  if (prev_reference_.points.empty()) {
    prev_reference_ = *getPreviousModuleOutput().path;
  }

  const auto reference_pose = prev_output_.shift_length.empty()
                                ? planner_data_->self_odometry->pose.pose
                                : getUnshiftedEgoPose(prev_output_);
  const auto centerline_path = calcCenterLinePath(planner_data_, reference_pose);

  constexpr double resample_interval = 1.0;
#ifdef USE_OLD_ARCHITECTURE
  reference_path_ = util::resamplePathWithSpline(centerline_path, resample_interval);
#else
  const auto backward_extened_path = extendBackwardLength(*getPreviousModuleOutput().path);
  reference_path_ = util::resamplePathWithSpline(backward_extened_path, resample_interval);
#endif

  path_shifter_.setPath(reference_path_);

  const auto & route_handler = planner_data_->route_handler;
  const auto & p = planner_data_->parameters;

  lanelet::ConstLanelet current_lane;
  if (!route_handler->getClosestLaneletWithinRoute(reference_pose, &current_lane)) {
    RCLCPP_ERROR_THROTTLE(
      getLogger(), *clock_, 5000, "failed to find closest lanelet within route!!!");
  }

  // For current_lanes with desired length
  current_lanelets_ = route_handler->getLaneletSequence(
    current_lane, reference_pose, p.backward_path_length, p.forward_path_length);

  const size_t nearest_idx = planner_data_->findEgoIndex(path_shifter_.getReferencePath().points);
  path_shifter_.removeBehindShiftLineAndSetBaseOffset(nearest_idx);

#ifndef USE_OLD_ARCHITECTURE
  if (planner_data_->lateral_offset != nullptr) {
    if (isReadyForNextRequest(parameters_->shift_request_time_limit)) {
      lateral_offset_change_request_ = true;
      requested_lateral_offset_ = planner_data_->lateral_offset->lateral_offset;
    }
  }
#endif
}

void SideShiftModule::replaceShiftLine()
{
  auto shift_lines = path_shifter_.getShiftLines();
  if (shift_lines.size() > 0) {
    shift_lines.clear();
  }

  const auto new_sl = calcShiftLine();

  // if no conflict, then add the new point.
  shift_lines.push_back(new_sl);
  const bool new_sl_is_same_with_previous =
    new_sl.end_shift_length == prev_shift_line_.end_shift_length;

  if (!new_sl_is_same_with_previous) {
    prev_shift_line_ = new_sl;
  }

  // set to path_shifter
  path_shifter_.setShiftLines(shift_lines);
  lateral_offset_change_request_ = false;
  inserted_lateral_offset_ = requested_lateral_offset_;
  inserted_shift_line_ = new_sl;

  return;
}

BehaviorModuleOutput SideShiftModule::plan()
{
  // Replace shift line
  if (
    lateral_offset_change_request_ && ((shift_status_ == SideShiftStatus::BEFORE_SHIFT) ||
                                       (shift_status_ == SideShiftStatus::AFTER_SHIFT))) {
    replaceShiftLine();
  } else if (shift_status_ != SideShiftStatus::BEFORE_SHIFT) {
    RCLCPP_DEBUG(getLogger(), "ego is shifting");
  } else {
    RCLCPP_DEBUG(getLogger(), "change is not requested");
  }

  // Refine path
  ShiftedPath shifted_path;
  path_shifter_.generate(&shifted_path);

  // Reset orientation
  setOrientation(&shifted_path.path);

  adjustDrivableArea(&shifted_path);

  BehaviorModuleOutput output;
  output.path = std::make_shared<PathWithLaneId>(shifted_path.path);
  output.reference_path = getPreviousModuleOutput().reference_path;

  prev_output_ = shifted_path;
  path_reference_ = getPreviousModuleOutput().reference_path;

  return output;
}

CandidateOutput SideShiftModule::planCandidate() const
{
  auto path_shifter_local = path_shifter_;

  path_shifter_local.addShiftLine(calcShiftLine());

  // Refine path
  ShiftedPath shifted_path;
  path_shifter_local.generate(&shifted_path);

  // Reset orientation
  setOrientation(&shifted_path.path);

  return CandidateOutput(shifted_path.path);
}

BehaviorModuleOutput SideShiftModule::planWaitingApproval()
{
  // Refine path
  ShiftedPath shifted_path;
  path_shifter_.generate(&shifted_path);

  // Reset orientation
  setOrientation(&shifted_path.path);

  adjustDrivableArea(&shifted_path);

  BehaviorModuleOutput output;
  output.path = std::make_shared<PathWithLaneId>(shifted_path.path);
  output.reference_path = getPreviousModuleOutput().reference_path;
  path_candidate_ = std::make_shared<PathWithLaneId>(planCandidate().path_candidate);
  path_reference_ = getPreviousModuleOutput().reference_path;

  prev_output_ = shifted_path;

  waitApproval();

  return output;
}

#ifdef USE_OLD_ARCHITECTURE
void SideShiftModule::onLateralOffset(const LateralOffset::ConstSharedPtr lateral_offset_msg)
{
  const double new_lateral_offset = lateral_offset_msg->lateral_offset;

  RCLCPP_DEBUG(
    getLogger(), "onLateralOffset start : lateral offset current = %f, new = &%f",
    requested_lateral_offset_, new_lateral_offset);

  // offset is not changed.
  if (std::abs(inserted_lateral_offset_ - new_lateral_offset) < 1e-4) {
    return;
  }

  if (parameters_->shift_request_time_limit < parameters_->time_to_start_shifting) {
    RCLCPP_DEBUG(
      getLogger(), "Shift request time might be too low. Generated trajectory might be wavy");
  }
  // new offset is requested.
  if (isReadyForNextRequest(parameters_->shift_request_time_limit)) {
    lateral_offset_change_request_ = true;
    requested_lateral_offset_ = new_lateral_offset;
  }
}
#endif

ShiftLine SideShiftModule::calcShiftLine() const
{
  const auto & p = parameters_;
  const auto ego_speed = std::abs(planner_data_->self_odometry->twist.twist.linear.x);

  const double dist_to_start =
    std::max(p->min_distance_to_start_shifting, ego_speed * p->time_to_start_shifting);

  const double dist_to_end = [&]() {
    const double shift_length = requested_lateral_offset_ - getClosestShiftLength();
    const double jerk_shifting_distance = path_shifter_.calcLongitudinalDistFromJerk(
      shift_length, p->shifting_lateral_jerk, std::max(ego_speed, p->min_shifting_speed));
    const double shifting_distance = std::max(jerk_shifting_distance, p->min_shifting_distance);
    const double dist_to_end = dist_to_start + shifting_distance;
    RCLCPP_DEBUG(
      getLogger(), "min_distance_to_start_shifting = %f, dist_to_start = %f, dist_to_end = %f",
      parameters_->min_distance_to_start_shifting, dist_to_start, dist_to_end);
    return dist_to_end;
  }();

  const size_t nearest_idx = planner_data_->findEgoIndex(reference_path_.points);
  ShiftLine shift_line;
  shift_line.end_shift_length = requested_lateral_offset_;
  shift_line.start_idx = util::getIdxByArclength(reference_path_, nearest_idx, dist_to_start);
  shift_line.start = reference_path_.points.at(shift_line.start_idx).point.pose;
  shift_line.end_idx = util::getIdxByArclength(reference_path_, nearest_idx, dist_to_end);
  shift_line.end = reference_path_.points.at(shift_line.end_idx).point.pose;

  return shift_line;
}

double SideShiftModule::getClosestShiftLength() const
{
  if (prev_output_.shift_length.empty()) {
    return 0.0;
  }

  const auto ego_point = planner_data_->self_odometry->pose.pose.position;
  const auto closest = motion_utils::findNearestIndex(prev_output_.path.points, ego_point);
  return prev_output_.shift_length.at(closest);
}

void SideShiftModule::adjustDrivableArea(ShiftedPath * path) const
{
  const auto itr = std::minmax_element(path->shift_length.begin(), path->shift_length.end());

  constexpr double threshold = 0.1;
  constexpr double margin = 0.5;
  const double left_offset = std::max(
    *itr.second + (*itr.first > threshold ? margin : 0.0),
    parameters_->drivable_area_left_bound_offset);
  const double right_offset = -std::min(
    *itr.first - (*itr.first < -threshold ? margin : 0.0),
    -parameters_->drivable_area_right_bound_offset);

  const auto drivable_lanes = util::generateDrivableLanes(current_lanelets_);
  const auto shorten_lanes = util::cutOverlappedLanes(path->path, drivable_lanes);
  const auto expanded_lanes = util::expandLanelets(
    shorten_lanes, left_offset, right_offset, parameters_->drivable_area_types_to_skip);

  {
    const auto & p = planner_data_->parameters;
    util::generateDrivableArea(path->path, expanded_lanes, p.vehicle_length, planner_data_);
  }
}

// NOTE: this function is ported from avoidance.
Pose SideShiftModule::getUnshiftedEgoPose(const ShiftedPath & prev_path) const
{
  const auto ego_pose = getEgoPose();
  if (prev_path.path.points.empty()) {
    return ego_pose;
  }

  // un-shifted fot current ideal pose
  const auto closest = motion_utils::findNearestIndex(prev_path.path.points, ego_pose.position);

  Pose unshifted_pose = ego_pose;

  util::shiftPose(&unshifted_pose, -prev_path.shift_length.at(closest));
  unshifted_pose.orientation = ego_pose.orientation;

  return unshifted_pose;
}

PathWithLaneId SideShiftModule::extendBackwardLength(const PathWithLaneId & original_path) const
{
  // special for avoidance: take behind distance upt ot shift-start-point if it exist.
  const auto longest_dist_to_shift_point = [&]() {
    double max_dist = 0.0;
    for (const auto & pnt : path_shifter_.getShiftLines()) {
      max_dist = std::max(max_dist, calcDistance2d(getEgoPose(), pnt.start));
    }
    return max_dist;
  }();

  const auto extra_margin = 10.0;  // Since distance does not consider arclength, but just line.
  const auto backward_length = std::max(
    planner_data_->parameters.backward_path_length, longest_dist_to_shift_point + extra_margin);

  const size_t orig_ego_idx = findNearestIndex(original_path.points, getEgoPose().position);
  const size_t prev_ego_idx = findNearestSegmentIndex(
    prev_reference_.points, getPoint(original_path.points.at(orig_ego_idx)));

  size_t clip_idx = 0;
  for (size_t i = 0; i < prev_ego_idx; ++i) {
    if (backward_length > calcSignedArcLength(prev_reference_.points, clip_idx, prev_ego_idx)) {
      break;
    }
    clip_idx = i;
  }

  PathWithLaneId extended_path{};
  {
    extended_path.points.insert(
      extended_path.points.end(), prev_reference_.points.begin() + clip_idx,
      prev_reference_.points.begin() + prev_ego_idx);
  }

  {
    extended_path.points.insert(
      extended_path.points.end(), original_path.points.begin() + orig_ego_idx,
      original_path.points.end());
  }

  return extended_path;
}

// NOTE: this function is ported from avoidance.
PathWithLaneId SideShiftModule::calcCenterLinePath(
  const std::shared_ptr<const PlannerData> & planner_data, const Pose & pose) const
{
  const auto & p = planner_data->parameters;
  const auto & route_handler = planner_data->route_handler;

  PathWithLaneId centerline_path;

  // special for avoidance: take behind distance upt ot shift-start-point if it exist.
  const auto longest_dist_to_shift_line = [&]() {
    double max_dist = 0.0;
    for (const auto & pnt : path_shifter_.getShiftLines()) {
      max_dist = std::max(max_dist, tier4_autoware_utils::calcDistance2d(getEgoPose(), pnt.start));
    }
    return max_dist;
  }();
  const auto extra_margin = 10.0;  // Since distance does not consider arclength, but just line.
  const auto backward_length =
    std::max(p.backward_path_length, longest_dist_to_shift_line + extra_margin);

  RCLCPP_DEBUG(
    getLogger(),
    "p.backward_path_length = %f, longest_dist_to_shift_line = %f, backward_length = %f",
    p.backward_path_length, longest_dist_to_shift_line, backward_length);

  const lanelet::ConstLanelets current_lanes =
    util::calcLaneAroundPose(route_handler, pose, p.forward_path_length, backward_length);
  centerline_path = util::getCenterLinePath(
    *route_handler, current_lanes, pose, backward_length, p.forward_path_length, p);

  centerline_path.header = route_handler->getRouteHeader();

  return centerline_path;
}

}  // namespace behavior_path_planner

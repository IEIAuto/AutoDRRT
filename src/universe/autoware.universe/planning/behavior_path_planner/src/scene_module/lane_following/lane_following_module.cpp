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

#include "behavior_path_planner/scene_module/lane_following/lane_following_module.hpp"

#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <lanelet2_extension/utility/utilities.hpp>

#include <memory>
#include <string>

namespace behavior_path_planner
{
LaneFollowingModule::LaneFollowingModule(
  const std::string & name, rclcpp::Node & node,
  const std::shared_ptr<LaneFollowingParameters> & parameters)
: SceneModuleInterface{name, node}, parameters_{parameters}
{
  initParam();
}

void LaneFollowingModule::initParam()
{
  clearWaitingApproval();  // no need approval
}

bool LaneFollowingModule::isExecutionRequested() const { return true; }

bool LaneFollowingModule::isExecutionReady() const { return true; }

BT::NodeStatus LaneFollowingModule::updateState()
{
  current_state_ = BT::NodeStatus::SUCCESS;
  return current_state_;
}

BehaviorModuleOutput LaneFollowingModule::plan()
{
  BehaviorModuleOutput output;
  output.path = std::make_shared<PathWithLaneId>(getReferencePath());
  return output;
}
CandidateOutput LaneFollowingModule::planCandidate() const
{
  return CandidateOutput(getReferencePath());
}
void LaneFollowingModule::onEntry()
{
  initParam();
  current_state_ = BT::NodeStatus::RUNNING;
  RCLCPP_DEBUG(getLogger(), "LANE_FOLLOWING onEntry");
}
void LaneFollowingModule::onExit()
{
  initParam();
  current_state_ = BT::NodeStatus::SUCCESS;
  RCLCPP_DEBUG(getLogger(), "LANE_FOLLOWING onExit");
}

void LaneFollowingModule::setParameters(const std::shared_ptr<LaneFollowingParameters> & parameters)
{
  parameters_ = parameters;
}

lanelet::ConstLanelets getLaneletsFromPath(
  const PathWithLaneId & path, const std::shared_ptr<route_handler::RouteHandler> & route_handler)
{
  std::vector<int64_t> unique_lanelet_ids;
  for (const auto & p : path.points) {
    const auto & lane_ids = p.lane_ids;
    for (const auto & lane_id : lane_ids) {
      if (
        std::find(unique_lanelet_ids.begin(), unique_lanelet_ids.end(), lane_id) ==
        unique_lanelet_ids.end()) {
        unique_lanelet_ids.push_back(lane_id);
      }
    }
  }

  lanelet::ConstLanelets lanelets;
  for (const auto & lane_id : unique_lanelet_ids) {
    lanelets.push_back(route_handler->getLaneletsFromId(lane_id));
  }

  return lanelets;
}

PathWithLaneId LaneFollowingModule::getReferencePath() const
{
  PathWithLaneId reference_path{};

  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = planner_data_->self_odometry->pose.pose;
  const auto p = planner_data_->parameters;

  // Set header
  reference_path.header = route_handler->getRouteHeader();

  lanelet::ConstLanelet current_lane;
  if (!planner_data_->route_handler->getClosestLaneletWithinRoute(current_pose, &current_lane)) {
    RCLCPP_ERROR_THROTTLE(
      getLogger(), *clock_, 5000, "failed to find closest lanelet within route!!!");
    return reference_path;  // TODO(Horibe)
  }

  // For current_lanes with desired length
  const auto current_lanes = planner_data_->route_handler->getLaneletSequence(
    current_lane, current_pose, p.backward_path_length, p.forward_path_length);

  if (current_lanes.empty()) {
    return reference_path;
  }

  // calculate path with backward margin to avoid end points' instability by spline interpolation
  constexpr double extra_margin = 10.0;
  const double backward_length = p.backward_path_length + extra_margin;
  const auto current_lanes_with_backward_margin =
    util::calcLaneAroundPose(route_handler, current_pose, p.forward_path_length, backward_length);
  reference_path = util::getCenterLinePath(
    *route_handler, current_lanes_with_backward_margin, current_pose, backward_length,
    p.forward_path_length, p);

  // clip backward length
  const size_t current_seg_idx = planner_data_->findEgoSegmentIndex(reference_path.points);
  util::clipPathLength(
    reference_path, current_seg_idx, p.forward_path_length, p.backward_path_length);
  const auto drivable_lanelets = getLaneletsFromPath(reference_path, route_handler);
  const auto drivable_lanes = util::generateDrivableLanes(drivable_lanelets);

  {
    const int num_lane_change =
      std::abs(route_handler->getNumLaneToPreferredLane(current_lanes.back()));

    const double lane_change_buffer = util::calcLaneChangeBuffer(p, num_lane_change);

    reference_path = util::setDecelerationVelocity(
      *route_handler, reference_path, current_lanes, parameters_->lane_change_prepare_duration,
      lane_change_buffer);
  }

  const auto shorten_lanes = util::cutOverlappedLanes(reference_path, drivable_lanes);

  const auto expanded_lanes = util::expandLanelets(
    shorten_lanes, parameters_->drivable_area_left_bound_offset,
    parameters_->drivable_area_right_bound_offset, parameters_->drivable_area_types_to_skip);

  util::generateDrivableArea(reference_path, expanded_lanes, p.vehicle_length, planner_data_);

  return reference_path;
}
}  // namespace behavior_path_planner

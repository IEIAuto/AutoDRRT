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

#ifndef BEHAVIOR_PATH_PLANNER__SCENE_MODULE__AVOIDANCE__AVOIDANCE_MODULE_HPP_
#define BEHAVIOR_PATH_PLANNER__SCENE_MODULE__AVOIDANCE__AVOIDANCE_MODULE_HPP_

#include "behavior_path_planner/scene_module/scene_module_interface.hpp"
#include "behavior_path_planner/scene_module/scene_module_visitor.hpp"
#include "behavior_path_planner/util/avoidance/avoidance_module_data.hpp"

#include <rclcpp/rclcpp.hpp>

#include <autoware_auto_perception_msgs/msg/predicted_object.hpp>
#include <autoware_auto_planning_msgs/msg/path_with_lane_id.hpp>
#include <autoware_auto_vehicle_msgs/msg/turn_indicators_command.hpp>
#include <tier4_planning_msgs/msg/avoidance_debug_msg.hpp>
#include <tier4_planning_msgs/msg/avoidance_debug_msg_array.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace behavior_path_planner
{

using motion_utils::calcSignedArcLength;
using motion_utils::findNearestIndex;

using tier4_planning_msgs::msg::AvoidanceDebugMsg;

class AvoidanceModule : public SceneModuleInterface
{
public:
#ifdef USE_OLD_ARCHITECTURE
  AvoidanceModule(
    const std::string & name, rclcpp::Node & node, std::shared_ptr<AvoidanceParameters> parameters);
#else
  AvoidanceModule(
    const std::string & name, rclcpp::Node & node, std::shared_ptr<AvoidanceParameters> parameters,
    std::shared_ptr<RTCInterface> & rtc_interface_left,
    std::shared_ptr<RTCInterface> & rtc_interface_right);
#endif

  bool isExecutionRequested() const override;
  bool isExecutionReady() const override;
  ModuleStatus updateState() override;
  ModuleStatus getNodeStatusWhileWaitingApproval() const override { return ModuleStatus::SUCCESS; }
  BehaviorModuleOutput plan() override;
  CandidateOutput planCandidate() const override;
  BehaviorModuleOutput planWaitingApproval() override;
  void onEntry() override;
  void onExit() override;
  void updateData() override;
  void acceptVisitor(const std::shared_ptr<SceneModuleVisitor> & visitor) const override;

#ifndef USE_OLD_ARCHITECTURE
  void updateModuleParams(const std::shared_ptr<AvoidanceParameters> & parameters)
  {
    parameters_ = parameters;
  }
#endif

  void publishRTCStatus() override
  {
    rtc_interface_left_->publishCooperateStatus(clock_->now());
    rtc_interface_right_->publishCooperateStatus(clock_->now());
  }

  bool isActivated() override
  {
    if (rtc_interface_left_->isRegistered(uuid_left_)) {
      return rtc_interface_left_->isActivated(uuid_left_);
    }
    if (rtc_interface_right_->isRegistered(uuid_right_)) {
      return rtc_interface_right_->isActivated(uuid_right_);
    }
    return false;
  }

  void lockRTCCommand() override
  {
    rtc_interface_left_->lockCommandUpdate();
    rtc_interface_right_->lockCommandUpdate();
  }

  void unlockRTCCommand() override
  {
    rtc_interface_left_->unlockCommandUpdate();
    rtc_interface_right_->unlockCommandUpdate();
  }
  std::shared_ptr<AvoidanceDebugMsgArray> get_debug_msg_array() const;

private:
  struct RegisteredShiftLine
  {
    UUID uuid;
    Pose start_pose;
    Pose finish_pose;
  };
  using RegisteredShiftLineArray = std::vector<RegisteredShiftLine>;

  std::shared_ptr<AvoidanceParameters> parameters_;

  AvoidancePlanningData avoidance_data_;

  PathShifter path_shifter_;

  std::shared_ptr<RTCInterface> rtc_interface_left_;
  std::shared_ptr<RTCInterface> rtc_interface_right_;

  RegisteredShiftLineArray left_shift_array_;
  RegisteredShiftLineArray right_shift_array_;
  UUID candidate_uuid_;
  UUID uuid_left_;
  UUID uuid_right_;

  void updateCandidateRTCStatus(const CandidateOutput & candidate)
  {
    if (candidate.lateral_shift > 0.0) {
      rtc_interface_left_->updateCooperateStatus(
        uuid_left_, isExecutionReady(), candidate.start_distance_to_path_change,
        candidate.finish_distance_to_path_change, clock_->now());
      candidate_uuid_ = uuid_left_;
      return;
    }
    if (candidate.lateral_shift < 0.0) {
      rtc_interface_right_->updateCooperateStatus(
        uuid_right_, isExecutionReady(), candidate.start_distance_to_path_change,
        candidate.finish_distance_to_path_change, clock_->now());
      candidate_uuid_ = uuid_right_;
      return;
    }

    RCLCPP_WARN_STREAM(
      getLogger(),
      "Direction is UNKNOWN, start_distance = " << candidate.start_distance_to_path_change);
  }

  void updateRegisteredRTCStatus(const PathWithLaneId & path)
  {
    const Point ego_position = planner_data_->self_odometry->pose.pose.position;

    for (const auto & left_shift : left_shift_array_) {
      const double start_distance =
        calcSignedArcLength(path.points, ego_position, left_shift.start_pose.position);
      const double finish_distance =
        calcSignedArcLength(path.points, ego_position, left_shift.finish_pose.position);
      rtc_interface_left_->updateCooperateStatus(
        left_shift.uuid, true, start_distance, finish_distance, clock_->now());
      if (finish_distance > -1.0e-03) {
        steering_factor_interface_ptr_->updateSteeringFactor(
          {left_shift.start_pose, left_shift.finish_pose}, {start_distance, finish_distance},
          SteeringFactor::AVOIDANCE_PATH_CHANGE, SteeringFactor::LEFT, SteeringFactor::TURNING, "");
      }
    }

    for (const auto & right_shift : right_shift_array_) {
      const double start_distance =
        calcSignedArcLength(path.points, ego_position, right_shift.start_pose.position);
      const double finish_distance =
        calcSignedArcLength(path.points, ego_position, right_shift.finish_pose.position);
      rtc_interface_right_->updateCooperateStatus(
        right_shift.uuid, true, start_distance, finish_distance, clock_->now());
      if (finish_distance > -1.0e-03) {
        steering_factor_interface_ptr_->updateSteeringFactor(
          {right_shift.start_pose, right_shift.finish_pose}, {start_distance, finish_distance},
          SteeringFactor::AVOIDANCE_PATH_CHANGE, SteeringFactor::RIGHT, SteeringFactor::TURNING,
          "");
      }
    }
  }

  void removeRTCStatus() override
  {
    rtc_interface_left_->clearCooperateStatus();
    rtc_interface_right_->clearCooperateStatus();
  }

  void removeCandidateRTCStatus()
  {
    if (rtc_interface_left_->isRegistered(candidate_uuid_)) {
      rtc_interface_left_->removeCooperateStatus(candidate_uuid_);
    } else if (rtc_interface_right_->isRegistered(candidate_uuid_)) {
      rtc_interface_right_->removeCooperateStatus(candidate_uuid_);
    }
  }

  void removePreviousRTCStatusLeft()
  {
    if (rtc_interface_left_->isRegistered(uuid_left_)) {
      rtc_interface_left_->removeCooperateStatus(uuid_left_);
    }
  }

  void removePreviousRTCStatusRight()
  {
    if (rtc_interface_right_->isRegistered(uuid_right_)) {
      rtc_interface_right_->removeCooperateStatus(uuid_right_);
    }
  }

  /**
   * object pre-process
   */
  void fillAvoidanceTargetObjects(AvoidancePlanningData & data, DebugData & debug) const;

  void fillObjectEnvelopePolygon(const Pose & closest_pose, ObjectData & object_data) const;

  void fillObjectMovingTime(ObjectData & object_data) const;

  void compensateDetectionLost(
    ObjectDataArray & target_objects, ObjectDataArray & other_objects) const;

  void fillShiftLine(AvoidancePlanningData & data, DebugData & debug) const;

  void fillDebugData(const AvoidancePlanningData & data, DebugData & debug) const;

  // data used in previous planning
  ShiftedPath prev_output_;
  ShiftedPath prev_linear_shift_path_;  // used for shift point check
  PathWithLaneId prev_reference_;

  // for raw_shift_line registration
  AvoidLineArray registered_raw_shift_lines_;
  AvoidLineArray current_raw_shift_lines_;
  void registerRawShiftLines(const AvoidLineArray & future_registered);
  void updateRegisteredRawShiftLines();

  // -- for state management --
  bool is_avoidance_maneuver_starts;
  bool isAvoidanceManeuverRunning();
  bool isAvoidancePlanRunning() const;

  // -- for pre-processing --
  void initVariables();
  AvoidancePlanningData calcAvoidancePlanningData(DebugData & debug) const;

  ObjectDataArray registered_objects_;
  void updateRegisteredObject(const ObjectDataArray & objects);

  // ========= shift line generator ======

  AvoidLineArray calcRawShiftLinesFromObjects(
    AvoidancePlanningData & data, DebugData & debug) const;

  AvoidLineArray applyPreProcessToRawShiftLines(
    AvoidLineArray & current_raw_shift_points, DebugData & debug) const;

  double getShiftLength(
    const ObjectData & object, const bool & is_object_on_right, const double & avoid_margin) const;

  // shift point generation: combiner
  AvoidLineArray combineRawShiftLinesWithUniqueCheck(
    const AvoidLineArray & base_lines, const AvoidLineArray & added_lines) const;

  // shift point generation: merger
  AvoidLineArray mergeShiftLines(const AvoidLineArray & raw_shift_lines, DebugData & debug) const;
  void generateTotalShiftLine(
    const AvoidLineArray & avoid_points, ShiftLineData & shift_line_data) const;
  AvoidLineArray extractShiftLinesFromLine(ShiftLineData & shift_line_data) const;
  std::vector<size_t> calcParentIds(
    const AvoidLineArray & parent_candidates, const AvoidLine & child) const;

  // shift point generation: trimmers
  AvoidLineArray trimShiftLine(const AvoidLineArray & shift_lines, DebugData & debug) const;
  void quantizeShiftLine(AvoidLineArray & shift_lines, const double interval) const;
  void trimSmallShiftLine(AvoidLineArray & shift_lines, const double shift_diff_thres) const;
  void trimSimilarGradShiftLine(AvoidLineArray & shift_lines, const double threshold) const;
  void trimMomentaryReturn(AvoidLineArray & shift_lines) const;
  void trimTooSharpShift(AvoidLineArray & shift_lines) const;
  void trimSharpReturn(AvoidLineArray & shift_lines) const;

  // shift point generation: return-shift generator
  void addReturnShiftLineFromEgo(
    AvoidLineArray & sl_candidates, AvoidLineArray & current_raw_shift_lines) const;

  // -- for shift point operations --
  void alignShiftLinesOrder(
    AvoidLineArray & shift_lines, const bool recalculate_start_length = true) const;
  AvoidLineArray fillAdditionalInfo(const AvoidLineArray & shift_lines) const;
  AvoidLine fillAdditionalInfo(const AvoidLine & shift_line) const;
  AvoidLine getNonStraightShiftLine(const AvoidLineArray & shift_lines) const;
  void fillAdditionalInfoFromPoint(AvoidLineArray & shift_lines) const;
  void fillAdditionalInfoFromLongitudinal(AvoidLineArray & shift_lines) const;

  // -- for new shift point approval --
  AvoidLineArray findNewShiftLine(
    const AvoidLineArray & shift_lines, const PathShifter & shifter) const;
  void addShiftLineIfApproved(const AvoidLineArray & point);
  void addNewShiftLines(PathShifter & path_shifter, const AvoidLineArray & shift_lines) const;

  // -- path generation --
  ShiftedPath generateAvoidancePath(PathShifter & shifter) const;
  void generateExtendedDrivableArea(PathWithLaneId & path) const;

  // -- velocity planning --
  std::shared_ptr<double> ego_velocity_starting_avoidance_ptr_;
  void modifyPathVelocityToPreventAccelerationOnAvoidance(ShiftedPath & shifted_path);

  // turn signal
  TurnSignalInfo calcTurnSignalInfo(const ShiftedPath & path) const;

  // intersection (old)
  boost::optional<AvoidLine> calcIntersectionShiftLine(const AvoidancePlanningData & data) const;

  bool isTargetObjectType(const PredictedObject & object) const;

  // debug
  mutable DebugData debug_data_;
  mutable std::shared_ptr<AvoidanceDebugMsgArray> debug_msg_ptr_;
  void setDebugData(
    const AvoidancePlanningData & data, const PathShifter & shifter, const DebugData & debug) const;
  void updateAvoidanceDebugData(std::vector<AvoidanceDebugMsg> & avoidance_debug_msg_array) const;
  mutable std::vector<AvoidanceDebugMsg> debug_avoidance_initializer_for_shift_line_;
  mutable rclcpp::Time debug_avoidance_initializer_for_shift_line_time_;

  double getLateralMarginFromVelocity(const double velocity) const;

  double getRSSLongitudinalDistance(
    const double v_ego, const double v_obj, const bool is_front_object) const;

  ObjectDataArray getAdjacentLaneObjects(const lanelet::ConstLanelets & adjacent_lanes) const;

  // ========= plan ======================

  AvoidanceState updateEgoState(const AvoidancePlanningData & data) const;

  void updateEgoBehavior(const AvoidancePlanningData & data, ShiftedPath & path);

  void insertWaitPoint(const bool use_constraints_for_decel, ShiftedPath & shifted_path) const;

  void insertPrepareVelocity(const bool avoidable, ShiftedPath & shifted_path) const;

  void insertYieldVelocity(ShiftedPath & shifted_path) const;

  void removeAllRegisteredShiftPoints(PathShifter & path_shifter)
  {
    current_raw_shift_lines_.clear();
    registered_raw_shift_lines_.clear();
    path_shifter.setShiftLines(ShiftLineArray{});
  }

  void postProcess(PathShifter & path_shifter) const
  {
    const size_t nearest_idx = planner_data_->findEgoIndex(path_shifter.getReferencePath().points);
    path_shifter.removeBehindShiftLineAndSetBaseOffset(nearest_idx);
  }

  double getFeasibleDecelDistance(const double target_velocity) const;

  double getMildDecelDistance(const double target_velocity) const;

  double getRelativeLengthFromPath(const AvoidLine & avoid_line) const;

  // ========= safety check ==============

  lanelet::ConstLanelets getAdjacentLane(
    const PathShifter & path_shifter, const double forward_distance,
    const double backward_distance) const;

  bool isSafePath(
    const PathShifter & path_shifter, ShiftedPath & shifted_path, DebugData & debug) const;

  bool isSafePath(
    const PathWithLaneId & path, const lanelet::ConstLanelets & check_lanes,
    DebugData & debug) const;

  bool isEnoughMargin(
    const PathPointWithLaneId & p_ego, const double t, const ObjectData & object,
    MarginData & margin_data) const;

  // ========= helper functions ==========

  double getEgoSpeed() const
  {
    return std::abs(planner_data_->self_odometry->twist.twist.linear.x);
  }

  double getNominalAvoidanceEgoSpeed() const
  {
    return std::max(getEgoSpeed(), parameters_->min_nominal_avoidance_speed);
  }

  double getSharpAvoidanceEgoSpeed() const
  {
    return std::max(getEgoSpeed(), parameters_->min_sharp_avoidance_speed);
  }

  float getMinimumAvoidanceEgoSpeed() const { return parameters_->target_velocity_matrix.front(); }

  float getMaximumAvoidanceEgoSpeed() const
  {
    return parameters_->target_velocity_matrix.at(parameters_->col_size - 1);
  }

  double getNominalPrepareDistance() const
  {
    const auto & p = parameters_;
    const auto epsilon_m = 0.01;  // for floating error to pass "has_enough_distance" check.
    const auto nominal_distance =
      std::max(getEgoSpeed() * p->prepare_time, p->min_prepare_distance);
    return nominal_distance + epsilon_m;
  }

  double getNominalAvoidanceDistance(const double shift_length) const
  {
    const auto & p = parameters_;
    const auto distance_by_jerk = PathShifter::calcLongitudinalDistFromJerk(
      shift_length, p->nominal_lateral_jerk, getNominalAvoidanceEgoSpeed());

    return std::max(p->min_avoidance_distance, distance_by_jerk);
  }

  double getMinimumAvoidanceDistance(const double shift_length) const
  {
    const auto & p = parameters_;
    const auto distance_by_jerk = path_shifter_.calcLongitudinalDistFromJerk(
      shift_length, p->nominal_lateral_jerk, getMinimumAvoidanceEgoSpeed());

    return std::max(p->min_avoidance_distance, distance_by_jerk);
  }

  double getSharpAvoidanceDistance(const double shift_length) const
  {
    const auto & p = parameters_;
    const auto distance_by_jerk = PathShifter::calcLongitudinalDistFromJerk(
      shift_length, p->max_lateral_jerk, getSharpAvoidanceEgoSpeed());

    return std::max(p->min_avoidance_distance, distance_by_jerk);
  }

  double getRightShiftBound() const
  {
    // TODO(Horibe) write me. Real lane boundary must be considered here.
    return -parameters_->max_right_shift_length;
  }

  double getLeftShiftBound() const
  {
    // TODO(Horibe) write me. Real lane boundary must be considered here.
    return parameters_->max_left_shift_length;
  }

  double getCurrentShift() const
  {
    return prev_output_.shift_length.at(
      findNearestIndex(prev_output_.path.points, getEgoPosition()));
  }

  double getCurrentLinearShift() const
  {
    return prev_linear_shift_path_.shift_length.at(
      findNearestIndex(prev_linear_shift_path_.path.points, getEgoPosition()));
  }

  double getCurrentBaseShift() const { return path_shifter_.getBaseOffset(); }

  Point getEgoPosition() const { return planner_data_->self_odometry->pose.pose.position; }

  Pose getEgoPose() const { return planner_data_->self_odometry->pose.pose; }

  Pose getUnshiftedEgoPose(const ShiftedPath & prev_path) const;

  PathWithLaneId extendBackwardLength(const PathWithLaneId & original_path) const;

  PathWithLaneId calcCenterLinePath(
    const std::shared_ptr<const PlannerData> & planner_data, const Pose & pose) const;

  // TODO(Horibe): think later.
  // for unique ID
  mutable uint64_t original_unique_id = 0;  // TODO(Horibe) remove mutable
  uint64_t getOriginalShiftLineUniqueId() const { return original_unique_id++; }

  /**
   * avoidance module misc data
   */
  mutable ObjectDataArray stopped_objects_;
};

}  // namespace behavior_path_planner

#endif  // BEHAVIOR_PATH_PLANNER__SCENE_MODULE__AVOIDANCE__AVOIDANCE_MODULE_HPP_

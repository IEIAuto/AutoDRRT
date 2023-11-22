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

#include "behavior_path_planner/scene_module/avoidance/avoidance_module.hpp"

#include "behavior_path_planner/marker_util/avoidance/debug.hpp"
#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/scene_module/scene_module_visitor.hpp"
#include "behavior_path_planner/util/avoidance/util.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <tier4_planning_msgs/msg/avoidance_debug_factor.hpp>
#include <tier4_planning_msgs/msg/avoidance_debug_msg.hpp>
#include <tier4_planning_msgs/msg/avoidance_debug_msg_array.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

// set as macro so that calling function name will be printed.
// debug print is heavy. turn on only when debugging.
#define DEBUG_PRINT(...) \
  RCLCPP_DEBUG_EXPRESSION(getLogger(), parameters_->print_debug_info, __VA_ARGS__)
#define printShiftLines(p, msg) DEBUG_PRINT("[%s] %s", msg, toStrInfo(p).c_str())

namespace behavior_path_planner
{
using motion_utils::calcLongitudinalOffsetPose;
using motion_utils::calcSignedArcLength;
using motion_utils::findNearestIndex;
using motion_utils::findNearestSegmentIndex;
using tier4_autoware_utils::calcDistance2d;
using tier4_autoware_utils::calcInterpolatedPose;
using tier4_autoware_utils::calcLateralDeviation;
using tier4_autoware_utils::calcLongitudinalDeviation;
using tier4_autoware_utils::calcYawDeviation;
using tier4_autoware_utils::getPoint;
using tier4_autoware_utils::getPose;
using tier4_planning_msgs::msg::AvoidanceDebugFactor;

namespace
{
bool isEndPointsConnected(
  const lanelet::ConstLanelet & left_lane, const lanelet::ConstLanelet & right_lane)
{
  const auto & left_back_point_2d = right_lane.leftBound2d().back().basicPoint();
  const auto & right_back_point_2d = left_lane.rightBound2d().back().basicPoint();

  constexpr double epsilon = 1e-5;
  return (right_back_point_2d - left_back_point_2d).norm() < epsilon;
}

template <typename T>
void pushUniqueVector(T & base_vector, const T & additional_vector)
{
  base_vector.insert(base_vector.end(), additional_vector.begin(), additional_vector.end());
}
}  // namespace

#ifdef USE_OLD_ARCHITECTURE
AvoidanceModule::AvoidanceModule(
  const std::string & name, rclcpp::Node & node, std::shared_ptr<AvoidanceParameters> parameters)
: SceneModuleInterface{name, node},
  parameters_{std::move(parameters)},
  uuid_left_{generateUUID()},
  uuid_right_{generateUUID()}
{
  using std::placeholders::_1;
  rtc_interface_left_ = std::make_shared<RTCInterface>(&node, "avoidance_left"),
  rtc_interface_right_ = std::make_shared<RTCInterface>(&node, "avoidance_right"),
  steering_factor_interface_ptr_ = std::make_unique<SteeringFactorInterface>(&node, "avoidance");
}
#else
AvoidanceModule::AvoidanceModule(
  const std::string & name, rclcpp::Node & node, std::shared_ptr<AvoidanceParameters> parameters,
  std::shared_ptr<RTCInterface> & rtc_interface_left,
  std::shared_ptr<RTCInterface> & rtc_interface_right)
: SceneModuleInterface{name, node},
  parameters_{std::move(parameters)},
  rtc_interface_left_{rtc_interface_left},
  rtc_interface_right_{rtc_interface_right},
  uuid_left_{generateUUID()},
  uuid_right_{generateUUID()}
{
  using std::placeholders::_1;
  steering_factor_interface_ptr_ = std::make_unique<SteeringFactorInterface>(&node, "avoidance");
}
#endif

bool AvoidanceModule::isExecutionRequested() const
{
  DEBUG_PRINT("AVOIDANCE isExecutionRequested");

  if (current_state_ == ModuleStatus::RUNNING) {
    return true;
  }

  // Check ego is in preferred lane
  const auto current_lanes = util::getCurrentLanes(planner_data_);
  lanelet::ConstLanelet current_lane;
  lanelet::utils::query::getClosestLanelet(
    current_lanes, planner_data_->self_odometry->pose.pose, &current_lane);
  const auto num = planner_data_->route_handler->getNumLaneToPreferredLane(current_lane);

  if (num != 0) {
    return false;
  }

  // Check avoidance targets exist
  const auto avoid_data = calcAvoidancePlanningData(debug_data_);

  if (parameters_->publish_debug_marker) {
    setDebugData(avoid_data, path_shifter_, debug_data_);
  }

  return !avoid_data.target_objects.empty();
}

bool AvoidanceModule::isExecutionReady() const
{
  DEBUG_PRINT("AVOIDANCE isExecutionReady");

  {
    DebugData debug;
    static_cast<void>(calcAvoidancePlanningData(debug));
  }

  if (current_state_ == ModuleStatus::RUNNING) {
    return true;
  }

  return true;
}

ModuleStatus AvoidanceModule::updateState()
{
  const auto is_plan_running = isAvoidancePlanRunning();
  const bool has_avoidance_target = !avoidance_data_.target_objects.empty();

  if (!is_plan_running && !has_avoidance_target) {
    current_state_ = ModuleStatus::SUCCESS;
  } else if (
    !has_avoidance_target && parameters_->enable_update_path_when_object_is_gone &&
    !isAvoidanceManeuverRunning()) {
    // if dynamic objects are removed on path, change current state to reset path
    current_state_ = ModuleStatus::SUCCESS;
  } else {
    current_state_ = ModuleStatus::RUNNING;
  }

  DEBUG_PRINT(
    "is_plan_running = %d, has_avoidance_target = %d", is_plan_running, has_avoidance_target);

  return current_state_;
}

bool AvoidanceModule::isAvoidancePlanRunning() const
{
  constexpr double AVOIDING_SHIFT_THR = 0.1;
  const bool has_base_offset = std::abs(path_shifter_.getBaseOffset()) > AVOIDING_SHIFT_THR;
  const bool has_shift_point = (path_shifter_.getShiftLinesSize() > 0);
  return has_base_offset || has_shift_point;
}
bool AvoidanceModule::isAvoidanceManeuverRunning()
{
  const auto path_idx = avoidance_data_.ego_closest_path_index;

  for (const auto & al : registered_raw_shift_lines_) {
    if (path_idx > al.start_idx || is_avoidance_maneuver_starts) {
      is_avoidance_maneuver_starts = true;
      return true;
    }
  }
  return false;
}

AvoidancePlanningData AvoidanceModule::calcAvoidancePlanningData(DebugData & debug) const
{
  AvoidancePlanningData data;

  // reference pose
  const auto reference_pose = getUnshiftedEgoPose(prev_output_);
  data.reference_pose = reference_pose;

  // center line path (output of this function must have size > 1)
  const auto center_path = calcCenterLinePath(planner_data_, reference_pose);
  debug.center_line = center_path;
  if (center_path.points.size() < 2) {
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 5000, "calcCenterLinePath() must return path which size > 1");
    return data;
  }

  // reference path
#ifdef USE_OLD_ARCHITECTURE
  data.reference_path =
    util::resamplePathWithSpline(center_path, parameters_->resample_interval_for_planning);
#else
  const auto backward_extened_path = extendBackwardLength(*getPreviousModuleOutput().path);
  data.reference_path = util::resamplePathWithSpline(
    backward_extened_path, parameters_->resample_interval_for_planning);
#endif

  if (data.reference_path.points.size() < 2) {
    // if the resampled path has only 1 point, use original path.
    data.reference_path = center_path;
  }

  const size_t nearest_segment_index =
    findNearestSegmentIndex(data.reference_path.points, data.reference_pose.position);
  data.ego_closest_path_index =
    std::min(nearest_segment_index + 1, data.reference_path.points.size() - 1);

  // arclength from ego pose (used in many functions)
  data.arclength_from_ego = util::calcPathArcLengthArray(
    data.reference_path, 0, data.reference_path.points.size(),
    calcSignedArcLength(data.reference_path.points, getEgoPosition(), 0));

  // lanelet info
  data.current_lanelets = util::calcLaneAroundPose(
    planner_data_->route_handler, reference_pose, planner_data_->parameters.forward_path_length,
    planner_data_->parameters.backward_path_length);

  // keep avoidance state
  data.state = avoidance_data_.state;

  // target objects for avoidance
  fillAvoidanceTargetObjects(data, debug);

  DEBUG_PRINT("target object size = %lu", data.target_objects.size());

  return data;
}

void AvoidanceModule::fillAvoidanceTargetObjects(
  AvoidancePlanningData & data, DebugData & debug) const
{
  using boost::geometry::return_centroid;
  using boost::geometry::within;
  using lanelet::geometry::distance2d;
  using lanelet::geometry::toArcCoordinates;
  using lanelet::utils::getId;
  using lanelet::utils::to2D;

  const auto & path_points = data.reference_path.points;
  const auto & ego_pos = getEgoPosition();

  // detection area filter
  // when expanding lanelets, right_offset must be minus.
  // This is because y axis is positive on the left.
  const auto expanded_lanelets = getTargetLanelets(
    planner_data_, data.current_lanelets, parameters_->detection_area_left_expand_dist,
    parameters_->detection_area_right_expand_dist * (-1.0));

  const auto [object_within_target_lane, object_outside_target_lane] =
    util::separateObjectsByLanelets(*planner_data_->dynamic_object, expanded_lanelets);

  for (const auto & object : object_outside_target_lane.objects) {
    ObjectData other_object;
    other_object.object = object;
    other_object.reason = "OutOfTargetArea";
    data.other_objects.push_back(other_object);
  }

  DEBUG_PRINT("dynamic_objects size = %lu", planner_data_->dynamic_object->objects.size());
  DEBUG_PRINT("lane_filtered_objects size = %lu", object_within_target_lane.objects.size());

  // for goal
  const auto & rh = planner_data_->route_handler;
  const auto dist_to_goal =
    rh->isInGoalRouteSection(expanded_lanelets.back())
      ? calcSignedArcLength(path_points, ego_pos, rh->getGoalPose().position)
      : std::numeric_limits<double>::max();

  lanelet::ConstLineStrings3d debug_linestring;
  debug_linestring.clear();
  // for filtered objects
  ObjectDataArray target_objects;
  std::vector<AvoidanceDebugMsg> avoidance_debug_msg_array;
  for (const auto & object : object_within_target_lane.objects) {
    const auto & object_pose = object.kinematics.initial_pose_with_covariance.pose;
    AvoidanceDebugMsg avoidance_debug_msg;
    const auto avoidance_debug_array_false_and_push_back =
      [&avoidance_debug_msg, &avoidance_debug_msg_array](const std::string & failed_reason) {
        avoidance_debug_msg.allow_avoidance = false;
        avoidance_debug_msg.failed_reason = failed_reason;
        avoidance_debug_msg_array.push_back(avoidance_debug_msg);
      };

    ObjectData object_data;
    object_data.object = object;
    avoidance_debug_msg.object_id = getUuidStr(object_data);

    if (!isTargetObjectType(object)) {
      avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::OBJECT_IS_NOT_TYPE);
      object_data.reason = AvoidanceDebugFactor::OBJECT_IS_NOT_TYPE;
      data.other_objects.push_back(object_data);
      continue;
    }

    const auto object_closest_index = findNearestIndex(path_points, object_pose.position);
    const auto object_closest_pose = path_points.at(object_closest_index).point.pose;

    // Calc envelop polygon.
    fillObjectEnvelopePolygon(object_closest_pose, object_data);

    // calc object centroid.
    object_data.centroid = return_centroid<Point2d>(object_data.envelope_poly);

    // calc longitudinal distance from ego to closest target object footprint point.
    fillLongitudinalAndLengthByClosestEnvelopeFootprint(data.reference_path, ego_pos, object_data);
    avoidance_debug_msg.longitudinal_distance = object_data.longitudinal;

    // Calc moving time.
    fillObjectMovingTime(object_data);

    // Calc lateral deviation from path to target object.
    object_data.lateral = calcLateralDeviation(object_closest_pose, object_pose.position);
    avoidance_debug_msg.lateral_distance_from_centerline = object_data.lateral;

    // Find the footprint point closest to the path, set to object_data.overhang_distance.
    object_data.overhang_dist = calcEnvelopeOverhangDistance(
      object_data, object_closest_pose, object_data.overhang_pose.position);

    // Check whether the the ego should avoid the object.
    const auto & vehicle_width = planner_data_->parameters.vehicle_width;
    const auto safety_margin = 0.5 * vehicle_width + parameters_->lateral_passable_safety_buffer;
    object_data.avoid_required =
      (isOnRight(object_data) && std::abs(object_data.overhang_dist) < safety_margin) ||
      (!isOnRight(object_data) && object_data.overhang_dist < safety_margin);

    if (object_data.move_time > parameters_->threshold_time_object_is_moving) {
      avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::MOVING_OBJECT);
      object_data.reason = AvoidanceDebugFactor::MOVING_OBJECT;
      data.other_objects.push_back(object_data);
      continue;
    }

    // object is behind ego or too far.
    if (object_data.longitudinal < -parameters_->object_check_backward_distance) {
      avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::OBJECT_IS_BEHIND_THRESHOLD);
      object_data.reason = AvoidanceDebugFactor::OBJECT_IS_BEHIND_THRESHOLD;
      data.other_objects.push_back(object_data);
      continue;
    }
    if (object_data.longitudinal > parameters_->object_check_forward_distance) {
      avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::OBJECT_IS_IN_FRONT_THRESHOLD);
      object_data.reason = AvoidanceDebugFactor::OBJECT_IS_IN_FRONT_THRESHOLD;
      data.other_objects.push_back(object_data);
      continue;
    }

    // Target object is behind the path goal -> ignore.
    if (object_data.longitudinal > dist_to_goal) {
      avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::OBJECT_BEHIND_PATH_GOAL);
      object_data.reason = AvoidanceDebugFactor::OBJECT_BEHIND_PATH_GOAL;
      data.other_objects.push_back(object_data);
      continue;
    }

    if (
      object_data.longitudinal + object_data.length / 2 + parameters_->object_check_goal_distance >
      dist_to_goal) {
      avoidance_debug_array_false_and_push_back("TooNearToGoal");
      object_data.reason = "TooNearToGoal";
      data.other_objects.push_back(object_data);
      continue;
    }

    lanelet::ConstLanelet overhang_lanelet;
    if (!rh->getClosestLaneletWithinRoute(object_closest_pose, &overhang_lanelet)) {
      continue;
    }

    if (overhang_lanelet.id()) {
      object_data.overhang_lanelet = overhang_lanelet;
      lanelet::BasicPoint3d overhang_basic_pose(
        object_data.overhang_pose.position.x, object_data.overhang_pose.position.y,
        object_data.overhang_pose.position.z);
      const bool get_left =
        isOnRight(object_data) && parameters_->enable_avoidance_over_same_direction;
      const bool get_right =
        !isOnRight(object_data) && parameters_->enable_avoidance_over_same_direction;

      const auto target_lines = rh->getFurthestLinestring(
        overhang_lanelet, get_right, get_left,
        parameters_->enable_avoidance_over_opposite_direction);

      if (isOnRight(object_data)) {
        object_data.to_road_shoulder_distance =
          distance2d(to2D(overhang_basic_pose), to2D(target_lines.back().basicLineString()));
        debug_linestring.push_back(target_lines.back());
      } else {
        object_data.to_road_shoulder_distance =
          distance2d(to2D(overhang_basic_pose), to2D(target_lines.front().basicLineString()));
        debug_linestring.push_back(target_lines.front());
      }
    }

    // calculate avoid_margin dynamically
    // NOTE: This calculation must be after calculating to_road_shoulder_distance.
    const double max_avoid_margin = parameters_->lateral_collision_safety_buffer +
                                    parameters_->lateral_collision_margin + 0.5 * vehicle_width;
    const double min_safety_lateral_distance =
      parameters_->lateral_collision_safety_buffer + 0.5 * vehicle_width;
    const auto max_allowable_lateral_distance = object_data.to_road_shoulder_distance -
                                                parameters_->road_shoulder_safety_margin -
                                                0.5 * vehicle_width;

    const auto avoid_margin = [&]() -> boost::optional<double> {
      if (max_allowable_lateral_distance < min_safety_lateral_distance) {
        return boost::none;
      }
      return std::min(max_allowable_lateral_distance, max_avoid_margin);
    }();

    // force avoidance for stopped vehicle
    {
      const auto to_traffic_light =
        util::getDistanceToNextTrafficLight(object_pose, data.current_lanelets);

      object_data.to_stop_factor_distance =
        std::min(to_traffic_light, object_data.to_stop_factor_distance);
    }

    if (
      object_data.stop_time > parameters_->threshold_time_force_avoidance_for_stopped_vehicle &&
      parameters_->enable_force_avoidance_for_stopped_vehicle) {
      if (
        object_data.to_stop_factor_distance > parameters_->object_check_force_avoidance_clearance) {
        object_data.last_seen = clock_->now();
        object_data.avoid_margin = avoid_margin;
        data.target_objects.push_back(object_data);
        continue;
      }
    }

    DEBUG_PRINT(
      "set object_data: longitudinal = %f, lateral = %f, largest_overhang = %f,"
      "to_road_shoulder_distance = %f",
      object_data.longitudinal, object_data.lateral, object_data.overhang_dist,
      object_data.to_road_shoulder_distance);

    // Object is on center line -> ignore.
    avoidance_debug_msg.lateral_distance_from_centerline = object_data.lateral;
    if (std::abs(object_data.lateral) < parameters_->threshold_distance_object_is_on_center) {
      avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::TOO_NEAR_TO_CENTERLINE);
      object_data.reason = AvoidanceDebugFactor::TOO_NEAR_TO_CENTERLINE;
      data.other_objects.push_back(object_data);
      continue;
    }

    lanelet::ConstLanelet object_closest_lanelet;
    const auto lanelet_map = rh->getLaneletMapPtr();
    if (!lanelet::utils::query::getClosestLanelet(
          lanelet::utils::query::laneletLayer(lanelet_map), object_pose, &object_closest_lanelet)) {
      continue;
    }

    lanelet::BasicPoint2d object_centroid(object_data.centroid.x(), object_data.centroid.y());

    /**
     * Is not object in adjacent lane?
     *   - Yes -> Is parking object?
     *     - Yes -> the object is avoidance target.
     *     - No -> ignore this object.
     *   - No -> the object is avoidance target no matter whether it is parking object or not.
     */
    const auto is_in_ego_lane =
      within(object_centroid, overhang_lanelet.polygon2d().basicPolygon());
    if (is_in_ego_lane) {
      /**
       * TODO(Satoshi Ota) use intersection area
       * under the assumption that there is no parking vehicle inside intersection,
       * ignore all objects that is in the ego lane as not parking objects.
       */
      std::string turn_direction = overhang_lanelet.attributeOr("turn_direction", "else");
      if (turn_direction == "right" || turn_direction == "left" || turn_direction == "straight") {
        avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::NOT_PARKING_OBJECT);
        object_data.reason = AvoidanceDebugFactor::NOT_PARKING_OBJECT;
        data.other_objects.push_back(object_data);
        continue;
      }

      const auto centerline_pose =
        lanelet::utils::getClosestCenterPose(object_closest_lanelet, object_pose.position);
      lanelet::BasicPoint3d centerline_point(
        centerline_pose.position.x, centerline_pose.position.y, centerline_pose.position.z);

      // ============================================ <- most_left_lanelet.leftBound()
      // y              road shoulder
      // ^ ------------------------------------------
      // |   x                                +
      // +---> --- object closest lanelet --- o ----- <- object_closest_lanelet.centerline()
      //
      // --------------------------------------------
      // +: object position
      // o: nearest point on centerline

      bool is_left_side_parked_vehicle = false;
      {
        auto [object_shiftable_distance, sub_type] = [&]() {
          const auto most_left_road_lanelet = rh->getMostLeftLanelet(object_closest_lanelet);
          const auto most_left_lanelet_candidates =
            rh->getLaneletMapPtr()->laneletLayer.findUsages(most_left_road_lanelet.leftBound());

          lanelet::ConstLanelet most_left_lanelet = most_left_road_lanelet;
          const lanelet::Attribute sub_type =
            most_left_lanelet.attribute(lanelet::AttributeName::Subtype);

          for (const auto & ll : most_left_lanelet_candidates) {
            const lanelet::Attribute sub_type = ll.attribute(lanelet::AttributeName::Subtype);
            if (sub_type.value() == "road_shoulder") {
              most_left_lanelet = ll;
            }
          }

          const auto center_to_left_boundary = distance2d(
            to2D(most_left_lanelet.leftBound().basicLineString()), to2D(centerline_point));

          return std::make_pair(
            center_to_left_boundary - 0.5 * object.shape.dimensions.y, sub_type);
        }();

        if (sub_type.value() != "road_shoulder") {
          object_shiftable_distance += parameters_->object_check_min_road_shoulder_width;
        }

        const auto arc_coordinates = toArcCoordinates(
          to2D(object_closest_lanelet.centerline().basicLineString()), object_centroid);
        object_data.shiftable_ratio = arc_coordinates.distance / object_shiftable_distance;

        is_left_side_parked_vehicle =
          object_data.shiftable_ratio > parameters_->object_check_shiftable_ratio;
      }

      bool is_right_side_parked_vehicle = false;
      {
        auto [object_shiftable_distance, sub_type] = [&]() {
          const auto most_right_road_lanelet = rh->getMostRightLanelet(object_closest_lanelet);
          const auto most_right_lanelet_candidates =
            rh->getLaneletMapPtr()->laneletLayer.findUsages(most_right_road_lanelet.rightBound());

          lanelet::ConstLanelet most_right_lanelet = most_right_road_lanelet;
          const lanelet::Attribute sub_type =
            most_right_lanelet.attribute(lanelet::AttributeName::Subtype);

          for (const auto & ll : most_right_lanelet_candidates) {
            const lanelet::Attribute sub_type = ll.attribute(lanelet::AttributeName::Subtype);
            if (sub_type.value() == "road_shoulder") {
              most_right_lanelet = ll;
            }
          }

          const auto center_to_right_boundary = distance2d(
            to2D(most_right_lanelet.rightBound().basicLineString()), to2D(centerline_point));

          return std::make_pair(
            center_to_right_boundary - 0.5 * object.shape.dimensions.y, sub_type);
        }();

        if (sub_type.value() != "road_shoulder") {
          object_shiftable_distance += parameters_->object_check_min_road_shoulder_width;
        }

        const auto arc_coordinates = toArcCoordinates(
          to2D(object_closest_lanelet.centerline().basicLineString()), object_centroid);
        object_data.shiftable_ratio = -1.0 * arc_coordinates.distance / object_shiftable_distance;

        is_right_side_parked_vehicle =
          object_data.shiftable_ratio > parameters_->object_check_shiftable_ratio;
      }

      if (!is_left_side_parked_vehicle && !is_right_side_parked_vehicle) {
        avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::NOT_PARKING_OBJECT);
        object_data.reason = AvoidanceDebugFactor::NOT_PARKING_OBJECT;
        data.other_objects.push_back(object_data);
        continue;
      }
    }

    object_data.last_seen = clock_->now();
    object_data.avoid_margin = avoid_margin;

    // set data
    data.target_objects.push_back(object_data);
  }

  // debug
  {
    updateAvoidanceDebugData(avoidance_debug_msg_array);
    debug.farthest_linestring_from_overhang =
      std::make_shared<lanelet::ConstLineStrings3d>(debug_linestring);
    debug.current_lanelets = std::make_shared<lanelet::ConstLanelets>(data.current_lanelets);
    debug.expanded_lanelets = std::make_shared<lanelet::ConstLanelets>(expanded_lanelets);
  }
}

void AvoidanceModule::fillObjectEnvelopePolygon(
  const Pose & closest_pose, ObjectData & object_data) const
{
  using boost::geometry::within;

  const auto id = object_data.object.object_id;
  const auto same_id_obj = std::find_if(
    registered_objects_.begin(), registered_objects_.end(),
    [&id](const auto & o) { return o.object.object_id == id; });

  if (same_id_obj == registered_objects_.end()) {
    object_data.envelope_poly =
      createEnvelopePolygon(object_data, closest_pose, parameters_->object_envelope_buffer);
    return;
  }

  Polygon2d object_polygon{};
  util::calcObjectPolygon(object_data.object, &object_polygon);

  if (!within(object_polygon, same_id_obj->envelope_poly)) {
    object_data.envelope_poly =
      createEnvelopePolygon(object_data, closest_pose, parameters_->object_envelope_buffer);
    return;
  }

  object_data.envelope_poly = same_id_obj->envelope_poly;
}

void AvoidanceModule::fillObjectMovingTime(ObjectData & object_data) const
{
  const auto & object_vel =
    object_data.object.kinematics.initial_twist_with_covariance.twist.linear.x;
  const auto is_faster_than_threshold = object_vel > parameters_->threshold_speed_object_is_stopped;

  const auto id = object_data.object.object_id;
  const auto same_id_obj = std::find_if(
    stopped_objects_.begin(), stopped_objects_.end(),
    [&id](const auto & o) { return o.object.object_id == id; });

  const auto is_new_object = same_id_obj == stopped_objects_.end();

  if (!is_faster_than_threshold) {
    object_data.last_stop = clock_->now();
    object_data.move_time = 0.0;
    if (is_new_object) {
      object_data.stop_time = 0.0;
      object_data.last_move = clock_->now();
      stopped_objects_.push_back(object_data);
    } else {
      same_id_obj->stop_time = (clock_->now() - same_id_obj->last_move).seconds();
      same_id_obj->last_stop = clock_->now();
      same_id_obj->move_time = 0.0;
      object_data.stop_time = same_id_obj->stop_time;
    }
    return;
  }

  if (is_new_object) {
    object_data.move_time = std::numeric_limits<double>::max();
    object_data.stop_time = 0.0;
    object_data.last_move = clock_->now();
    return;
  }

  object_data.last_stop = same_id_obj->last_stop;
  object_data.move_time = (clock_->now() - same_id_obj->last_stop).seconds();
  object_data.stop_time = 0.0;

  if (object_data.move_time > parameters_->threshold_time_object_is_moving) {
    stopped_objects_.erase(same_id_obj);
  }
}

void AvoidanceModule::fillShiftLine(AvoidancePlanningData & data, DebugData & debug) const
{
  constexpr double AVOIDING_SHIFT_THR = 0.1;
  data.avoiding_now = std::abs(getCurrentShift()) > AVOIDING_SHIFT_THR;

  auto path_shifter = path_shifter_;

  /**
   * STEP 1
   * Create raw shift points from target object. The lateral margin between the ego and the target
   * object varies depending on the relative speed between the ego and the target object.
   */
  data.unapproved_raw_sl = calcRawShiftLinesFromObjects(data, debug);

  /**
   * STEP 2
   * Modify the raw shift points. (Merging, Trimming)
   */
  const auto processed_raw_sp = applyPreProcessToRawShiftLines(data.unapproved_raw_sl, debug);

  /**
   * STEP 3
   * Find new shift point
   */
  data.unapproved_new_sl = findNewShiftLine(processed_raw_sp, path_shifter);
  const auto found_new_sl = data.unapproved_new_sl.size() > 0;
  const auto registered = path_shifter.getShiftLines().size() > 0;
  data.found_avoidance_path = found_new_sl || registered;

  /**
   * STEP 4
   * If there are new shift points, these shift points are registered in path_shifter.
   */
  if (!data.unapproved_new_sl.empty()) {
    addNewShiftLines(path_shifter, data.unapproved_new_sl);
  }

  /**
   * STEP 5
   * Generate avoidance path.
   */
  auto candidate_path = generateAvoidancePath(path_shifter);

  /**
   * STEP 6
   * Check avoidance path safety. For each target objects and the objects in adjacent lanes,
   * check that there is a certain amount of margin in the lateral and longitudinal direction.
   */
  data.safe = isSafePath(path_shifter, candidate_path, debug);

  if (data.safe) {
    data.safe_new_sl = data.unapproved_new_sl;
    data.candidate_path = candidate_path;
  }

  /**
   * Find the nearest object that should be avoid. When the ego follows reference path,
   * if the lateral distance is smaller than minimum margin, the ego should avoid the object.
   */
  for (const auto & o : data.target_objects) {
    if (o.avoid_required) {
      data.avoid_required = true;
      data.stop_target_object = o;
    }
  }

  /**
   * If the avoidance path is not safe in situation where the ego should avoid object, the ego
   * stops in front of the front object with the necessary distance to avoid the object.
   */
  if (!data.safe && data.avoid_required) {
    data.yield_required = data.found_avoidance_path && data.avoid_required;
    data.candidate_path = toShiftedPath(data.reference_path);
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 5000, "not found safe avoidance path. transit yield maneuver...");
  }

  /**
   * Even if data.avoid_required is false, the module cancels registered shift point when the
   * approved avoidance path is not safe.
   */
  if (!data.safe && registered) {
    data.yield_required = true;
    data.candidate_path = toShiftedPath(data.reference_path);
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 5000,
      "found safe avoidance path, but it is not safe. canceling avoidance path...");
  }

  /**
   * TODO(Satoshi OTA) Think yield maneuver in the middle of avoidance.
   * Even if it is determined that a yield is necessary, the yield maneuver is not executed
   * if the avoidance has already been initiated.
   */
  if (!data.safe && data.avoiding_now) {
    data.yield_required = false;
    data.safe = true;  // OVERWRITE SAFETY JUDGE
    data.safe_new_sl = data.unapproved_new_sl;
    RCLCPP_WARN_THROTTLE(
      getLogger(), *clock_, 5000, "avoiding now. could not transit yield maneuver!!!");
  }

  fillDebugData(data, debug);
}

void AvoidanceModule::fillDebugData(const AvoidancePlanningData & data, DebugData & debug) const
{
  debug.output_shift = data.candidate_path.shift_length;
  debug.current_raw_shift = data.unapproved_raw_sl;
  debug.new_shift_lines = data.unapproved_new_sl;

  if (!data.stop_target_object) {
    return;
  }

  if (data.avoiding_now) {
    return;
  }

  if (data.unapproved_new_sl.empty()) {
    return;
  }

  const auto o_front = data.stop_target_object.get();
  const auto & base_link2front = planner_data_->parameters.base_link2front;
  const auto & vehicle_width = planner_data_->parameters.vehicle_width;

  const auto max_avoid_margin = parameters_->lateral_collision_safety_buffer +
                                parameters_->lateral_collision_margin + 0.5 * vehicle_width;

  const auto variable =
    getSharpAvoidanceDistance(getShiftLength(o_front, isOnRight(o_front), max_avoid_margin));
  const auto constant = getNominalPrepareDistance() +
                        parameters_->longitudinal_collision_safety_buffer + base_link2front;
  const auto total_avoid_distance = variable + constant;

  const auto opt_feasible_bound = calcLongitudinalOffsetPose(
    data.reference_path.points, getEgoPosition(), o_front.longitudinal - total_avoid_distance);

  if (opt_feasible_bound) {
    debug.feasible_bound = opt_feasible_bound.get();
  } else {
    debug.feasible_bound = getPose(data.reference_path.points.front());
  }
}

AvoidanceState AvoidanceModule::updateEgoState(const AvoidancePlanningData & data) const
{
  if (data.yield_required && parameters_->enable_yield_maneuver) {
    return AvoidanceState::YIELD;
  }

  if (!data.avoid_required) {
    return AvoidanceState::NOT_AVOID;
  }

  if (!data.found_avoidance_path) {
    return AvoidanceState::AVOID_PATH_NOT_READY;
  }

  if (isWaitingApproval() && path_shifter_.getShiftLines().empty()) {
    return AvoidanceState::AVOID_PATH_READY;
  }

  return AvoidanceState::AVOID_EXECUTE;
}

void AvoidanceModule::updateEgoBehavior(const AvoidancePlanningData & data, ShiftedPath & path)
{
  switch (data.state) {
    case AvoidanceState::NOT_AVOID: {
      break;
    }
    case AvoidanceState::YIELD: {
      insertYieldVelocity(path);
      insertWaitPoint(parameters_->use_constraints_for_decel, path);
      removeAllRegisteredShiftPoints(path_shifter_);
      clearWaitingApproval();
      unlockNewModuleLaunch();
      removeRTCStatus();
      break;
    }
    case AvoidanceState::AVOID_PATH_NOT_READY: {
      insertPrepareVelocity(false, path);
      insertWaitPoint(parameters_->use_constraints_for_decel, path);
      break;
    }
    case AvoidanceState::AVOID_PATH_READY: {
      insertPrepareVelocity(true, path);
      insertWaitPoint(parameters_->use_constraints_for_decel, path);
      break;
    }
    case AvoidanceState::AVOID_EXECUTE: {
      break;
    }
    default:
      throw std::domain_error("invalid behavior");
  }
}

/**
 * updateRegisteredRawShiftLines
 *
 *  - update path index of the registered objects
 *  - remove old objects whose end point is behind ego pose.
 */
void AvoidanceModule::updateRegisteredRawShiftLines()
{
  fillAdditionalInfoFromPoint(registered_raw_shift_lines_);

  AvoidLineArray avoid_lines;
  const int margin = 0;
  const auto deadline = static_cast<size_t>(
    std::max(static_cast<int>(avoidance_data_.ego_closest_path_index) - margin, 0));

  for (const auto & al : registered_raw_shift_lines_) {
    if (al.end_idx > deadline) {
      avoid_lines.push_back(al);
    }
  }

  DEBUG_PRINT(
    "ego_closest_path_index = %lu, registered_raw_shift_lines_ size: %lu -> %lu",
    avoidance_data_.ego_closest_path_index, registered_raw_shift_lines_.size(), avoid_lines.size());

  printShiftLines(registered_raw_shift_lines_, "registered_raw_shift_lines_ (before)");
  printShiftLines(avoid_lines, "registered_raw_shift_lines_ (after)");

  registered_raw_shift_lines_ = avoid_lines;
  debug_data_.registered_raw_shift = registered_raw_shift_lines_;
}

AvoidLineArray AvoidanceModule::applyPreProcessToRawShiftLines(
  AvoidLineArray & current_raw_shift_lines, DebugData & debug) const
{
  /**
   * Use all registered points. For the current points, if the similar one of the current
   * points are already registered, will not use it.
   * TODO(Horibe): enrich this logic to be able to consider the removal of the registered
   *               shift, because it cannot handle the case like "we don't have to avoid
   *               the object anymore".
   */
  auto total_raw_shift_lines =
    combineRawShiftLinesWithUniqueCheck(registered_raw_shift_lines_, current_raw_shift_lines);

  printShiftLines(current_raw_shift_lines, "current_raw_shift_lines");
  printShiftLines(registered_raw_shift_lines_, "registered_raw_shift_lines");
  printShiftLines(total_raw_shift_lines, "total_raw_shift_lines");

  /*
   * Add return-to-center shift point from the last shift point, if needed.
   * If there is no shift points, set return-to center shift from ego.
   */
  // TODO(Horibe) Here, the return point is calculated considering the prepare distance,
  // but there is an issue that sometimes this prepare distance is erased by the trimSimilarGrad,
  // and it suddenly tries to return from ego. Then steer rotates aggressively.
  // It is temporally solved by changing the threshold of trimSimilarGrad, but it needs to be
  // fixed in a proper way.
  // Maybe after merge, all shift points before the prepare distance can be deleted.
  addReturnShiftLineFromEgo(total_raw_shift_lines, current_raw_shift_lines);
  printShiftLines(total_raw_shift_lines, "total_raw_shift_lines_with_extra_return_shift");

  /**
   * On each path point, compute shift length with considering the raw shift points.
   * Then create a merged shift points by finding the change point of the gradient of shifting.
   *  - take maximum shift length if there is duplicate shift point
   *  - take sum if there are shifts for opposite direction (right and left)
   *  - shift length is interpolated linearly.
   * Note: Because this function just foolishly extracts points, it includes
   *       insignificant small (useless) shift points, which should be removed in post-process.
   */
  auto merged_shift_lines = mergeShiftLines(total_raw_shift_lines, debug);
  debug.merged = merged_shift_lines;

  /*
   * Remove unnecessary shift points
   *  - Quantize the shift length to reduce the shift point noise
   *  - Change the shift length to the previous one if the deviation is small.
   *  - Combine shift points that have almost same gradient
   *  - Remove unnecessary return shift (back to the center line).
   */
  auto shift_lines = trimShiftLine(merged_shift_lines, debug);
  DEBUG_PRINT("final shift point size = %lu", shift_lines.size());

  return shift_lines;
}

void AvoidanceModule::registerRawShiftLines(const AvoidLineArray & future)
{
  if (future.empty()) {
    RCLCPP_ERROR(getLogger(), "future is empty! return.");
    return;
  }

  const auto old_size = registered_raw_shift_lines_.size();

  const auto future_with_info = fillAdditionalInfo(future);
  printShiftLines(future_with_info, "future_with_info");
  printShiftLines(registered_raw_shift_lines_, "registered_raw_shift_lines_");
  printShiftLines(current_raw_shift_lines_, "current_raw_shift_lines_");

  const auto isAlreadyRegistered = [this](const auto id) {
    const auto & r = registered_raw_shift_lines_;
    return std::any_of(r.begin(), r.end(), [id](const auto & r_sl) { return r_sl.id == id; });
  };

  const auto getAvoidLineByID = [this](const auto id) {
    for (const auto & sl : current_raw_shift_lines_) {
      if (sl.id == id) {
        return sl;
      }
    }
    return AvoidLine{};
  };

  for (const auto & al : future_with_info) {
    if (al.parent_ids.empty()) {
      RCLCPP_ERROR(getLogger(), "avoid line for path_shifter must have parent_id.");
    }
    for (const auto parent_id : al.parent_ids) {
      if (!isAlreadyRegistered(parent_id)) {
        registered_raw_shift_lines_.push_back(getAvoidLineByID(parent_id));
      }
    }
  }

  DEBUG_PRINT("registered object size: %lu -> %lu", old_size, registered_raw_shift_lines_.size());
}

double AvoidanceModule::getShiftLength(
  const ObjectData & object, const bool & is_object_on_right, const double & avoid_margin) const
{
  const auto shift_length =
    behavior_path_planner::calcShiftLength(is_object_on_right, object.overhang_dist, avoid_margin);
  return is_object_on_right ? std::min(shift_length, getLeftShiftBound())
                            : std::max(shift_length, getRightShiftBound());
}

/**
 * calcRawShiftLinesFromObjects
 *
 * Calculate the shift points (start/end point, shift length) from the object lateral
 * and longitudinal positions in the Frenet coordinate. The jerk limit is also considered here.
 */
AvoidLineArray AvoidanceModule::calcRawShiftLinesFromObjects(
  AvoidancePlanningData & data, DebugData & debug) const
{
  {
    debug_avoidance_initializer_for_shift_line_.clear();
    debug.unavoidable_objects.clear();
  }

  const auto prepare_distance = getNominalPrepareDistance();

  // To be consistent with changes in the ego position, the current shift length is considered.
  const auto current_ego_shift = getCurrentShift();
  // // implement lane detection here.

  AvoidLineArray avoid_lines;
  std::vector<AvoidanceDebugMsg> avoidance_debug_msg_array;
  avoidance_debug_msg_array.reserve(data.target_objects.size());
  for (auto & o : data.target_objects) {
    AvoidanceDebugMsg avoidance_debug_msg;
    const auto avoidance_debug_array_false_and_push_back =
      [&avoidance_debug_msg, &avoidance_debug_msg_array](const std::string & failed_reason) {
        avoidance_debug_msg.allow_avoidance = false;
        avoidance_debug_msg.failed_reason = failed_reason;
        avoidance_debug_msg_array.push_back(avoidance_debug_msg);
      };

    avoidance_debug_msg.object_id = getUuidStr(o);
    avoidance_debug_msg.longitudinal_distance = o.longitudinal;
    avoidance_debug_msg.lateral_distance_from_centerline = o.lateral;
    avoidance_debug_msg.to_furthest_linestring_distance = o.to_road_shoulder_distance;
    // avoidance_debug_msg.max_shift_length = max_allowable_lateral_distance;

    if (!o.avoid_margin) {
      avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::INSUFFICIENT_LATERAL_MARGIN);
      o.reason = AvoidanceDebugFactor::INSUFFICIENT_LATERAL_MARGIN;
      debug.unavoidable_objects.push_back(o);
      continue;
    }

    const auto is_object_on_right = isOnRight(o);
    const auto shift_length = getShiftLength(o, is_object_on_right, o.avoid_margin.get());
    if (isSameDirectionShift(is_object_on_right, shift_length)) {
      avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::SAME_DIRECTION_SHIFT);
      o.reason = AvoidanceDebugFactor::SAME_DIRECTION_SHIFT;
      debug.unavoidable_objects.push_back(o);
      continue;
    }

    const auto avoiding_shift = shift_length - current_ego_shift;
    const auto return_shift = shift_length;

    // use absolute dist for return-to-center, relative dist from current for avoiding.
    const auto nominal_avoid_distance = getNominalAvoidanceDistance(avoiding_shift);
    const auto nominal_return_distance = getNominalAvoidanceDistance(return_shift);

    /**
     * Is there enough distance from ego to object for avoidance?
     *   - Yes -> use the nominal distance.
     *   - No -> check if it is possible to avoid within maximum jerk limit.
     *     - Yes -> use the stronger jerk.
     *     - No -> ignore this object. Expected behavior is that the vehicle will stop in front
     *             of the obstacle, then start avoidance.
     */
    const bool has_enough_distance = o.longitudinal > (prepare_distance + nominal_avoid_distance);
    const auto remaining_distance = o.longitudinal - prepare_distance;
    if (!has_enough_distance) {
      if (remaining_distance <= 0.0) {
        // TODO(Horibe) Even if there is no enough distance for avoidance shift, the
        // return-to-center shift must be considered for each object if the current_shift
        // is not zero.
        avoidance_debug_array_false_and_push_back(
          AvoidanceDebugFactor::REMAINING_DISTANCE_LESS_THAN_ZERO);
        if (!data.avoiding_now) {
          o.reason = AvoidanceDebugFactor::REMAINING_DISTANCE_LESS_THAN_ZERO;
          debug.unavoidable_objects.push_back(o);
        }
        continue;
      }

      // This is the case of exceeding the jerk limit. Use the sharp avoidance ego speed.
      const auto required_jerk = path_shifter_.calcJerkFromLatLonDistance(
        avoiding_shift, remaining_distance, getSharpAvoidanceEgoSpeed());
      avoidance_debug_msg.required_jerk = required_jerk;
      avoidance_debug_msg.maximum_jerk = parameters_->max_lateral_jerk;
      if (required_jerk > parameters_->max_lateral_jerk) {
        avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::TOO_LARGE_JERK);
        if (!data.avoiding_now) {
          o.reason = AvoidanceDebugFactor::TOO_LARGE_JERK;
          debug.unavoidable_objects.push_back(o);
        }
        continue;
      }
    }
    const auto avoiding_distance =
      has_enough_distance ? nominal_avoid_distance : remaining_distance;

    DEBUG_PRINT(
      "nominal_lateral_jerk = %f, getNominalAvoidanceEgoSpeed() = %f, prepare_distance = %f, "
      "has_enough_distance = %d",
      parameters_->nominal_lateral_jerk, getNominalAvoidanceEgoSpeed(), prepare_distance,
      has_enough_distance);

    AvoidLine al_avoid;
    al_avoid.end_shift_length = shift_length;
    al_avoid.start_shift_length = current_ego_shift;
    al_avoid.end_longitudinal = o.longitudinal;
    al_avoid.start_longitudinal = o.longitudinal - avoiding_distance;
    al_avoid.id = getOriginalShiftLineUniqueId();
    al_avoid.object = o;
    avoid_lines.push_back(al_avoid);

    // The end_margin also has the purpose of preventing the return path from NOT being
    // triggered at the end point.
    const auto end_margin = 1.0;
    const auto return_remaining_distance =
      std::max(avoidance_data_.arclength_from_ego.back() - o.longitudinal - end_margin, 0.0);

    AvoidLine al_return;
    al_return.end_shift_length = 0.0;
    al_return.start_shift_length = shift_length;
    al_return.start_longitudinal = o.longitudinal + o.length;
    al_return.end_longitudinal =
      o.longitudinal + o.length + std::min(nominal_return_distance, return_remaining_distance);
    al_return.id = getOriginalShiftLineUniqueId();
    al_return.object = o;
    avoid_lines.push_back(al_return);

    DEBUG_PRINT(
      "object is set: avoid_shift = %f, return_shift = %f, dist = (avoidStart: %3.3f, avoidEnd: "
      "%3.3f, returnEnd: %3.3f), avoiding_dist = (nom:%f, res:%f), avoid_margin = %f, return_dist "
      "= %f",
      avoiding_shift, return_shift, al_avoid.start_longitudinal, al_avoid.end_longitudinal,
      al_return.end_longitudinal, nominal_avoid_distance, avoiding_distance, o.avoid_margin.get(),
      nominal_return_distance);
    avoidance_debug_msg.allow_avoidance = true;
    avoidance_debug_msg_array.push_back(avoidance_debug_msg);

    o.is_avoidable = true;
  }

  debug_avoidance_initializer_for_shift_line_ = std::move(avoidance_debug_msg_array);
  debug_avoidance_initializer_for_shift_line_time_ = clock_->now();
  fillAdditionalInfoFromLongitudinal(avoid_lines);

  return avoid_lines;
}

AvoidLineArray AvoidanceModule::fillAdditionalInfo(const AvoidLineArray & shift_lines) const
{
  if (shift_lines.empty()) {
    return shift_lines;
  }

  auto out_points = shift_lines;

  const auto & path = avoidance_data_.reference_path;
  const auto arclength = avoidance_data_.arclength_from_ego;

  // calc longitudinal
  for (auto & sl : out_points) {
    sl.start_idx = findNearestIndex(path.points, sl.start.position);
    sl.start_longitudinal = arclength.at(sl.start_idx);
    sl.end_idx = findNearestIndex(path.points, sl.end.position);
    sl.end_longitudinal = arclength.at(sl.end_idx);
  }

  // sort by longitudinal
  std::sort(out_points.begin(), out_points.end(), [](auto a, auto b) {
    return a.end_longitudinal < b.end_longitudinal;
  });

  // calc relative lateral length
  out_points.front().start_shift_length = getCurrentBaseShift();
  for (size_t i = 1; i < shift_lines.size(); ++i) {
    out_points.at(i).start_shift_length = shift_lines.at(i - 1).end_shift_length;
  }

  return out_points;
}
AvoidLine AvoidanceModule::fillAdditionalInfo(const AvoidLine & shift_line) const
{
  const auto ret = fillAdditionalInfo(AvoidLineArray{shift_line});
  return ret.front();
}

AvoidLine AvoidanceModule::getNonStraightShiftLine(const AvoidLineArray & shift_lines) const
{
  for (const auto & sl : shift_lines) {
    if (fabs(getRelativeLengthFromPath(sl)) > 0.01) {
      return sl;
    }
  }

  return {};
}

void AvoidanceModule::fillAdditionalInfoFromPoint(AvoidLineArray & shift_lines) const
{
  if (shift_lines.empty()) {
    return;
  }

  const auto & path = avoidance_data_.reference_path;
  const auto arclength = util::calcPathArcLengthArray(path);
  const auto dist_path_front_to_ego =
    calcSignedArcLength(path.points, 0, avoidance_data_.ego_closest_path_index);

  // calc longitudinal
  for (auto & sl : shift_lines) {
    sl.start_idx = findNearestIndex(path.points, sl.start.position);
    sl.start_longitudinal = arclength.at(sl.start_idx) - dist_path_front_to_ego;
    sl.end_idx = findNearestIndex(path.points, sl.end.position);
    sl.end_longitudinal = arclength.at(sl.end_idx) - dist_path_front_to_ego;
  }
}

void AvoidanceModule::fillAdditionalInfoFromLongitudinal(AvoidLineArray & shift_lines) const
{
  const auto & path = avoidance_data_.reference_path;
  const auto arclength = util::calcPathArcLengthArray(path);
  const auto path_front_to_ego =
    calcSignedArcLength(path.points, 0, avoidance_data_.ego_closest_path_index);

  for (auto & sl : shift_lines) {
    sl.start_idx = findPathIndexFromArclength(arclength, sl.start_longitudinal + path_front_to_ego);
    sl.start = path.points.at(sl.start_idx).point.pose;
    sl.end_idx = findPathIndexFromArclength(arclength, sl.end_longitudinal + path_front_to_ego);
    sl.end = path.points.at(sl.end_idx).point.pose;
  }
}
/*
 * combineRawShiftLinesWithUniqueCheck
 *
 * Combine points A into B. If shift_line of A which has same object_id and
 * similar shape is already in B, it will not be added into B.
 */
AvoidLineArray AvoidanceModule::combineRawShiftLinesWithUniqueCheck(
  const AvoidLineArray & base_lines, const AvoidLineArray & added_lines) const
{
  // TODO(Horibe) parametrize
  const auto isSimilar = [](const AvoidLine & a, const AvoidLine & b) {
    using tier4_autoware_utils::calcDistance2d;
    if (calcDistance2d(a.start, b.start) > 1.0) {
      return false;
    }
    if (calcDistance2d(a.end, b.end) > 1.0) {
      return false;
    }
    if (std::abs(a.end_shift_length - b.end_shift_length) > 0.5) {
      return false;
    }
    return true;
  };
  const auto hasSameObjectId = [](const auto & a, const auto & b) {
    return a.object.object.object_id == b.object.object.object_id;
  };

  auto combined = base_lines;  // initialized
  for (const auto & added_line : added_lines) {
    bool skip = false;

    for (const auto & base_line : base_lines) {
      if (hasSameObjectId(added_line, base_line) && isSimilar(added_line, base_line)) {
        skip = true;
        break;
      }
    }
    if (!skip) {
      combined.push_back(added_line);
    }
  }

  return combined;
}

void AvoidanceModule::generateTotalShiftLine(
  const AvoidLineArray & avoid_lines, ShiftLineData & shift_line_data) const
{
  const auto & path = avoidance_data_.reference_path;
  const auto & arcs = avoidance_data_.arclength_from_ego;
  const auto N = path.points.size();

  auto & sl = shift_line_data;

  sl.shift_line = std::vector<double>(N, 0.0);
  sl.shift_line_grad = std::vector<double>(N, 0.0);

  sl.pos_shift_line = std::vector<double>(N, 0.0);
  sl.neg_shift_line = std::vector<double>(N, 0.0);

  sl.pos_shift_line_grad = std::vector<double>(N, 0.0);
  sl.neg_shift_line_grad = std::vector<double>(N, 0.0);

  // debug
  sl.shift_line_history = std::vector<std::vector<double>>(avoid_lines.size(), sl.shift_line);

  // take minmax for same directional shift length
  for (size_t j = 0; j < avoid_lines.size(); ++j) {
    const auto & al = avoid_lines.at(j);
    for (size_t i = 0; i < N; ++i) {
      // calc current interpolated shift
      const auto i_shift = lerpShiftLengthOnArc(arcs.at(i), al);

      // update maximum shift for positive direction
      if (i_shift > sl.pos_shift_line.at(i)) {
        sl.pos_shift_line.at(i) = i_shift;
        sl.pos_shift_line_grad.at(i) = al.getGradient();
      }

      // update minumum shift for negative direction
      if (i_shift < sl.neg_shift_line.at(i)) {
        sl.neg_shift_line.at(i) = i_shift;
        sl.neg_shift_line_grad.at(i) = al.getGradient();
      }

      // store for debug print
      sl.shift_line_history.at(j).at(i) = i_shift;
    }
  }

  // Merge shift length of opposite directions.
  for (size_t i = 0; i < N; ++i) {
    sl.shift_line.at(i) = sl.pos_shift_line.at(i) + sl.neg_shift_line.at(i);
    sl.shift_line_grad.at(i) = sl.pos_shift_line_grad.at(i) + sl.neg_shift_line_grad.at(i);
  }

  // overwrite shift with current_ego_shift until ego pose.
  const auto current_shift = getCurrentLinearShift();
  for (size_t i = 0; i <= avoidance_data_.ego_closest_path_index; ++i) {
    sl.shift_line.at(i) = current_shift;
    sl.shift_line_grad.at(i) = 0.0;
  }

  // If the shift point does not have an associated object,
  // use previous value.
  for (size_t i = 1; i < N; ++i) {
    bool has_object = false;
    for (const auto & al : avoid_lines) {
      if (al.start_idx < i && i < al.end_idx) {
        has_object = true;
        break;
      }
    }
    if (!has_object) {
      sl.shift_line.at(i) = sl.shift_line.at(i - 1);
    }
  }
  sl.shift_line_history.push_back(sl.shift_line);
}

AvoidLineArray AvoidanceModule::extractShiftLinesFromLine(ShiftLineData & shift_line_data) const
{
  const auto & path = avoidance_data_.reference_path;
  const auto & arcs = avoidance_data_.arclength_from_ego;
  const auto N = path.points.size();

  auto & sl = shift_line_data;

  const auto getBwdGrad = [&](const size_t i) {
    if (i == 0) {
      return sl.shift_line_grad.at(i);
    }
    const double ds = arcs.at(i) - arcs.at(i - 1);
    if (ds < 1.0e-5) {
      return sl.shift_line_grad.at(i);
    }  // use theoretical value when ds is too small.
    return (sl.shift_line.at(i) - sl.shift_line.at(i - 1)) / ds;
  };

  const auto getFwdGrad = [&](const size_t i) {
    if (i == arcs.size() - 1) {
      return sl.shift_line_grad.at(i);
    }
    const double ds = arcs.at(i + 1) - arcs.at(i);
    if (ds < 1.0e-5) {
      return sl.shift_line_grad.at(i);
    }  // use theoretical value when ds is too small.
    return (sl.shift_line.at(i + 1) - sl.shift_line.at(i)) / ds;
  };

  // calculate forward and backward gradient of the shift length.
  // This will be used for grad-change-point check.
  sl.forward_grad = std::vector<double>(N, 0.0);
  sl.backward_grad = std::vector<double>(N, 0.0);
  for (size_t i = 0; i < N - 1; ++i) {
    sl.forward_grad.at(i) = getFwdGrad(i);
    sl.backward_grad.at(i) = getBwdGrad(i);
  }

  AvoidLineArray merged_avoid_lines;
  AvoidLine al{};
  bool found_first_start = false;
  constexpr auto CREATE_SHIFT_GRAD_THR = 0.001;
  constexpr auto IS_ALREADY_SHIFTING_THR = 0.001;
  for (size_t i = avoidance_data_.ego_closest_path_index; i < N - 1; ++i) {
    const auto & p = path.points.at(i).point.pose;
    const auto shift = sl.shift_line.at(i);

    // If the vehicle is already on the avoidance (checked by the first point has shift),
    // set a start point at the first path point.
    if (!found_first_start && std::abs(shift) > IS_ALREADY_SHIFTING_THR) {
      setStartData(al, 0.0, p, i, arcs.at(i));  // start length is overwritten later.
      found_first_start = true;
      DEBUG_PRINT("shift (= %f) is not zero at i = %lu. set start shift here.", shift, i);
    }

    // find the point where the gradient of the shift is changed
    const bool set_shift_line_flag =
      std::abs(sl.forward_grad.at(i) - sl.backward_grad.at(i)) > CREATE_SHIFT_GRAD_THR;

    if (!set_shift_line_flag) {
      continue;
    }

    if (!found_first_start) {
      setStartData(al, 0.0, p, i, arcs.at(i));  // start length is overwritten later.
      found_first_start = true;
      DEBUG_PRINT("grad change detected. start at i = %lu", i);
    } else {
      setEndData(al, shift, p, i, arcs.at(i));
      al.id = getOriginalShiftLineUniqueId();
      merged_avoid_lines.push_back(al);
      setStartData(al, 0.0, p, i, arcs.at(i));  // start length is overwritten later.
      DEBUG_PRINT("end and start point found at i = %lu", i);
    }
  }
  return merged_avoid_lines;
}

AvoidLineArray AvoidanceModule::mergeShiftLines(
  const AvoidLineArray & raw_shift_lines, DebugData & debug) const
{
  // Generate shift line by merging raw_shift_lines.
  ShiftLineData shift_line_data;
  generateTotalShiftLine(raw_shift_lines, shift_line_data);

  // Re-generate shift points by detecting gradient-change point of the shift line.
  auto merged_shift_lines = extractShiftLinesFromLine(shift_line_data);

  // set parent id
  for (auto & al : merged_shift_lines) {
    al.parent_ids = calcParentIds(raw_shift_lines, al);
  }

  // sort by distance from ego.
  alignShiftLinesOrder(merged_shift_lines);

  // debug visualize
  {
    debug.pos_shift = shift_line_data.pos_shift_line;
    debug.neg_shift = shift_line_data.neg_shift_line;
    debug.total_shift = shift_line_data.shift_line;
  }

  // debug print
  {
    const auto & arc = avoidance_data_.arclength_from_ego;
    const auto & closest = avoidance_data_.ego_closest_path_index;
    const auto & sl = shift_line_data.shift_line;
    const auto & sg = shift_line_data.shift_line_grad;
    const auto & fg = shift_line_data.forward_grad;
    const auto & bg = shift_line_data.backward_grad;
    using std::setw;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "\n[idx, arc, shift (for each shift points, filtered | total), grad (ideal, bwd, fwd)]: "
          "closest = "
       << closest << ", raw_shift_lines size = " << raw_shift_lines.size() << std::endl;
    for (size_t i = 0; i < arc.size(); ++i) {
      ss << "i = " << i << " | arc: " << arc.at(i) << " | shift: (";
      for (const auto & p : shift_line_data.shift_line_history) {
        ss << setw(5) << p.at(i) << ", ";
      }
      ss << "| total: " << setw(5) << sl.at(i) << ") | grad: (" << sg.at(i) << ", " << fg.at(i)
         << ", " << bg.at(i) << ")" << std::endl;
    }
    DEBUG_PRINT("%s", ss.str().c_str());
  }

  printShiftLines(merged_shift_lines, "merged_shift_lines");

  return merged_shift_lines;
}

std::vector<size_t> AvoidanceModule::calcParentIds(
  const AvoidLineArray & parent_candidates, const AvoidLine & child) const
{
  // Get the ID of the original AP whose transition area overlaps with the given AP,
  // and set it to the parent id.
  std::set<uint64_t> ids;
  for (const auto & al : parent_candidates) {
    const auto p_s = al.start_longitudinal;
    const auto p_e = al.end_longitudinal;
    const auto has_overlap = !(p_e < child.start_longitudinal || child.end_longitudinal < p_s);

    if (!has_overlap) {
      continue;
    }

    // Id the shift is overlapped, insert the shift point. Additionally, the shift which refers
    // to the same object id (created by the same object) will be set.
    //
    // Why? : think that there are two shifts, avoiding and .
    // If you register only the avoiding shift, the return-to-center shift will not be generated
    // when you get too close to or over the obstacle. The return-shift can be handled with
    // addReturnShift(), but it maybe reasonable to register the return-to-center shift for the
    // object at the same time as registering the avoidance shift to remove the complexity of the
    // addReturnShift().
    for (const auto & al_local : parent_candidates) {
      if (al_local.object.object.object_id == al.object.object.object_id) {
        ids.insert(al_local.id);
      }
    }
  }
  return std::vector<size_t>(ids.begin(), ids.end());
}

/*
 * Remove unnecessary avoid points
 * - Combine avoid points that have almost same gradient
 * - Quantize the shift length to reduce the shift point noise
 * - Change the shift length to the previous one if the deviation is small.
 * - Remove unnecessary return shift (back to the center line).
 */
AvoidLineArray AvoidanceModule::trimShiftLine(
  const AvoidLineArray & shift_lines, DebugData & debug) const
{
  if (shift_lines.empty()) {
    return shift_lines;
  }

  AvoidLineArray sl_array_trimmed = shift_lines;

  // sort shift points from front to back.
  alignShiftLinesOrder(sl_array_trimmed);

  // - Combine avoid points that have almost same gradient.
  // this is to remove the noise.
  {
    const auto CHANGE_SHIFT_THRESHOLD_FOR_NOISE = 0.1;
    trimSimilarGradShiftLine(sl_array_trimmed, CHANGE_SHIFT_THRESHOLD_FOR_NOISE);
    debug.trim_similar_grad_shift = sl_array_trimmed;
    printShiftLines(sl_array_trimmed, "after trim_similar_grad_shift");
  }

  // - Quantize the shift length to reduce the shift point noise
  // This is to remove the noise coming from detection accuracy, interpolation, resampling, etc.
  {
    constexpr double QUANTIZATION_DISTANCE = 0.2;
    quantizeShiftLine(sl_array_trimmed, QUANTIZATION_DISTANCE);
    printShiftLines(sl_array_trimmed, "after sl_array_trimmed");
    debug.quantized = sl_array_trimmed;
  }

  // - Change the shift length to the previous one if the deviation is small.
  {
    // constexpr double SHIFT_DIFF_THRES = 0.5;
    // trimSmallShiftLine(sl_array_trimmed, SHIFT_DIFF_THRES);
    debug.trim_small_shift = sl_array_trimmed;
    printShiftLines(sl_array_trimmed, "after trim_small_shift");
  }

  // - Combine avoid points that have almost same gradient (again)
  {
    const auto CHANGE_SHIFT_THRESHOLD = 0.2;
    trimSimilarGradShiftLine(sl_array_trimmed, CHANGE_SHIFT_THRESHOLD);
    debug.trim_similar_grad_shift_second = sl_array_trimmed;
    printShiftLines(sl_array_trimmed, "after trim_similar_grad_shift_second");
  }

  // - trimTooSharpShift
  // Check if it is not too sharp for the return-to-center shift point.
  // If the shift is sharp, it is combined with the next shift point until it gets non-sharp.
  {
    trimSharpReturn(sl_array_trimmed);
    debug.trim_too_sharp_shift = sl_array_trimmed;
    printShiftLines(sl_array_trimmed, "after trimSharpReturn");
  }

  return sl_array_trimmed;
}

void AvoidanceModule::alignShiftLinesOrder(
  AvoidLineArray & shift_lines, const bool recalculate_start_length) const
{
  if (shift_lines.empty()) {
    return;
  }

  // sort shift points from front to back.
  std::sort(shift_lines.begin(), shift_lines.end(), [](auto a, auto b) {
    return a.end_longitudinal < b.end_longitudinal;
  });

  // calc relative length
  // NOTE: the input shift point must not have conflict range. Otherwise relative
  // length value will be broken.
  if (recalculate_start_length) {
    shift_lines.front().start_shift_length = getCurrentLinearShift();
    for (size_t i = 1; i < shift_lines.size(); ++i) {
      shift_lines.at(i).start_shift_length = shift_lines.at(i - 1).end_shift_length;
    }
  }
}

void AvoidanceModule::quantizeShiftLine(AvoidLineArray & shift_lines, const double interval) const
{
  if (interval < 1.0e-5) {
    return;  // no need to process
  }

  for (auto & sl : shift_lines) {
    sl.end_shift_length = std::round(sl.end_shift_length / interval) * interval;
  }

  alignShiftLinesOrder(shift_lines);
}

void AvoidanceModule::trimSmallShiftLine(
  AvoidLineArray & shift_lines, const double shift_diff_thres) const
{
  AvoidLineArray shift_lines_orig = shift_lines;
  shift_lines.clear();

  shift_lines.push_back(shift_lines_orig.front());  // Take the first one anyway (think later)

  for (size_t i = 1; i < shift_lines_orig.size(); ++i) {
    auto sl_now = shift_lines_orig.at(i);
    const auto sl_prev = shift_lines.back();
    const auto shift_diff = sl_now.end_shift_length - sl_prev.end_shift_length;

    auto sl_modified = sl_now;

    // remove the shift point if the length is almost same as the previous one.
    if (std::abs(shift_diff) < shift_diff_thres) {
      sl_modified.end_shift_length = sl_prev.end_shift_length;
      sl_modified.start_shift_length = sl_prev.end_shift_length;
      DEBUG_PRINT(
        "i = %lu, relative shift = %f is small. set with relative shift = 0.", i, shift_diff);
    } else {
      DEBUG_PRINT("i = %lu, shift = %f is large. take this one normally.", i, shift_diff);
    }

    shift_lines.push_back(sl_modified);
  }

  alignShiftLinesOrder(shift_lines);

  DEBUG_PRINT("size %lu -> %lu", shift_lines_orig.size(), shift_lines.size());
}

void AvoidanceModule::trimSimilarGradShiftLine(
  AvoidLineArray & avoid_lines, const double change_shift_dist_threshold) const
{
  AvoidLineArray avoid_lines_orig = avoid_lines;
  avoid_lines.clear();

  avoid_lines.push_back(avoid_lines_orig.front());  // Take the first one anyway (think later)

  // Save the points being merged. When merging consecutively, also check previously merged points.
  AvoidLineArray being_merged_points;

  for (size_t i = 1; i < avoid_lines_orig.size(); ++i) {
    const auto al_now = avoid_lines_orig.at(i);
    const auto al_prev = avoid_lines.back();

    being_merged_points.push_back(al_prev);  // This point is about to be merged.

    auto combined_al = al_prev;
    setEndData(
      combined_al, al_now.end_shift_length, al_now.end, al_now.end_idx, al_now.end_longitudinal);
    combined_al.parent_ids = concatParentIds(combined_al.parent_ids, al_prev.parent_ids);

    const auto has_large_length_change = [&]() {
      for (const auto & original : being_merged_points) {
        const auto longitudinal = original.end_longitudinal - combined_al.start_longitudinal;
        const auto new_length =
          combined_al.getGradient() * longitudinal + combined_al.start_shift_length;
        const bool has_large_change =
          std::abs(new_length - original.end_shift_length) > change_shift_dist_threshold;

        DEBUG_PRINT(
          "original.end_shift_length: %f, original.end_longitudinal: %f, "
          "combined_al.start_longitudinal: "
          "%f, combined_al.Gradient: %f, new_length: %f, has_large_change: %d",
          original.end_shift_length, original.end_longitudinal, combined_al.start_longitudinal,
          combined_al.getGradient(), new_length, has_large_change);

        if (std::abs(new_length - original.end_shift_length) > change_shift_dist_threshold) {
          return true;
        }
      }
      return false;
    }();

    if (has_large_length_change) {
      // If this point is merged with the previous points, it makes a large changes.
      // Do not merge this.
      avoid_lines.push_back(al_now);
      being_merged_points.clear();
      DEBUG_PRINT("use this point. has_large_length_change = %d", has_large_length_change);
    } else {
      avoid_lines.back() = combined_al;  // Update the last points by merging the current point
      being_merged_points.push_back(al_prev);
      DEBUG_PRINT("trim! has_large_length_change = %d", has_large_length_change);
    }
  }

  alignShiftLinesOrder(avoid_lines);

  DEBUG_PRINT("size %lu -> %lu", avoid_lines_orig.size(), avoid_lines.size());
}

/**
 * Remove short "return to center" shift point. ¯¯\_/¯¯　-> ¯¯¯¯¯¯
 *
 * Is the shift point for "return to center"?
 *  - no : Do not trim anything.
 *  - yes: Is it short distance enough to be removed?
 *     - no : Do not trim anything.
 *     - yes: Remove the "return" shift point.
 *            Recalculate longitudinal distance and modify the shift point.
 */
void AvoidanceModule::trimMomentaryReturn(AvoidLineArray & shift_lines) const
{
  const auto isZero = [](double v) { return std::abs(v) < 1.0e-5; };

  AvoidLineArray shift_lines_orig = shift_lines;
  shift_lines.clear();

  const double DISTANCE_AFTER_RETURN_THR = 5.0 * getNominalAvoidanceEgoSpeed();

  const auto & arclength = avoidance_data_.arclength_from_ego;

  const auto check_reduce_shift = [](const double now_length, const double prev_length) {
    const auto abs_shift_diff = std::abs(now_length) - std::abs(prev_length);
    const auto has_same_sign = (now_length * prev_length >= 0.0);
    const bool is_reduce_shift = (abs_shift_diff < 0.0 && has_same_sign);
    return is_reduce_shift;
  };

  for (size_t i = 0; i < shift_lines_orig.size(); ++i) {
    const auto sl_now = shift_lines_orig.at(i);
    const auto sl_prev_length =
      shift_lines.empty() ? getCurrentLinearShift() : shift_lines.back().end_shift_length;
    const auto abs_shift_diff = std::abs(sl_now.end_shift_length) - std::abs(sl_prev_length);
    const bool is_reduce_shift = check_reduce_shift(sl_now.end_shift_length, sl_prev_length);

    // Do nothing for non-reduce shift point
    if (!is_reduce_shift) {
      shift_lines.push_back(sl_now);
      DEBUG_PRINT(
        "i = %lu, not reduce shift. take this one.abs_shift_diff = %f, sl_now.length = %f, "
        "sl_prev_length = %f, sl_now.length * sl_prev_length = %f",
        i, abs_shift_diff, sl_now.end_shift_length, sl_prev_length,
        sl_now.end_shift_length * sl_prev_length);
      continue;
    }

    // The last point is out of target of this function.
    const bool is_last_sl = (i == shift_lines_orig.size() - 1);
    if (is_last_sl) {
      shift_lines.push_back(sl_now);
      DEBUG_PRINT("i = %lu, last shift. take this one.", i);
      continue;
    }

    // --- From here, the shift point is "return to center" or "straight". ---
    // -----------------------------------------------------------------------

    const auto sl_next = shift_lines_orig.at(i + 1);

    // there is no straight interval, combine them. ¯¯\/¯¯ -> ¯¯¯¯¯¯
    if (!isZero(sl_next.getRelativeLength())) {
      DEBUG_PRINT(
        "i = %lu, return-shift is detected, next shift_diff (%f) is nonzero. combine them. (skip "
        "next shift).",
        i, sl_next.getRelativeLength());
      auto sl_modified = sl_next;
      setStartData(
        sl_modified, sl_now.end_shift_length, sl_now.start, sl_now.start_idx,
        sl_now.start_longitudinal);
      sl_modified.parent_ids = concatParentIds(sl_modified.parent_ids, sl_now.parent_ids);
      shift_lines.push_back(sl_modified);
      ++i;  // skip next shift point
      continue;
    }

    // Find next shifting point, i.e.  ¯¯\____"/"¯¯
    //                               now ↑     ↑ target
    const auto next_avoid_idx = [&]() {
      for (size_t j = i + 1; j < shift_lines_orig.size(); ++j) {
        if (!isZero(shift_lines_orig.at(j).getRelativeLength())) {
          return j;
        }
      }
      return shift_lines_orig.size();
    }();

    // The straight distance lasts until end. take this one.
    // ¯¯\______
    if (next_avoid_idx == shift_lines_orig.size()) {
      shift_lines.push_back(sl_now);
      DEBUG_PRINT("i = %lu, back -> straight lasts until end. take this one.", i);
      continue;
    }

    const auto sl_next_avoid = shift_lines_orig.at(next_avoid_idx);
    const auto straight_distance = sl_next_avoid.start_longitudinal - sl_now.end_longitudinal;

    // The straight distance after "return to center" is long enough. take this one.
    // ¯¯\______/¯¯ (enough long straight line!)
    if (straight_distance > DISTANCE_AFTER_RETURN_THR) {
      shift_lines.push_back(sl_now);
      DEBUG_PRINT("i = %lu, back -> straight: distance is long. take this one", i);
      continue;
    }

    // From here, back to center and go straight, straight distance is too short.
    // ¯¯\______/¯¯ (short straight line!)

    const auto relative_shift = sl_next_avoid.end_shift_length - sl_now.end_shift_length;
    const auto avoid_distance = getNominalAvoidanceDistance(relative_shift);

    // Calculate start point from end point and avoidance distance.
    auto sl_next_modified = sl_next_avoid;
    sl_next_modified.start_shift_length = sl_prev_length;
    sl_next_modified.start_longitudinal =
      std::max(sl_next_avoid.end_longitudinal - avoid_distance, sl_now.start_longitudinal);
    sl_next_modified.start_idx =
      findPathIndexFromArclength(arclength, sl_next_modified.start_longitudinal);
    sl_next_modified.start =
      avoidance_data_.reference_path.points.at(sl_next_modified.start_idx).point.pose;
    sl_next_modified.parent_ids = calcParentIds(current_raw_shift_lines_, sl_next_modified);

    // Straight shift point
    if (sl_next_modified.start_idx > sl_now.start_idx) {  // the case where a straight route exists.
      auto sl_now_modified = sl_now;
      sl_now_modified.start_shift_length = sl_prev_length;
      setEndData(
        sl_now_modified, sl_prev_length, sl_next_modified.start, sl_next_modified.start_idx,
        sl_next_modified.start_longitudinal);
      sl_now_modified.parent_ids = calcParentIds(current_raw_shift_lines_, sl_now_modified);
      shift_lines.push_back(sl_now_modified);
    }
    shift_lines.push_back(sl_next_modified);

    DEBUG_PRINT(
      "i = %lu, find remove target!: next_avoid_idx = %lu, shift length = (now: %f, prev: %f, "
      "next_avoid: %f, next_mod: %f).",
      i, next_avoid_idx, sl_now.end_shift_length, sl_prev_length, sl_next_avoid.end_shift_length,
      sl_next_modified.end_shift_length);

    i = next_avoid_idx;  // skip shifting until next_avoid_idx.
  }

  alignShiftLinesOrder(shift_lines);

  DEBUG_PRINT("trimMomentaryReturn: size %lu -> %lu", shift_lines_orig.size(), shift_lines.size());
}

void AvoidanceModule::trimSharpReturn(AvoidLineArray & shift_lines) const
{
  AvoidLineArray shift_lines_orig = shift_lines;
  shift_lines.clear();

  const auto isZero = [](double v) { return std::abs(v) < 0.01; };

  // check if the shift point is positive (avoiding) shift
  const auto isPositive = [&](const auto & sl) {
    constexpr auto POSITIVE_SHIFT_THR = 0.1;
    return std::abs(sl.end_shift_length) - std::abs(sl.start_shift_length) > POSITIVE_SHIFT_THR;
  };

  // check if the shift point is negative (returning) shift
  const auto isNegative = [&](const auto & sl) {
    constexpr auto NEGATIVE_SHIFT_THR = -0.1;
    return std::abs(sl.end_shift_length) - std::abs(sl.start_shift_length) < NEGATIVE_SHIFT_THR;
  };

  // combine two shift points. Be careful the order of "now" and "next".
  const auto combineShiftLine = [this](const auto & sl_next, const auto & sl_now) {
    auto sl_modified = sl_now;
    setEndData(
      sl_modified, sl_next.end_shift_length, sl_next.end, sl_next.end_idx,
      sl_next.end_longitudinal);
    sl_modified.parent_ids = concatParentIds(sl_modified.parent_ids, sl_now.parent_ids);
    return sl_modified;
  };

  // Check if the merged shift has a conflict with the original shifts.
  const auto hasViolation = [this](const auto & combined, const auto & combined_src) {
    constexpr auto VIOLATION_SHIFT_THR = 0.3;
    for (const auto & sl : combined_src) {
      const auto combined_shift = lerpShiftLengthOnArc(sl.end_longitudinal, combined);
      if (
        sl.end_shift_length < -0.01 && combined_shift > sl.end_shift_length + VIOLATION_SHIFT_THR) {
        return true;
      }
      if (
        sl.end_shift_length > 0.01 && combined_shift < sl.end_shift_length - VIOLATION_SHIFT_THR) {
        return true;
      }
    }
    return false;
  };

  // check for all shift points
  for (size_t i = 0; i < shift_lines_orig.size(); ++i) {
    auto sl_now = shift_lines_orig.at(i);
    sl_now.start_shift_length =
      shift_lines.empty() ? getCurrentLinearShift() : shift_lines.back().end_shift_length;

    if (sl_now.end_shift_length * sl_now.start_shift_length < -0.01) {
      DEBUG_PRINT("i = %lu, This is avoid shift for opposite direction. take this one", i);
      continue;
    }

    // Do nothing for non-reduce shift point
    if (!isNegative(sl_now)) {
      shift_lines.push_back(sl_now);
      DEBUG_PRINT(
        "i = %lu, positive shift. take this one. sl_now.length * sl_now.start_length = %f", i,
        sl_now.end_shift_length * sl_now.start_shift_length);
      continue;
    }

    // The last point is out of target of this function.
    if (i == shift_lines_orig.size() - 1) {
      shift_lines.push_back(sl_now);
      DEBUG_PRINT("i = %lu, last shift. take this one.", i);
      continue;
    }

    // -----------------------------------------------------------------------
    // ------------ From here, the shift point is "negative" -----------------
    // -----------------------------------------------------------------------

    // if next shift is negative, combine them. loop until combined shift line
    // exceeds merged shift point.
    DEBUG_PRINT("i = %lu, found negative dist. search.", i);
    {
      auto sl_combined = sl_now;
      auto sl_combined_prev = sl_combined;
      AvoidLineArray sl_combined_array{sl_now};
      size_t j = i + 1;
      for (; i < shift_lines_orig.size(); ++j) {
        const auto sl_combined = combineShiftLine(shift_lines_orig.at(j), sl_now);

        {
          std::stringstream ss;
          ss << "i = " << i << ", j = " << j << ": sl_combined = " << toStrInfo(sl_combined);
          DEBUG_PRINT("%s", ss.str().c_str());
        }

        // it gets positive. Finish merging.
        if (isPositive(sl_combined)) {
          shift_lines.push_back(sl_combined);
          DEBUG_PRINT("reach positive.");
          break;
        }

        // Still negative, but it violates the original shift points.
        // Finish with the previous merge result.
        if (hasViolation(sl_combined, sl_combined_array)) {
          shift_lines.push_back(sl_combined_prev);
          DEBUG_PRINT("violation found.");
          --j;
          break;
        }

        // Still negative, but it has an enough long distance. Finish merging.
        const auto nominal_distance = getNominalAvoidanceDistance(sl_combined.getRelativeLength());
        const auto long_distance =
          isZero(sl_combined.end_shift_length) ? nominal_distance : nominal_distance * 5.0;
        if (sl_combined.getRelativeLongitudinal() > long_distance) {
          shift_lines.push_back(sl_combined);
          DEBUG_PRINT("still negative, but long enough. Threshold = %f", long_distance);
          break;
        }

        // It reaches the last point. Still the shift is sharp, but merge with the current result.
        if (j == shift_lines_orig.size() - 1) {
          shift_lines.push_back(sl_combined);
          DEBUG_PRINT("reach end point.");
          break;
        }

        // Still negative shift, and the distance is not enough. Search next.
        sl_combined_prev = sl_combined;
        sl_combined_array.push_back(shift_lines_orig.at(j));
      }
      i = j;
      continue;
    }
  }

  alignShiftLinesOrder(shift_lines);

  DEBUG_PRINT("trimSharpReturn: size %lu -> %lu", shift_lines_orig.size(), shift_lines.size());
}

void AvoidanceModule::trimTooSharpShift(AvoidLineArray & avoid_lines) const
{
  if (avoid_lines.empty()) {
    return;
  }

  AvoidLineArray avoid_lines_orig = avoid_lines;
  avoid_lines.clear();

  const auto isInJerkLimit = [this](const auto & al) {
    const auto required_jerk = path_shifter_.calcJerkFromLatLonDistance(
      al.getRelativeLength(), al.getRelativeLongitudinal(), getSharpAvoidanceEgoSpeed());
    return std::fabs(required_jerk) < parameters_->max_lateral_jerk;
  };

  for (size_t i = 0; i < avoid_lines_orig.size(); ++i) {
    auto al_now = avoid_lines_orig.at(i);

    if (isInJerkLimit(al_now)) {
      avoid_lines.push_back(al_now);
      continue;
    }

    DEBUG_PRINT("over jerk is detected: i = %lu", i);
    printShiftLines(AvoidLineArray{al_now}, "points with over jerk");

    // The avoidance_point_now exceeds jerk limit, so merge it with the next avoidance_point.
    for (size_t j = i + 1; j < avoid_lines_orig.size(); ++j) {
      auto al_next = avoid_lines_orig.at(j);
      setEndData(
        al_now, al_next.end_shift_length, al_next.end, al_next.end_idx, al_next.end_longitudinal);
      if (isInJerkLimit(al_now)) {
        avoid_lines.push_back(al_now);
        DEBUG_PRINT("merge finished. i = %lu, j = %lu", i, j);
        i = j;  // skip check until j index.
        break;
      }
    }
  }

  alignShiftLinesOrder(avoid_lines);

  DEBUG_PRINT("size %lu -> %lu", avoid_lines_orig.size(), avoid_lines.size());
}

/*
 * addReturnShiftLine
 *
 * Pick up the last shift point, which is the most farthest from ego, from the current candidate
 * avoidance points and registered points in the shifter. If the last shift length of the point is
 * non-zero, add a return-shift to center line from the point. If there is no shift point in
 * candidate avoidance points nor registered points, and base_shift > 0, add a return-shift to
 * center line from ego.
 */
void AvoidanceModule::addReturnShiftLineFromEgo(
  AvoidLineArray & sl_candidates, AvoidLineArray & current_raw_shift_lines) const
{
  constexpr double ep = 1.0e-3;
  const bool has_candidate_point = !sl_candidates.empty();
  const bool has_registered_point = !path_shifter_.getShiftLines().empty();

  // If the return-to-center shift points are already registered, do nothing.
  if (!has_registered_point && std::fabs(getCurrentBaseShift()) < ep) {
    DEBUG_PRINT("No shift points, not base offset. Do not have to add return-shift.");
    return;
  }

  constexpr double RETURN_SHIFT_THRESHOLD = 0.1;
  DEBUG_PRINT("registered last shift = %f", path_shifter_.getLastShiftLength());
  if (std::abs(path_shifter_.getLastShiftLength()) < RETURN_SHIFT_THRESHOLD) {
    DEBUG_PRINT("Return shift is already registered. do nothing.");
    return;
  }

  // From here, the return-to-center is not registered. But perhaps the candidate is
  // already generated.

  // If it has a shift point, add return shift from the existing last shift point.
  // If not, add return shift from ego point. (prepare distance is considered for both.)
  ShiftLine last_sl;  // the return-shift will be generated after the last shift point.
  {
    // avoidance points: Yes, shift points: No -> select last avoidance point.
    if (has_candidate_point && !has_registered_point) {
      alignShiftLinesOrder(sl_candidates, false);
      last_sl = sl_candidates.back();
    }

    // avoidance points: No, shift points: Yes -> select last shift point.
    if (!has_candidate_point && has_registered_point) {
      last_sl = fillAdditionalInfo(AvoidLine{path_shifter_.getLastShiftLine().get()});
    }

    // avoidance points: Yes, shift points: Yes -> select the last one from both.
    if (has_candidate_point && has_registered_point) {
      alignShiftLinesOrder(sl_candidates, false);
      const auto & al = sl_candidates.back();
      const auto & sl = fillAdditionalInfo(AvoidLine{path_shifter_.getLastShiftLine().get()});
      last_sl = (sl.end_longitudinal > al.end_longitudinal) ? sl : al;
    }

    // avoidance points: No, shift points: No -> set the ego position to the last shift point
    // so that the return-shift will be generated from ego position.
    if (!has_candidate_point && !has_registered_point) {
      last_sl.end = getEgoPose();
      last_sl.end_idx = avoidance_data_.ego_closest_path_index;
      last_sl.end_shift_length = getCurrentBaseShift();
    }
  }
  printShiftLines(ShiftLineArray{last_sl}, "last shift point");

  // There already is a shift point candidates to go back to center line, but it could be too sharp
  // due to detection noise or timing.
  // Here the return-shift from ego is added for the in case.
  if (std::fabs(last_sl.end_shift_length) < RETURN_SHIFT_THRESHOLD) {
    const auto current_base_shift = getCurrentShift();
    if (std::abs(current_base_shift) < ep) {
      DEBUG_PRINT("last shift almost is zero, and current base_shift is zero. do nothing.");
      return;
    }

    // Is there a shift point in the opposite direction of the current_base_shift?
    //   No  -> we can overwrite the return shift, because the other shift points that decrease
    //          the shift length are for return-shift.
    //   Yes -> we can NOT overwrite, because it might be not a return-shift, but a avoiding
    //          shift to the opposite direction which can not be overwritten by the return-shift.
    for (const auto & sl : sl_candidates) {
      if (
        (current_base_shift > 0.0 && sl.end_shift_length < -ep) ||
        (current_base_shift < 0.0 && sl.end_shift_length > ep)) {
        DEBUG_PRINT(
          "try to put overwrite return shift, but there is shift for opposite direction. Skip "
          "adding return shift.");
        return;
      }
    }

    // set the return-shift from ego.
    DEBUG_PRINT(
      "return shift already exists, but they are all candidates. Add return shift for overwrite.");
    last_sl.end = getEgoPose();
    last_sl.end_idx = avoidance_data_.ego_closest_path_index;
    last_sl.end_shift_length = current_base_shift;
  }

  const auto & arclength_from_ego = avoidance_data_.arclength_from_ego;

  const auto nominal_prepare_distance = getNominalPrepareDistance();
  const auto nominal_avoid_distance = getNominalAvoidanceDistance(last_sl.end_shift_length);

  if (arclength_from_ego.empty()) {
    return;
  }

  const auto remaining_distance = arclength_from_ego.back();

  // If the avoidance point has already been set, the return shift must be set after the point.
  const auto last_sl_distance = avoidance_data_.arclength_from_ego.at(last_sl.end_idx);

  // check if there is enough distance for return.
  if (last_sl_distance + 1.0 > remaining_distance) {  // tmp: add some small number (+1.0)
    DEBUG_PRINT("No enough distance for return.");
    return;
  }

  // If the remaining distance is not enough, the return shift needs to be shrunk.
  // (or another option is just to ignore the return-shift.)
  // But we do not want to change the last shift point, so we will shrink the distance after
  // the last shift point.
  //
  //  The line "===" is fixed, "---" is scaled.
  //
  // [Before Scaling]
  //  ego              last_sl_end             prepare_end            path_end    avoid_end
  // ==o====================o----------------------o----------------------o------------o
  //   |            prepare_dist                   |          avoid_dist               |
  //
  // [After Scaling]
  // ==o====================o------------------o--------------------------o
  //   |        prepare_dist_scaled            |    avoid_dist_scaled     |
  //
  const double variable_prepare_distance =
    std::max(nominal_prepare_distance - last_sl_distance, 0.0);

  double prepare_distance_scaled = std::max(nominal_prepare_distance, last_sl_distance);
  double avoid_distance_scaled = nominal_avoid_distance;
  if (remaining_distance < prepare_distance_scaled + avoid_distance_scaled) {
    const auto scale = (remaining_distance - last_sl_distance) /
                       std::max(nominal_avoid_distance + variable_prepare_distance, 0.1);
    prepare_distance_scaled = last_sl_distance + scale * nominal_prepare_distance;
    avoid_distance_scaled *= scale;
    DEBUG_PRINT(
      "last_sl_distance = %f, nominal_prepare_distance = %f, nominal_avoid_distance = %f, "
      "remaining_distance = %f, variable_prepare_distance = %f, scale = %f, "
      "prepare_distance_scaled = %f,avoid_distance_scaled = %f",
      last_sl_distance, nominal_prepare_distance, nominal_avoid_distance, remaining_distance,
      variable_prepare_distance, scale, prepare_distance_scaled, avoid_distance_scaled);
  } else {
    DEBUG_PRINT("there is enough distance. Use nominal for prepare & avoidance.");
  }

  // shift point for prepare distance: from last shift to return-start point.
  if (nominal_prepare_distance > last_sl_distance) {
    AvoidLine al;
    al.id = getOriginalShiftLineUniqueId();
    al.start_idx = last_sl.end_idx;
    al.start = last_sl.end;
    al.start_longitudinal = arclength_from_ego.at(al.start_idx);
    al.end_idx = findPathIndexFromArclength(arclength_from_ego, prepare_distance_scaled);
    al.end = avoidance_data_.reference_path.points.at(al.end_idx).point.pose;
    al.end_longitudinal = arclength_from_ego.at(al.end_idx);
    al.end_shift_length = last_sl.end_shift_length;
    al.start_shift_length = last_sl.end_shift_length;
    sl_candidates.push_back(al);
    printShiftLines(AvoidLineArray{al}, "prepare for return");
    debug_data_.extra_return_shift.push_back(al);

    // TODO(Horibe) think how to store the current object
    current_raw_shift_lines.push_back(al);
  }

  // shift point for return to center line
  {
    AvoidLine al;
    al.id = getOriginalShiftLineUniqueId();
    al.start_idx = findPathIndexFromArclength(arclength_from_ego, prepare_distance_scaled);
    al.start = avoidance_data_.reference_path.points.at(al.start_idx).point.pose;
    al.start_longitudinal = arclength_from_ego.at(al.start_idx);
    al.end_idx = findPathIndexFromArclength(
      arclength_from_ego, prepare_distance_scaled + avoid_distance_scaled);
    al.end = avoidance_data_.reference_path.points.at(al.end_idx).point.pose;
    al.end_longitudinal = arclength_from_ego.at(al.end_idx);
    al.end_shift_length = 0.0;
    al.start_shift_length = last_sl.end_shift_length;
    sl_candidates.push_back(al);
    printShiftLines(AvoidLineArray{al}, "return point");
    debug_data_.extra_return_shift = AvoidLineArray{al};

    // TODO(Horibe) think how to store the current object
    current_raw_shift_lines.push_back(al);
  }

  DEBUG_PRINT("Return Shift is added.");
}

bool AvoidanceModule::isSafePath(
  const PathShifter & path_shifter, ShiftedPath & shifted_path, DebugData & debug) const
{
  const auto & p = parameters_;

  if (!p->enable_safety_check) {
    return true;  // if safety check is disabled, it always return safe.
  }

  const auto & forward_check_distance = p->object_check_forward_distance;
  const auto & backward_check_distance = p->safety_check_backward_distance;
  const auto check_lanes =
    getAdjacentLane(path_shifter, forward_check_distance, backward_check_distance);

  auto path_with_current_velocity = shifted_path.path;
  path_with_current_velocity = util::resamplePathWithSpline(path_with_current_velocity, 0.5);

  const size_t ego_idx = planner_data_->findEgoIndex(path_with_current_velocity.points);
  util::clipPathLength(path_with_current_velocity, ego_idx, forward_check_distance, 0.0);

  constexpr double MIN_EGO_VEL_IN_PREDICTION = 1.38;  // 5km/h
  for (auto & p : path_with_current_velocity.points) {
    p.point.longitudinal_velocity_mps = std::max(getEgoSpeed(), MIN_EGO_VEL_IN_PREDICTION);
  }

  {
    debug_data_.path_with_planned_velocity = path_with_current_velocity;
  }

  return isSafePath(path_with_current_velocity, check_lanes, debug);
}

bool AvoidanceModule::isSafePath(
  const PathWithLaneId & path, const lanelet::ConstLanelets & check_lanes, DebugData & debug) const
{
  if (path.points.empty()) {
    return true;
  }

  const auto path_with_time = [&path]() {
    std::vector<std::pair<PathPointWithLaneId, double>> ret{};

    float travel_time = 0.0;
    ret.emplace_back(path.points.front(), travel_time);

    for (size_t i = 1; i < path.points.size(); ++i) {
      const auto & p1 = path.points.at(i - 1);
      const auto & p2 = path.points.at(i);

      const auto v = std::max(p1.point.longitudinal_velocity_mps, float{1.0});
      const auto ds = calcDistance2d(p1, p2);

      travel_time += ds / v;

      ret.emplace_back(p2, travel_time);
    }

    return ret;
  }();

  const auto move_objects = getAdjacentLaneObjects(check_lanes);

  {
    debug.unsafe_objects.clear();
    debug.margin_data_array.clear();
    debug.exist_adjacent_objects = !move_objects.empty();
  }

  bool is_safe = true;
  for (const auto & p : path_with_time) {
    MarginData margin_data{};
    margin_data.pose = getPose(p.first);

    if (p.second > parameters_->safety_check_time_horizon) {
      break;
    }

    for (const auto & o : move_objects) {
      const auto is_enough_margin = isEnoughMargin(p.first, p.second, o, margin_data);

      if (!is_enough_margin) {
        debug.unsafe_objects.push_back(o);
      }

      is_safe = is_safe && is_enough_margin;
    }

    debug.margin_data_array.push_back(margin_data);
  }

  return is_safe;
}

bool AvoidanceModule::isEnoughMargin(
  const PathPointWithLaneId & p_ego, const double t, const ObjectData & object,
  MarginData & margin_data) const
{
  const auto & common_param = planner_data_->parameters;
  const auto & vehicle_width = common_param.vehicle_width;
  const auto & base_link2front = common_param.base_link2front;
  const auto & base_link2rear = common_param.base_link2rear;

  const auto p_ref = [this, &p_ego]() {
    const auto idx = findNearestIndex(avoidance_data_.reference_path.points, getPoint(p_ego));
    return getPose(avoidance_data_.reference_path.points.at(idx));
  }();

  const auto & v_ego = p_ego.point.longitudinal_velocity_mps;
  const auto & v_ego_lon = getLongitudinalVelocity(p_ref, getPose(p_ego), v_ego);
  const auto & v_obj = object.object.kinematics.initial_twist_with_covariance.twist.linear.x;

  if (!isTargetObjectType(object.object)) {
    return true;
  }

  // |           centerline
  // |               ^ x
  // |  +-------+    |
  // |  |       |    |
  // |  |       | D1 |     D2      D4
  // |  |  obj  |<-->|<---------->|<->|
  // |  |       | D3 |        +-------+
  // |  |       |<----------->|       |
  // |  +-------+    |        |       |
  // |               |        |  ego  |
  // |               |        |       |
  // |               |        |       |
  // |               |        +-------+
  // |        y <----+
  // D1: overhang_dist (signed value)
  // D2: shift_length (signed value)
  // D3: lateral_distance (should be larger than margin that's calculated from relative velocity.)
  // D4: vehicle_width (unsigned value)

  const auto reliable_path = std::max_element(
    object.object.kinematics.predicted_paths.begin(),
    object.object.kinematics.predicted_paths.end(),
    [](const PredictedPath & a, const PredictedPath & b) { return a.confidence < b.confidence; });

  if (reliable_path == object.object.kinematics.predicted_paths.end()) {
    return true;
  }

  const auto p_obj = [&t, &reliable_path]() {
    boost::optional<Pose> ret{boost::none};

    const auto dt = rclcpp::Duration(reliable_path->time_step).seconds();
    const auto idx = static_cast<size_t>(std::floor(t / dt));
    const auto res = t - dt * idx;

    if (idx > reliable_path->path.size() - 2) {
      return ret;
    }

    const auto & p_src = reliable_path->path.at(idx);
    const auto & p_dst = reliable_path->path.at(idx + 1);
    ret = calcInterpolatedPose(p_src, p_dst, res / dt);
    return ret;
  }();

  if (!p_obj) {
    return true;
  }

  const auto v_obj_lon = getLongitudinalVelocity(p_ref, p_obj.get(), v_obj);

  double hysteresis_factor = 1.0;
  if (avoidance_data_.state == AvoidanceState::YIELD) {
    hysteresis_factor = parameters_->safety_check_hysteresis_factor;
  }

  const auto shift_length = calcLateralDeviation(p_ref, getPoint(p_ego));
  const auto lateral_distance = std::abs(object.overhang_dist - shift_length) - 0.5 * vehicle_width;
  const auto lateral_margin = getLateralMarginFromVelocity(std::abs(v_ego_lon - v_obj_lon));

  if (lateral_distance > lateral_margin * hysteresis_factor) {
    return true;
  }

  const auto lon_deviation = calcLongitudinalDeviation(getPose(p_ego), p_obj.get().position);
  const auto is_front_object = lon_deviation > 0.0;
  const auto longitudinal_margin =
    getRSSLongitudinalDistance(v_ego_lon, v_obj_lon, is_front_object);
  const auto vehicle_offset = is_front_object ? base_link2front : base_link2rear;
  const auto longitudinal_distance =
    std::abs(lon_deviation) - vehicle_offset - 0.5 * object.object.shape.dimensions.x;

  {
    margin_data.pose.orientation = p_ref.orientation;
    margin_data.enough_lateral_margin = false;
    margin_data.longitudinal_distance =
      std::min(margin_data.longitudinal_distance, longitudinal_distance);
    margin_data.longitudinal_margin =
      std::max(margin_data.longitudinal_margin, longitudinal_margin);
    margin_data.vehicle_width = vehicle_width;
    margin_data.base_link2front = base_link2front;
    margin_data.base_link2rear = base_link2rear;
  }

  if (longitudinal_distance > longitudinal_margin * hysteresis_factor) {
    return true;
  }

  return false;
}

double AvoidanceModule::getLateralMarginFromVelocity(const double velocity) const
{
  const auto & p = parameters_;

  if (p->col_size < 2 || p->col_size * 2 != p->target_velocity_matrix.size()) {
    throw std::logic_error("invalid matrix col size.");
  }

  if (velocity < p->target_velocity_matrix.front()) {
    return p->target_velocity_matrix.at(p->col_size);
  }

  if (velocity > p->target_velocity_matrix.at(p->col_size - 1)) {
    return p->target_velocity_matrix.back();
  }

  for (size_t i = 1; i < p->col_size; ++i) {
    if (velocity < p->target_velocity_matrix.at(i)) {
      const auto v1 = p->target_velocity_matrix.at(i - 1);
      const auto v2 = p->target_velocity_matrix.at(i);
      const auto m1 = p->target_velocity_matrix.at(i - 1 + p->col_size);
      const auto m2 = p->target_velocity_matrix.at(i + p->col_size);

      const auto v_clamp = std::clamp(velocity, v1, v2);
      return m1 + (m2 - m1) * (v_clamp - v1) / (v2 - v1);
    }
  }

  return p->target_velocity_matrix.back();
}

double AvoidanceModule::getRSSLongitudinalDistance(
  const double v_ego, const double v_obj, const bool is_front_object) const
{
  const auto & accel_for_rss = parameters_->safety_check_accel_for_rss;
  const auto & idling_time = parameters_->safety_check_idling_time;

  const auto opposite_lane_vehicle = v_obj < 0.0;

  /**
   * object and ego already pass each other.
   * =======================================
   *                          Ego-->
   * ---------------------------------------
   *       <--Obj
   * =======================================
   */
  if (!is_front_object && opposite_lane_vehicle) {
    return 0.0;
  }

  /**
   * object drive opposite direction.
   * =======================================
   *       Ego-->
   * ---------------------------------------
   *                          <--Obj
   * =======================================
   */
  if (is_front_object && opposite_lane_vehicle) {
    return v_ego * idling_time + 0.5 * accel_for_rss * std::pow(idling_time, 2.0) +
           std::pow(v_ego + accel_for_rss * idling_time, 2.0) / (2.0 * accel_for_rss) +
           std::abs(v_obj) * idling_time + 0.5 * accel_for_rss * std::pow(idling_time, 2.0) +
           std::pow(v_obj + accel_for_rss * idling_time, 2.0) / (2.0 * accel_for_rss);
  }

  /**
   * object is in front of ego, and drive same direction.
   * =======================================
   *       Ego-->
   * ---------------------------------------
   *                          Obj-->
   * =======================================
   */
  if (is_front_object && !opposite_lane_vehicle) {
    return v_ego * idling_time + 0.5 * accel_for_rss * std::pow(idling_time, 2.0) +
           std::pow(v_ego + accel_for_rss * idling_time, 2.0) / (2.0 * accel_for_rss) -
           std::pow(v_obj, 2.0) / (2.0 * accel_for_rss);
  }

  /**
   * object is behind ego, and drive same direction.
   * =======================================
   *                          Ego-->
   * ---------------------------------------
   *       Obj-->
   * =======================================
   */
  if (!is_front_object && !opposite_lane_vehicle) {
    return v_obj * idling_time + 0.5 * accel_for_rss * std::pow(idling_time, 2.0) +
           std::pow(v_obj + accel_for_rss * idling_time, 2.0) / (2.0 * accel_for_rss) -
           std::pow(v_ego, 2.0) / (2.0 * accel_for_rss);
  }

  return 0.0;
}

lanelet::ConstLanelets AvoidanceModule::getAdjacentLane(
  const PathShifter & path_shifter, const double forward_distance,
  const double backward_distance) const
{
  const auto & rh = planner_data_->route_handler;

  bool has_left_shift = false;
  bool has_right_shift = false;

  for (const auto & sp : path_shifter.getShiftLines()) {
    if (sp.end_shift_length > 0.01) {
      has_left_shift = true;
      continue;
    }

    if (sp.end_shift_length < -0.01) {
      has_right_shift = true;
      continue;
    }
  }

  lanelet::ConstLanelet current_lane;
  if (!rh->getClosestLaneletWithinRoute(getEgoPose(), &current_lane)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("behavior_path_planner").get_child("avoidance"),
      "failed to find closest lanelet within route!!!");
    return {};  // TODO(Satoshi Ota)
  }

  const auto ego_succeeding_lanes =
    rh->getLaneletSequence(current_lane, getEgoPose(), backward_distance, forward_distance);

  lanelet::ConstLanelets check_lanes{};
  for (const auto & lane : ego_succeeding_lanes) {
    const auto opt_left_lane = rh->getLeftLanelet(lane);
    if (has_left_shift && opt_left_lane) {
      check_lanes.push_back(opt_left_lane.get());
    }

    const auto opt_right_lane = rh->getRightLanelet(lane);
    if (has_right_shift && opt_right_lane) {
      check_lanes.push_back(opt_right_lane.get());
    }

    const auto right_opposite_lanes = rh->getRightOppositeLanelets(lane);
    if (has_right_shift && !right_opposite_lanes.empty()) {
      check_lanes.push_back(right_opposite_lanes.front());
    }
  }

  return check_lanes;
}

ObjectDataArray AvoidanceModule::getAdjacentLaneObjects(
  const lanelet::ConstLanelets & adjacent_lanes) const
{
  ObjectDataArray objects;
  for (const auto & o : avoidance_data_.other_objects) {
    if (isCentroidWithinLanelets(o.object, adjacent_lanes)) {
      objects.push_back(o);
    }
  }

  return objects;
}

// TODO(murooka) judge when and which way to extend drivable area. current implementation is keep
// extending during avoidance module
// TODO(murooka) freespace during turning in intersection where there is no neighbor lanes
// NOTE: Assume that there is no situation where there is an object in the middle lane of more than
// two lanes since which way to avoid is not obvious
void AvoidanceModule::generateExtendedDrivableArea(PathWithLaneId & path) const
{
  const auto has_same_lane =
    [](const lanelet::ConstLanelets lanes, const lanelet::ConstLanelet & lane) {
      if (lanes.empty()) return false;
      const auto has_same = [&](const auto & ll) { return ll.id() == lane.id(); };
      return std::find_if(lanes.begin(), lanes.end(), has_same) != lanes.end();
    };

  const auto & route_handler = planner_data_->route_handler;
  const auto & current_lanes = avoidance_data_.current_lanelets;
  const auto & enable_opposite = parameters_->enable_avoidance_over_opposite_direction;
  std::vector<DrivableLanes> drivable_lanes;

  for (const auto & current_lane : current_lanes) {
    DrivableLanes current_drivable_lanes;
    current_drivable_lanes.left_lane = current_lane;
    current_drivable_lanes.right_lane = current_lane;

    if (!parameters_->enable_avoidance_over_same_direction) {
      drivable_lanes.push_back(current_drivable_lanes);
      continue;
    }

    // 1. get left/right side lanes
    const auto update_left_lanelets = [&](const lanelet::ConstLanelet & target_lane) {
      const auto all_left_lanelets =
        route_handler->getAllLeftSharedLinestringLanelets(target_lane, enable_opposite, true);
      if (!all_left_lanelets.empty()) {
        current_drivable_lanes.left_lane = all_left_lanelets.back();  // leftmost lanelet
        pushUniqueVector(
          current_drivable_lanes.middle_lanes,
          lanelet::ConstLanelets(all_left_lanelets.begin(), all_left_lanelets.end() - 1));
      }
    };
    const auto update_right_lanelets = [&](const lanelet::ConstLanelet & target_lane) {
      const auto all_right_lanelets =
        route_handler->getAllRightSharedLinestringLanelets(target_lane, enable_opposite, true);
      if (!all_right_lanelets.empty()) {
        current_drivable_lanes.right_lane = all_right_lanelets.back();  // rightmost lanelet
        pushUniqueVector(
          current_drivable_lanes.middle_lanes,
          lanelet::ConstLanelets(all_right_lanelets.begin(), all_right_lanelets.end() - 1));
      }
    };

    update_left_lanelets(current_lane);
    update_right_lanelets(current_lane);

    // 2.1 when there are multiple lanes whose previous lanelet is the same
    const auto get_next_lanes_from_same_previous_lane =
      [&route_handler](const lanelet::ConstLanelet & lane) {
        // get previous lane, and return false if previous lane does not exist
        lanelet::ConstLanelets prev_lanes;
        if (!route_handler->getPreviousLaneletsWithinRoute(lane, &prev_lanes)) {
          return lanelet::ConstLanelets{};
        }

        lanelet::ConstLanelets next_lanes;
        for (const auto & prev_lane : prev_lanes) {
          const auto next_lanes_from_prev = route_handler->getNextLanelets(prev_lane);
          pushUniqueVector(next_lanes, next_lanes_from_prev);
        }
        return next_lanes;
      };

    const auto next_lanes_for_right =
      get_next_lanes_from_same_previous_lane(current_drivable_lanes.right_lane);
    const auto next_lanes_for_left =
      get_next_lanes_from_same_previous_lane(current_drivable_lanes.left_lane);

    // 2.2 look for neighbor lane recursively, where end line of the lane is connected to end line
    // of the original lane
    const auto update_drivable_lanes =
      [&](const lanelet::ConstLanelets & next_lanes, const bool is_left) {
        for (const auto & next_lane : next_lanes) {
          const auto & edge_lane =
            is_left ? current_drivable_lanes.left_lane : current_drivable_lanes.right_lane;
          if (next_lane.id() == edge_lane.id()) {
            continue;
          }

          const auto & left_lane = is_left ? next_lane : edge_lane;
          const auto & right_lane = is_left ? edge_lane : next_lane;
          if (!isEndPointsConnected(left_lane, right_lane)) {
            continue;
          }

          if (is_left) {
            current_drivable_lanes.left_lane = next_lane;
          } else {
            current_drivable_lanes.right_lane = next_lane;
          }

          if (!has_same_lane(current_drivable_lanes.middle_lanes, edge_lane)) {
            if (is_left) {
              if (current_drivable_lanes.right_lane.id() != edge_lane.id()) {
                current_drivable_lanes.middle_lanes.push_back(edge_lane);
              }
            } else {
              if (current_drivable_lanes.left_lane.id() != edge_lane.id()) {
                current_drivable_lanes.middle_lanes.push_back(edge_lane);
              }
            }
          }

          return true;
        }
        return false;
      };

    const auto expand_drivable_area_recursively =
      [&](const lanelet::ConstLanelets & next_lanes, const bool is_left) {
        // NOTE: set max search num to avoid infinity loop for drivable area expansion
        constexpr size_t max_recursive_search_num = 3;
        for (size_t i = 0; i < max_recursive_search_num; ++i) {
          const bool is_update_kept = update_drivable_lanes(next_lanes, is_left);
          if (!is_update_kept) {
            break;
          }
          if (i == max_recursive_search_num - 1) {
            RCLCPP_ERROR(
              rclcpp::get_logger("behavior_path_planner").get_child("avoidance"),
              "Drivable area expansion reaches max iteration.");
          }
        }
      };
    expand_drivable_area_recursively(next_lanes_for_right, false);
    expand_drivable_area_recursively(next_lanes_for_left, true);

    // 3. update again for new left/right lanes
    update_left_lanelets(current_drivable_lanes.left_lane);
    update_right_lanelets(current_drivable_lanes.right_lane);

    // 4. compensate that current_lane is in either of left_lane, right_lane or middle_lanes.
    if (
      current_drivable_lanes.left_lane.id() != current_lane.id() &&
      current_drivable_lanes.right_lane.id() != current_lane.id()) {
      current_drivable_lanes.middle_lanes.push_back(current_lane);
    }

    drivable_lanes.push_back(current_drivable_lanes);
  }

  const auto shorten_lanes = util::cutOverlappedLanes(path, drivable_lanes);

  const auto extended_lanes = util::expandLanelets(
    shorten_lanes, parameters_->drivable_area_left_bound_offset,
    parameters_->drivable_area_right_bound_offset, parameters_->drivable_area_types_to_skip);

  {
    const auto & p = planner_data_->parameters;
    generateDrivableArea(
      path, drivable_lanes, p.vehicle_length, planner_data_, avoidance_data_.target_objects,
      parameters_->enable_bound_clipping, parameters_->disable_path_update,
      parameters_->object_envelope_buffer);
  }
}

void AvoidanceModule::modifyPathVelocityToPreventAccelerationOnAvoidance(ShiftedPath & path)
{
  const auto ego_idx = avoidance_data_.ego_closest_path_index;
  const auto N = path.shift_length.size();

  if (!ego_velocity_starting_avoidance_ptr_) {
    ego_velocity_starting_avoidance_ptr_ = std::make_shared<double>(getEgoSpeed());
  }

  // find first shift-change point from ego
  constexpr auto SHIFT_DIFF_THR = 0.1;
  size_t target_idx = N;
  const auto current_shift = path.shift_length.at(ego_idx);
  for (size_t i = ego_idx + 1; i < N; ++i) {
    if (std::abs(path.shift_length.at(i) - current_shift) > SHIFT_DIFF_THR) {
      // this index do not have to be accurate, so it can be i or i + 1.
      // but if the ego point is already on the shift-change point, ego index should be a target_idx
      // so that the distance for acceleration will be 0 and the ego speed is directly applied
      // to the path velocity (no acceleration while avoidance)
      target_idx = i - 1;
      break;
    }
  }
  if (target_idx == N) {
    DEBUG_PRINT("shift length has no changes. No velocity limit is applied.");
    return;
  }

  constexpr auto NO_ACCEL_TIME_THR = 3.0;

  // update ego velocity if the shift point is far
  const auto s_from_ego = avoidance_data_.arclength_from_ego.at(target_idx) -
                          avoidance_data_.arclength_from_ego.at(ego_idx);
  const auto t_from_ego = s_from_ego / std::max(getEgoSpeed(), 1.0);
  if (t_from_ego > NO_ACCEL_TIME_THR) {
    *ego_velocity_starting_avoidance_ptr_ = getEgoSpeed();
  }

  // calc index and velocity to NO_ACCEL_TIME_THR
  const auto v0 = *ego_velocity_starting_avoidance_ptr_;
  auto vmax = 0.0;
  size_t insert_idx = ego_idx;
  for (size_t i = ego_idx; i <= target_idx; ++i) {
    const auto s =
      avoidance_data_.arclength_from_ego.at(target_idx) - avoidance_data_.arclength_from_ego.at(i);
    const auto t = s / std::max(v0, 1.0);
    if (t < NO_ACCEL_TIME_THR) {
      insert_idx = i;
      vmax = std::max(
        parameters_->min_avoidance_speed_for_acc_prevention,
        std::sqrt(v0 * v0 + 2.0 * s * parameters_->max_avoidance_acceleration));
      break;
    }
  }

  // apply velocity limit
  constexpr size_t V_LIM_APPLY_IDX_MARGIN = 0;
  for (size_t i = insert_idx + V_LIM_APPLY_IDX_MARGIN; i < N; ++i) {
    path.path.points.at(i).point.longitudinal_velocity_mps =
      std::min(path.path.points.at(i).point.longitudinal_velocity_mps, static_cast<float>(vmax));
  }

  DEBUG_PRINT(
    "s: %f, t: %f, v0: %f, a: %f, vmax: %f, ego_i: %lu, target_i: %lu", s_from_ego, t_from_ego, v0,
    parameters_->max_avoidance_acceleration, vmax, ego_idx, target_idx);
}

PathWithLaneId AvoidanceModule::extendBackwardLength(const PathWithLaneId & original_path) const
{
  // special for avoidance: take behind distance upt ot shift-start-point if it exist.
  const auto longest_dist_to_shift_point = [&]() {
    double max_dist = 0.0;
    for (const auto & pnt : path_shifter_.getShiftLines()) {
      max_dist = std::max(max_dist, calcDistance2d(getEgoPose(), pnt.start));
    }
    for (const auto & sp : registered_raw_shift_lines_) {
      max_dist = std::max(max_dist, calcDistance2d(getEgoPose(), sp.start));
    }
    return max_dist;
  }();

  const auto extra_margin = 10.0;  // Since distance does not consider arclength, but just line.
  const auto backward_length = std::max(
    planner_data_->parameters.backward_path_length, longest_dist_to_shift_point + extra_margin);

  const size_t orig_ego_idx = findNearestIndex(original_path.points, getEgoPosition());
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

// TODO(Horibe) clean up functions: there is a similar code in util as well.
PathWithLaneId AvoidanceModule::calcCenterLinePath(
  const std::shared_ptr<const PlannerData> & planner_data, const Pose & pose) const
{
  const auto & p = planner_data->parameters;
  const auto & route_handler = planner_data->route_handler;

  PathWithLaneId centerline_path;

  // special for avoidance: take behind distance upt ot shift-start-point if it exist.
  const auto longest_dist_to_shift_line = [&]() {
    double max_dist = 0.0;
    for (const auto & pnt : path_shifter_.getShiftLines()) {
      max_dist = std::max(max_dist, calcDistance2d(getEgoPose(), pnt.start));
    }
    for (const auto & sl : registered_raw_shift_lines_) {
      max_dist = std::max(max_dist, calcDistance2d(getEgoPose(), sl.start));
    }
    return max_dist;
  }();

  printShiftLines(path_shifter_.getShiftLines(), "path_shifter_.getShiftLines()");
  printShiftLines(registered_raw_shift_lines_, "registered_raw_shift_lines_");

  const auto extra_margin = 10.0;  // Since distance does not consider arclength, but just line.
  const auto backward_length =
    std::max(p.backward_path_length, longest_dist_to_shift_line + extra_margin);

  DEBUG_PRINT(
    "p.backward_path_length = %f, longest_dist_to_shift_line = %f, backward_length = %f",
    p.backward_path_length, longest_dist_to_shift_line, backward_length);

  const lanelet::ConstLanelets current_lanes =
    util::calcLaneAroundPose(route_handler, pose, p.forward_path_length, backward_length);
  centerline_path = util::getCenterLinePath(
    *route_handler, current_lanes, pose, backward_length, p.forward_path_length, p);

  // for debug: check if the path backward distance is same as the desired length.
  // {
  //   const auto back_to_ego = motion_utils::calcSignedArcLength(
  //     centerline_path.points, centerline_path.points.front().point.pose.position,
  //     getEgoPosition());
  //   RCLCPP_INFO(getLogger(), "actual back_to_ego distance = %f", back_to_ego);
  // }

  centerline_path.header = route_handler->getRouteHeader();

  return centerline_path;
}

boost::optional<AvoidLine> AvoidanceModule::calcIntersectionShiftLine(
  const AvoidancePlanningData & data) const
{
  boost::optional<PathPointWithLaneId> intersection_point{};
  for (const auto & p : avoidance_data_.reference_path.points) {
    for (const auto & id : p.lane_ids) {
      const lanelet::ConstLanelet ll = planner_data_->route_handler->getLaneletsFromId(id);
      std::string turn_direction = ll.attributeOr("turn_direction", "else");
      if (turn_direction == "right" || turn_direction == "left") {
        intersection_point = p;
        RCLCPP_INFO(getLogger(), "intersection is found.");
        break;
      }
    }
    if (intersection_point) {
      break;
    }
  }

  const auto calcBehindPose = [&data](const Point & p, const double dist) {
    const auto & path = data.reference_path;
    const size_t start = findNearestIndex(path.points, p);
    double sum = 0.0;
    for (size_t i = start - 1; i > 1; --i) {
      sum += calcDistance2d(path.points.at(i), path.points.at(i + 1));
      if (sum > dist) {
        return path.points.at(i).point.pose;
      }
    }
    return path.points.at(0).point.pose;
  };

  const auto intersection_shift_line = [&]() {
    boost::optional<AvoidLine> shift_line{};
    if (!intersection_point) {
      RCLCPP_INFO(getLogger(), "No intersection.");
      return shift_line;
    }

    const double ego_to_intersection_dist = calcSignedArcLength(
      data.reference_path.points, getEgoPosition(), intersection_point->point.pose.position);

    if (ego_to_intersection_dist <= 5.0) {
      RCLCPP_INFO(getLogger(), "No enough margin to intersection.");
      return shift_line;
    }

    // Search obstacles around the intersection.
    // If it exists, use its shift distance on the intersection.
    constexpr double intersection_obstacle_check_dist = 10.0;
    constexpr double intersection_shift_margin = 1.0;

    double shift_length = 0.0;  // default (no obstacle) is zero.
    for (const auto & obj : avoidance_data_.target_objects) {
      if (
        std::abs(obj.longitudinal - ego_to_intersection_dist) > intersection_obstacle_check_dist) {
        continue;
      }
      if (isOnRight(obj)) {
        continue;  // TODO(Horibe) Now only think about the left side obstacle.
      }
      shift_length = std::min(shift_length, obj.overhang_dist - intersection_shift_margin);
    }
    RCLCPP_INFO(getLogger(), "Intersection shift_length = %f", shift_length);

    AvoidLine p{};
    p.end_shift_length = shift_length;
    p.start =
      calcBehindPose(intersection_point->point.pose.position, intersection_obstacle_check_dist);
    p.end = intersection_point->point.pose;
    shift_line = p;
    return shift_line;
  }();

  return intersection_shift_line;
}

BehaviorModuleOutput AvoidanceModule::plan()
{
  const auto & data = avoidance_data_;

  resetPathCandidate();
  resetPathReference();

  /**
   * Has new shift point?
   *   Yes -> Is it approved?
   *       Yes -> add the shift point.
   *       No  -> set approval_handler to WAIT_APPROVAL state.
   *   No -> waiting approval?
   *       Yes -> clear WAIT_APPROVAL state.
   *       No  -> do nothing.
   */
  if (!data.safe_new_sl.empty()) {
    debug_data_.new_shift_lines = data.safe_new_sl;
    DEBUG_PRINT("new_shift_lines size = %lu", data.safe_new_sl.size());
    printShiftLines(data.safe_new_sl, "new_shift_lines");

    const auto sl = getNonStraightShiftLine(data.safe_new_sl);
    if (getRelativeLengthFromPath(sl) > 0.0) {
      removePreviousRTCStatusRight();
    } else if (getRelativeLengthFromPath(sl) < 0.0) {
      removePreviousRTCStatusLeft();
    } else {
      RCLCPP_WARN_STREAM(getLogger(), "Direction is UNKNOWN");
    }
    if (!parameters_->disable_path_update) {
      addShiftLineIfApproved(data.safe_new_sl);
    }
  } else if (isWaitingApproval()) {
    clearWaitingApproval();
    removeCandidateRTCStatus();
  }

  // generate path with shift points that have been inserted.
  auto avoidance_path = generateAvoidancePath(path_shifter_);
  debug_data_.output_shift = avoidance_path.shift_length;

  // modify max speed to prevent acceleration in avoidance maneuver.
  modifyPathVelocityToPreventAccelerationOnAvoidance(avoidance_path);

  // post processing
  {
    postProcess(path_shifter_);  // remove old shift points
    prev_output_ = avoidance_path;
    prev_linear_shift_path_ = toShiftedPath(avoidance_data_.reference_path);
    path_shifter_.generate(&prev_linear_shift_path_, true, SHIFT_TYPE::LINEAR);
    prev_reference_ = avoidance_data_.reference_path;
  }

  BehaviorModuleOutput output;
  output.turn_signal_info = calcTurnSignalInfo(avoidance_path);
  // sparse resampling for computational cost
  {
    avoidance_path.path =
      util::resamplePathWithSpline(avoidance_path.path, parameters_->resample_interval_for_output);
  }

  avoidance_data_.state = updateEgoState(data);
  if (!parameters_->disable_path_update) {
    updateEgoBehavior(data, avoidance_path);
  }

  if (parameters_->publish_debug_marker) {
    setDebugData(avoidance_data_, path_shifter_, debug_data_);
  } else {
    debug_marker_.markers.clear();
  }

  output.path = std::make_shared<PathWithLaneId>(avoidance_path.path);
  output.reference_path = getPreviousModuleOutput().reference_path;

  const size_t ego_idx = planner_data_->findEgoIndex(output.path->points);
  util::clipPathLength(*output.path, ego_idx, planner_data_->parameters);

  // Drivable area generation.
  generateExtendedDrivableArea(*output.path);

  DEBUG_PRINT("exit plan(): set prev output (back().lat = %f)", prev_output_.shift_length.back());

  updateRegisteredRTCStatus(avoidance_path.path);

  return output;
}

CandidateOutput AvoidanceModule::planCandidate() const
{
  const auto & data = avoidance_data_;

  CandidateOutput output;

  auto shifted_path = data.candidate_path;

  if (!data.safe_new_sl.empty()) {  // clip from shift start index for visualize
    clipByMinStartIdx(data.safe_new_sl, shifted_path.path);

    const auto sl = getNonStraightShiftLine(data.safe_new_sl);
    const auto sl_front = data.safe_new_sl.front();
    const auto sl_back = data.safe_new_sl.back();

    output.lateral_shift = getRelativeLengthFromPath(sl);
    output.start_distance_to_path_change = sl_front.start_longitudinal;
    output.finish_distance_to_path_change = sl_back.end_longitudinal;

    const uint16_t steering_factor_direction = std::invoke([&output]() {
      if (output.lateral_shift > 0.0) {
        return SteeringFactor::LEFT;
      }
      return SteeringFactor::RIGHT;
    });
    steering_factor_interface_ptr_->updateSteeringFactor(
      {sl_front.start, sl_back.end},
      {output.start_distance_to_path_change, output.finish_distance_to_path_change},
      SteeringFactor::AVOIDANCE_PATH_CHANGE, steering_factor_direction, SteeringFactor::APPROACHING,
      "");
  }

  const size_t ego_idx = planner_data_->findEgoIndex(shifted_path.path.points);
  util::clipPathLength(shifted_path.path, ego_idx, planner_data_->parameters);

  output.path_candidate = shifted_path.path;

  return output;
}

BehaviorModuleOutput AvoidanceModule::planWaitingApproval()
{
  // we can execute the plan() since it handles the approval appropriately.
  BehaviorModuleOutput out = plan();
#ifndef USE_OLD_ARCHITECTURE
  if (path_shifter_.getShiftLines().empty()) {
    out.turn_signal_info = getPreviousModuleOutput().turn_signal_info;
  }
#endif
  const auto candidate = planCandidate();
  constexpr double threshold_to_update_status = -1.0e-03;
  if (candidate.start_distance_to_path_change > threshold_to_update_status) {
    updateCandidateRTCStatus(candidate);
    waitApproval();
  } else {
    clearWaitingApproval();
    removeCandidateRTCStatus();
  }
  path_candidate_ = std::make_shared<PathWithLaneId>(candidate.path_candidate);
  path_reference_ = getPreviousModuleOutput().reference_path;
  return out;
}

void AvoidanceModule::addShiftLineIfApproved(const AvoidLineArray & shift_lines)
{
  if (isActivated()) {
    DEBUG_PRINT("We want to add this shift point, and approved. ADD SHIFT POINT!");
    const size_t prev_size = path_shifter_.getShiftLinesSize();
    addNewShiftLines(path_shifter_, shift_lines);

    current_raw_shift_lines_ = avoidance_data_.unapproved_raw_sl;

    // register original points for consistency
    registerRawShiftLines(shift_lines);

    const auto sl = getNonStraightShiftLine(shift_lines);
    const auto sl_front = shift_lines.front();
    const auto sl_back = shift_lines.back();

    if (getRelativeLengthFromPath(sl) > 0.0) {
      left_shift_array_.push_back({uuid_left_, sl_front.start, sl_back.end});
    } else if (getRelativeLengthFromPath(sl) < 0.0) {
      right_shift_array_.push_back({uuid_right_, sl_front.start, sl_back.end});
    }

    uuid_left_ = generateUUID();
    uuid_right_ = generateUUID();
    candidate_uuid_ = generateUUID();

    lockNewModuleLaunch();

    DEBUG_PRINT("shift_line size: %lu -> %lu", prev_size, path_shifter_.getShiftLinesSize());
  } else {
    DEBUG_PRINT("We want to add this shift point, but NOT approved. waiting...");
    waitApproval();
  }
}

/**
 * set new shift points. remove old shift points if it has a conflict.
 */
void AvoidanceModule::addNewShiftLines(
  PathShifter & path_shifter, const AvoidLineArray & new_shift_lines) const
{
  ShiftLineArray future = toShiftLineArray(new_shift_lines);

  size_t min_start_idx = std::numeric_limits<size_t>::max();
  for (const auto & sl : new_shift_lines) {
    min_start_idx = std::min(min_start_idx, sl.start_idx);
  }

  const auto current_shift_lines = path_shifter.getShiftLines();

  DEBUG_PRINT("min_start_idx = %lu", min_start_idx);

  // Remove shift points that starts later than the new_shift_line from path_shifter.
  //
  // Why? Because shifter sorts by start position and applies shift points, so if there is a
  // shift point that starts after the one you are going to put in, new one will be affected
  // by the old one.
  //
  // Is it ok? This is a situation where the vehicle was originally going to avoid at the farther
  // point, but decided to avoid it at a closer point. In this case, it is reasonable to cancel the
  // farther avoidance.
  for (const auto & sl : current_shift_lines) {
    if (sl.start_idx >= min_start_idx) {
      DEBUG_PRINT(
        "sl.start_idx = %lu, this sl starts after new proposal. remove this one.", sl.start_idx);
    } else {
      DEBUG_PRINT("sl.start_idx = %lu, no conflict. keep this one.", sl.start_idx);
      future.push_back(sl);
    }
  }

  path_shifter.setShiftLines(future);
}

AvoidLineArray AvoidanceModule::findNewShiftLine(
  const AvoidLineArray & candidates, const PathShifter & shifter) const
{
  (void)shifter;

  if (candidates.empty()) {
    DEBUG_PRINT("shift candidates is empty. return None.");
    return {};
  }

  printShiftLines(candidates, "findNewShiftLine: candidates");

  // Retrieve the subsequent linear shift point from the given index point.
  const auto getShiftLineWithSubsequentStraight = [this, &candidates](size_t i) {
    AvoidLineArray subsequent{candidates.at(i)};
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      const auto next_shift = candidates.at(j);
      if (std::abs(next_shift.getRelativeLength()) < 1.0e-2) {
        subsequent.push_back(next_shift);
        DEBUG_PRINT("j = %lu, relative shift is zero. add together.", j);
      } else {
        DEBUG_PRINT("j = %lu, relative shift is not zero = %f.", j, next_shift.getRelativeLength());
        break;
      }
    }
    return subsequent;
  };

  const auto calcJerk = [this](const auto & al) {
    return path_shifter_.calcJerkFromLatLonDistance(
      al.getRelativeLength(), al.getRelativeLongitudinal(), getSharpAvoidanceEgoSpeed());
  };

  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto & candidate = candidates.at(i);
    std::stringstream ss;
    ss << "i = " << i << ", id = " << candidate.id;
    const auto pfx = ss.str().c_str();

    if (prev_reference_.points.size() != prev_linear_shift_path_.path.points.size()) {
      throw std::logic_error("prev_reference_ and prev_linear_shift_path_ must have same size.");
    }

    // new shift points must exist in front of Ego
    // this value should be larger than -eps consider path shifter calculation error.
    const double eps = 0.01;
    if (candidate.start_longitudinal < -eps) {
      continue;
    }

    // TODO(Horibe): this code prohibits the changes on ego pose. Think later.
    // if (candidate.start_idx < avoidance_data_.ego_closest_path_index) {
    //   DEBUG_PRINT("%s, start_idx is behind ego. skip.", pfx);
    //   continue;
    // }

    const auto current_shift = prev_linear_shift_path_.shift_length.at(
      findNearestIndex(prev_reference_.points, candidate.end.position));

    // TODO(Horibe) test fails with this print. why?
    // DEBUG_PRINT("%s, shift current: %f, candidate: %f", pfx, current_shift,
    // candidate.end_shift_length);

    const auto new_point_threshold = parameters_->avoidance_execution_lateral_threshold;
    if (std::abs(candidate.end_shift_length - current_shift) > new_point_threshold) {
      if (calcJerk(candidate) > parameters_->max_lateral_jerk) {
        DEBUG_PRINT(
          "%s, Failed to find new shift: jerk limit over (%f).", pfx, calcJerk(candidate));
        break;
      }

      DEBUG_PRINT(
        "%s, New shift point is found!!! shift change: %f -> %f", pfx, current_shift,
        candidate.end_shift_length);
      return getShiftLineWithSubsequentStraight(i);
    }
  }

  DEBUG_PRINT("No new shift point exists.");
  return {};
}

Pose AvoidanceModule::getUnshiftedEgoPose(const ShiftedPath & prev_path) const
{
  const auto ego_pose = getEgoPose();

  if (prev_path.path.points.empty()) {
    return ego_pose;
  }

  // un-shifted fot current ideal pose
  const auto closest = findNearestIndex(prev_path.path.points, ego_pose.position);

  // NOTE: Considering avoidance by motion, we set unshifted_pose as previous path instead of
  // ego_pose.
  Pose unshifted_pose = motion_utils::calcInterpolatedPoint(prev_path.path, ego_pose).point.pose;

  util::shiftPose(&unshifted_pose, -prev_path.shift_length.at(closest));
  unshifted_pose.orientation = ego_pose.orientation;

  return unshifted_pose;
}

ShiftedPath AvoidanceModule::generateAvoidancePath(PathShifter & path_shifter) const
{
  DEBUG_PRINT("path_shifter: base shift = %f", getCurrentBaseShift());
  printShiftLines(path_shifter.getShiftLines(), "path_shifter shift points");

  ShiftedPath shifted_path;
  if (!path_shifter.generate(&shifted_path)) {
    RCLCPP_ERROR(getLogger(), "failed to generate shifted path.");
    return prev_output_;
  }

  return shifted_path;
}

void AvoidanceModule::updateData()
{
#ifndef USE_OLD_ARCHITECTURE
  // for the first time
  if (prev_output_.path.points.empty()) {
    prev_output_.path = *getPreviousModuleOutput().path;
    prev_output_.shift_length = std::vector<double>(prev_output_.path.points.size(), 0.0);
  }
  if (prev_linear_shift_path_.path.points.empty()) {
    prev_linear_shift_path_.path = *getPreviousModuleOutput().path;
    prev_linear_shift_path_.shift_length =
      std::vector<double>(prev_linear_shift_path_.path.points.size(), 0.0);
  }
  if (prev_reference_.points.empty()) {
    prev_reference_ = *getPreviousModuleOutput().path;
  }
#endif

  debug_data_ = DebugData();
  avoidance_data_ = calcAvoidancePlanningData(debug_data_);

  // TODO(Horibe): this is not tested yet, disable now.
  updateRegisteredObject(avoidance_data_.target_objects);
  compensateDetectionLost(avoidance_data_.target_objects, avoidance_data_.other_objects);

  std::sort(
    avoidance_data_.target_objects.begin(), avoidance_data_.target_objects.end(),
    [](auto a, auto b) { return a.longitudinal < b.longitudinal; });

  path_shifter_.setPath(avoidance_data_.reference_path);

  // update registered shift point for new reference path & remove past objects
  updateRegisteredRawShiftLines();

#ifdef USE_OLD_ARCHITECTURE
  // for the first time
  if (prev_output_.path.points.empty()) {
    prev_output_.path = avoidance_data_.reference_path;
    prev_output_.shift_length = std::vector<double>(prev_output_.path.points.size(), 0.0);
  }
  if (prev_linear_shift_path_.path.points.empty()) {
    prev_linear_shift_path_.path = avoidance_data_.reference_path;
    prev_linear_shift_path_.shift_length =
      std::vector<double>(prev_linear_shift_path_.path.points.size(), 0.0);
  }
  if (prev_reference_.points.empty()) {
    prev_reference_ = avoidance_data_.reference_path;
  }
#endif

  fillShiftLine(avoidance_data_, debug_data_);
}

/*
 * updateRegisteredObject
 *
 * Same object is observed this time -> update registered object with the new one.
 * Not observed -> increment the lost_count. if it exceeds the threshold, remove it.
 * How to check if it is same object?
 *   - it has same ID
 *   - it has different id, but sn object is found around similar position
 */
void AvoidanceModule::updateRegisteredObject(const ObjectDataArray & now_objects)
{
  const auto updateIfDetectedNow = [&now_objects, this](auto & registered_object) {
    const auto & n = now_objects;
    const auto r_id = registered_object.object.object_id;
    const auto same_id_obj = std::find_if(
      n.begin(), n.end(), [&r_id](const auto & o) { return o.object.object_id == r_id; });

    // same id object is detected. update registered.
    if (same_id_obj != n.end()) {
      registered_object = *same_id_obj;
      return true;
    }

    constexpr auto POS_THR = 1.5;
    const auto r_pos = registered_object.object.kinematics.initial_pose_with_covariance.pose;
    const auto similar_pos_obj = std::find_if(n.begin(), n.end(), [&](const auto & o) {
      return calcDistance2d(r_pos, o.object.kinematics.initial_pose_with_covariance.pose) < POS_THR;
    });

    // same id object is not detected, but object is found around registered. update registered.
    if (similar_pos_obj != n.end()) {
      registered_object = *similar_pos_obj;
      return true;
    }

    // Same ID nor similar position object does not found.
    return false;
  };

  // -- check registered_objects, remove if lost_count exceeds limit. --
  for (int i = static_cast<int>(registered_objects_.size()) - 1; i >= 0; --i) {
    auto & r = registered_objects_.at(i);
    const std::string s = getUuidStr(r);

    // registered object is not detected this time. lost count up.
    if (!updateIfDetectedNow(r)) {
      r.lost_time = (clock_->now() - r.last_seen).seconds();
    } else {
      r.last_seen = clock_->now();
      r.lost_time = 0.0;
    }

    // lost count exceeds threshold. remove object from register.
    if (r.lost_time > parameters_->object_last_seen_threshold) {
      registered_objects_.erase(registered_objects_.begin() + i);
    }
  }

  const auto isAlreadyRegistered = [this](const auto & n_id) {
    const auto & r = registered_objects_;
    return std::any_of(
      r.begin(), r.end(), [&n_id](const auto & o) { return o.object.object_id == n_id; });
  };

  // -- check now_objects, add it if it has new object id --
  for (const auto & now_obj : now_objects) {
    if (!isAlreadyRegistered(now_obj.object.object_id)) {
      registered_objects_.push_back(now_obj);
    }
  }
}

/*
 * CompensateDetectionLost
 *
 * add registered object if the now_objects does not contain the same object_id.
 *
 */
void AvoidanceModule::compensateDetectionLost(
  ObjectDataArray & now_objects, ObjectDataArray & other_objects) const
{
  const auto old_size = now_objects.size();  // for debug

  const auto isDetectedNow = [&](const auto & r_id) {
    const auto & n = now_objects;
    return std::any_of(
      n.begin(), n.end(), [&r_id](const auto & o) { return o.object.object_id == r_id; });
  };

  const auto isIgnoreObject = [&](const auto & r_id) {
    const auto & n = other_objects;
    return std::any_of(
      n.begin(), n.end(), [&r_id](const auto & o) { return o.object.object_id == r_id; });
  };

  for (const auto & registered : registered_objects_) {
    if (
      !isDetectedNow(registered.object.object_id) && !isIgnoreObject(registered.object.object_id)) {
      now_objects.push_back(registered);
    }
  }
  DEBUG_PRINT("object size: %lu -> %lu", old_size, now_objects.size());
}

void AvoidanceModule::onEntry()
{
  DEBUG_PRINT("AVOIDANCE onEntry. wait approval!");
  initVariables();
#ifdef USE_OLD_ARCHITECTURE
  current_state_ = ModuleStatus::SUCCESS;
#else
  current_state_ = ModuleStatus::IDLE;
#endif
}

void AvoidanceModule::onExit()
{
  DEBUG_PRINT("AVOIDANCE onExit");
  initVariables();
  current_state_ = ModuleStatus::SUCCESS;
  clearWaitingApproval();
  removeRTCStatus();
  unlockNewModuleLaunch();
  steering_factor_interface_ptr_->clearSteeringFactors();
}

void AvoidanceModule::initVariables()
{
  prev_output_ = ShiftedPath();
  prev_linear_shift_path_ = ShiftedPath();
  prev_reference_ = PathWithLaneId();
  path_shifter_ = PathShifter{};
  left_shift_array_.clear();
  right_shift_array_.clear();

  debug_data_ = DebugData();
  debug_marker_.markers.clear();
  resetPathCandidate();
  resetPathReference();
  registered_raw_shift_lines_ = {};
  current_raw_shift_lines_ = {};
  original_unique_id = 0;
  is_avoidance_maneuver_starts = false;
}

bool AvoidanceModule::isTargetObjectType(const PredictedObject & object) const
{
  using autoware_auto_perception_msgs::msg::ObjectClassification;
  const auto t = util::getHighestProbLabel(object.classification);
  const auto is_object_type =
    ((t == ObjectClassification::CAR && parameters_->avoid_car) ||
     (t == ObjectClassification::TRUCK && parameters_->avoid_truck) ||
     (t == ObjectClassification::BUS && parameters_->avoid_bus) ||
     (t == ObjectClassification::TRAILER && parameters_->avoid_trailer) ||
     (t == ObjectClassification::UNKNOWN && parameters_->avoid_unknown) ||
     (t == ObjectClassification::BICYCLE && parameters_->avoid_bicycle) ||
     (t == ObjectClassification::MOTORCYCLE && parameters_->avoid_motorcycle) ||
     (t == ObjectClassification::PEDESTRIAN && parameters_->avoid_pedestrian));
  return is_object_type;
}

TurnSignalInfo AvoidanceModule::calcTurnSignalInfo(const ShiftedPath & path) const
{
  const auto shift_lines = path_shifter_.getShiftLines();
  if (shift_lines.empty()) {
    return {};
  }

  const auto front_shift_line = shift_lines.front();
  const size_t start_idx = front_shift_line.start_idx;
  const size_t end_idx = front_shift_line.end_idx;

  const auto current_shift_length = getCurrentShift();
  const double start_shift_length = path.shift_length.at(start_idx);
  const double end_shift_length = path.shift_length.at(end_idx);
  const double segment_shift_length = end_shift_length - start_shift_length;

  const double turn_signal_shift_length_threshold =
    planner_data_->parameters.turn_signal_shift_length_threshold;
  const double turn_signal_search_time = planner_data_->parameters.turn_signal_search_time;
  const double turn_signal_minimum_search_distance =
    planner_data_->parameters.turn_signal_minimum_search_distance;

  // If shift length is shorter than the threshold, it does not need to turn on blinkers
  if (std::fabs(segment_shift_length) < turn_signal_shift_length_threshold) {
    return {};
  }

  // If the vehicle does not shift anymore, we turn off the blinker
  if (std::fabs(end_shift_length - current_shift_length) < 0.1) {
    return {};
  }

  // compute blinker start idx and end idx
  const size_t blinker_start_idx = [&]() {
    for (size_t idx = start_idx; idx <= end_idx; ++idx) {
      const double current_shift_length = path.shift_length.at(idx);
      if (current_shift_length > 0.1) {
        return idx;
      }
    }
    return start_idx;
  }();
  const size_t blinker_end_idx = end_idx;

  const auto blinker_start_pose = path.path.points.at(blinker_start_idx).point.pose;
  const auto blinker_end_pose = path.path.points.at(blinker_end_idx).point.pose;

  const double ego_vehicle_offset =
    planner_data_->parameters.vehicle_info.max_longitudinal_offset_m;
  const auto signal_prepare_distance =
    std::max(getEgoSpeed() * turn_signal_search_time, turn_signal_minimum_search_distance);
  const auto ego_front_to_shift_start =
    calcSignedArcLength(path.path.points, getEgoPosition(), blinker_start_pose.position) -
    ego_vehicle_offset;

  if (signal_prepare_distance < ego_front_to_shift_start) {
    return {};
  }

  bool turn_signal_on_swerving = planner_data_->parameters.turn_signal_on_swerving;

  TurnSignalInfo turn_signal_info{};
  if (turn_signal_on_swerving) {
    if (segment_shift_length > 0.0) {
      turn_signal_info.turn_signal.command = TurnIndicatorsCommand::ENABLE_LEFT;
    } else {
      turn_signal_info.turn_signal.command = TurnIndicatorsCommand::ENABLE_RIGHT;
    }
  } else {
    turn_signal_info.turn_signal.command = TurnIndicatorsCommand::DISABLE;
  }

  if (ego_front_to_shift_start > 0.0) {
    turn_signal_info.desired_start_point = planner_data_->self_odometry->pose.pose;
  } else {
    turn_signal_info.desired_start_point = blinker_start_pose;
  }
  turn_signal_info.desired_end_point = blinker_end_pose;
  turn_signal_info.required_start_point = blinker_start_pose;
  turn_signal_info.required_end_point = blinker_end_pose;

  return turn_signal_info;
}

void AvoidanceModule::setDebugData(
  const AvoidancePlanningData & data, const PathShifter & shifter, const DebugData & debug) const
{
  using marker_utils::createLaneletsAreaMarkerArray;
  using marker_utils::createObjectsMarkerArray;
  using marker_utils::createPathMarkerArray;
  using marker_utils::createPoseMarkerArray;
  using marker_utils::createShiftLengthMarkerArray;
  using marker_utils::createShiftLineMarkerArray;
  using marker_utils::avoidance_marker::createAvoidLineMarkerArray;
  using marker_utils::avoidance_marker::createEgoStatusMarkerArray;
  using marker_utils::avoidance_marker::createOtherObjectsMarkerArray;
  using marker_utils::avoidance_marker::createOverhangFurthestLineStringMarkerArray;
  using marker_utils::avoidance_marker::createPredictedVehiclePositions;
  using marker_utils::avoidance_marker::createSafetyCheckMarkerArray;
  using marker_utils::avoidance_marker::createTargetObjectsMarkerArray;
  using marker_utils::avoidance_marker::createUnavoidableObjectsMarkerArray;
  using marker_utils::avoidance_marker::createUnsafeObjectsMarkerArray;
  using marker_utils::avoidance_marker::makeOverhangToRoadShoulderMarkerArray;
  using motion_utils::createDeadLineVirtualWallMarker;
  using motion_utils::createSlowDownVirtualWallMarker;
  using motion_utils::createStopVirtualWallMarker;
  using tier4_autoware_utils::appendMarkerArray;
  using tier4_autoware_utils::calcOffsetPose;

  debug_marker_.markers.clear();
  const auto & base_link2front = planner_data_->parameters.base_link2front;
  const auto current_time = rclcpp::Clock{RCL_ROS_TIME}.now();

  const auto add = [this](const MarkerArray & added) {
    tier4_autoware_utils::appendMarkerArray(added, &debug_marker_);
  };

  const auto addAvoidLine =
    [&](const AvoidLineArray & al_arr, const auto & ns, auto r, auto g, auto b, double w = 0.1) {
      add(createAvoidLineMarkerArray(al_arr, ns, r, g, b, w));
    };

  const auto addShiftLine =
    [&](const ShiftLineArray & sl_arr, const auto & ns, auto r, auto g, auto b, double w = 0.1) {
      add(createShiftLineMarkerArray(sl_arr, shifter.getBaseOffset(), ns, r, g, b, w));
    };

  add(createEgoStatusMarkerArray(data, getEgoPose(), "ego_status"));
  add(createPredictedVehiclePositions(
    debug.path_with_planned_velocity, "predicted_vehicle_positions"));

  const auto & path = data.reference_path;
  add(createPathMarkerArray(debug.center_line, "centerline", 0, 0.0, 0.5, 0.9));
  add(createPathMarkerArray(path, "centerline_resampled", 0, 0.0, 0.9, 0.5));
  add(createPathMarkerArray(prev_linear_shift_path_.path, "prev_linear_shift", 0, 0.5, 0.4, 0.6));
  add(createPoseMarkerArray(data.reference_pose, "reference_pose", 0, 0.9, 0.3, 0.3));

  if (debug.stop_pose) {
    const auto p_front = calcOffsetPose(debug.stop_pose.get(), base_link2front, 0.0, 0.0);
    appendMarkerArray(
      createStopVirtualWallMarker(p_front, "avoidance stop", current_time, 0L), &debug_marker_);
  }

  if (debug.slow_pose) {
    const auto p_front = calcOffsetPose(debug.slow_pose.get(), base_link2front, 0.0, 0.0);
    appendMarkerArray(
      createSlowDownVirtualWallMarker(p_front, "avoidance slow", current_time, 0L), &debug_marker_);
  }

  if (debug.feasible_bound) {
    const auto p_front = calcOffsetPose(debug.feasible_bound.get(), base_link2front, 0.0, 0.0);
    appendMarkerArray(
      createDeadLineVirtualWallMarker(p_front, "feasible bound", current_time, 0L), &debug_marker_);
  }

  add(createSafetyCheckMarkerArray(data.state, getEgoPose(), debug));

  add(createLaneletsAreaMarkerArray(*debug.current_lanelets, "current_lanelet", 0.0, 1.0, 0.0));
  add(createLaneletsAreaMarkerArray(*debug.expanded_lanelets, "expanded_lanelet", 0.8, 0.8, 0.0));
  add(createTargetObjectsMarkerArray(data.target_objects, "target_objects"));
  add(createOtherObjectsMarkerArray(data.other_objects, "other_objects"));
  add(makeOverhangToRoadShoulderMarkerArray(data.target_objects, "overhang"));
  add(createOverhangFurthestLineStringMarkerArray(
    *debug.farthest_linestring_from_overhang, "farthest_linestring_from_overhang", 1.0, 0.0, 1.0));

  add(createUnavoidableObjectsMarkerArray(debug.unavoidable_objects, "unavoidable_objects"));
  add(createUnsafeObjectsMarkerArray(debug.unsafe_objects, "unsafe_objects"));

  // parent object info
  addAvoidLine(debug.registered_raw_shift, "p_registered_shift", 0.8, 0.8, 0.0);
  addAvoidLine(debug.current_raw_shift, "p_current_raw_shift", 0.5, 0.2, 0.2);
  addAvoidLine(debug.extra_return_shift, "p_extra_return_shift", 0.0, 0.5, 0.8);

  // merged shift
  const auto & linear_shift = prev_linear_shift_path_.shift_length;
  add(createShiftLengthMarkerArray(debug.pos_shift, path, "m_pos_shift_line", 0, 0.7, 0.5));
  add(createShiftLengthMarkerArray(debug.neg_shift, path, "m_neg_shift_line", 0, 0.5, 0.7));
  add(createShiftLengthMarkerArray(debug.total_shift, path, "m_total_shift_line", 0.99, 0.4, 0.2));
  add(createShiftLengthMarkerArray(debug.output_shift, path, "m_output_shift_line", 0.8, 0.8, 0.2));
  add(createShiftLengthMarkerArray(linear_shift, path, "m_output_linear_line", 0.9, 0.3, 0.3));

  // child shift points
  addAvoidLine(debug.merged, "c_0_merged", 0.345, 0.968, 1.0);
  addAvoidLine(debug.trim_similar_grad_shift, "c_1_trim_similar_grad_shift", 0.976, 0.328, 0.910);
  addAvoidLine(debug.quantized, "c_2_quantized", 0.505, 0.745, 0.969);
  addAvoidLine(debug.trim_small_shift, "c_3_trim_small_shift", 0.663, 0.525, 0.941);
  addAvoidLine(
    debug.trim_similar_grad_shift_second, "c_4_trim_similar_grad_shift", 0.97, 0.32, 0.91);
  addAvoidLine(debug.trim_momentary_return, "c_5_trim_momentary_return", 0.976, 0.078, 0.878);
  addAvoidLine(debug.trim_too_sharp_shift, "c_6_trim_too_sharp_shift", 0.576, 0.0, 0.978);

  addShiftLine(shifter.getShiftLines(), "path_shifter_registered_points", 0.99, 0.99, 0.0, 0.5);
  addAvoidLine(debug.new_shift_lines, "path_shifter_proposed_points", 0.99, 0.0, 0.0, 0.5);
}

void AvoidanceModule::updateAvoidanceDebugData(
  std::vector<AvoidanceDebugMsg> & avoidance_debug_msg_array) const
{
  debug_data_.avoidance_debug_msg_array.avoidance_info.clear();
  auto & debug_data_avoidance = debug_data_.avoidance_debug_msg_array.avoidance_info;
  debug_data_avoidance = avoidance_debug_msg_array;
  if (!debug_avoidance_initializer_for_shift_line_.empty()) {
    const bool is_info_old_ =
      (clock_->now() - debug_avoidance_initializer_for_shift_line_time_).seconds() > 0.1;
    if (!is_info_old_) {
      debug_data_avoidance.insert(
        debug_data_avoidance.end(), debug_avoidance_initializer_for_shift_line_.begin(),
        debug_avoidance_initializer_for_shift_line_.end());
    }
  }
}

double AvoidanceModule::getFeasibleDecelDistance(const double target_velocity) const
{
  const auto & a_now = planner_data_->self_acceleration->accel.accel.linear.x;
  const auto & a_lim = parameters_->max_deceleration;
  const auto & j_lim = parameters_->max_jerk;
  return calcDecelDistWithJerkAndAccConstraints(
    getEgoSpeed(), target_velocity, a_now, a_lim, j_lim, -1.0 * j_lim);
}

double AvoidanceModule::getMildDecelDistance(const double target_velocity) const
{
  const auto & a_now = planner_data_->self_acceleration->accel.accel.linear.x;
  const auto & a_lim = parameters_->nominal_deceleration;
  const auto & j_lim = parameters_->nominal_jerk;
  return calcDecelDistWithJerkAndAccConstraints(
    getEgoSpeed(), target_velocity, a_now, a_lim, j_lim, -1.0 * j_lim);
}

double AvoidanceModule::getRelativeLengthFromPath(const AvoidLine & avoid_line) const
{
  if (prev_reference_.points.size() != prev_linear_shift_path_.path.points.size()) {
    throw std::logic_error("prev_reference_ and prev_linear_shift_path_ must have same size.");
  }

  const auto current_shift_length = prev_linear_shift_path_.shift_length.at(
    findNearestIndex(prev_reference_.points, avoid_line.end.position));

  return avoid_line.end_shift_length - current_shift_length;
}

void AvoidanceModule::insertWaitPoint(
  const bool use_constraints_for_decel, ShiftedPath & shifted_path) const
{
  const auto & p = parameters_;
  const auto & data = avoidance_data_;
  const auto & base_link2front = planner_data_->parameters.base_link2front;
  const auto & vehicle_width = planner_data_->parameters.vehicle_width;

  if (!data.stop_target_object) {
    return;
  }

  if (data.avoiding_now) {
    return;
  }

  //         D5
  //      |<---->|                               D4
  //      |<----------------------------------------------------------------------->|
  // +-----------+            D1                 D2                      D3         +-----------+
  // |           |        |<------->|<------------------------->|<----------------->|           |
  // |    ego    |======= x ======= x ========================= x ==================|    obj    |
  // |           |    stop_point  avoid                       avoid                 |           |
  // +-----------+                start                        end                  +-----------+
  //
  // D1: p.min_prepare_distance
  // D2: min_avoid_distance
  // D3: longitudinal_avoid_margin_front (margin + D5)
  // D4: o_front.longitudinal
  // D5: base_link2front

  const auto o_front = data.stop_target_object.get();

  const auto avoid_margin =
    p->lateral_collision_safety_buffer + p->lateral_collision_margin + 0.5 * vehicle_width;
  const auto variable =
    getMinimumAvoidanceDistance(getShiftLength(o_front, isOnRight(o_front), avoid_margin));
  const auto constant =
    p->min_prepare_distance + p->longitudinal_collision_safety_buffer + base_link2front;
  const auto start_longitudinal =
    o_front.longitudinal -
    std::clamp(variable + constant, p->stop_min_distance, p->stop_max_distance);

  if (!use_constraints_for_decel) {
    insertDecelPoint(
      getEgoPosition(), start_longitudinal, 0.0, shifted_path.path, debug_data_.stop_pose);
    return;
  }

  const auto stop_distance = getMildDecelDistance(0.0);

  const auto insert_distance = std::max(start_longitudinal, stop_distance);

  insertDecelPoint(
    getEgoPosition(), insert_distance, 0.0, shifted_path.path, debug_data_.stop_pose);
}

void AvoidanceModule::insertYieldVelocity(ShiftedPath & shifted_path) const
{
  const auto & p = parameters_;
  const auto & data = avoidance_data_;

  if (data.target_objects.empty()) {
    return;
  }

  if (data.avoiding_now) {
    return;
  }

  const auto decel_distance = getMildDecelDistance(p->yield_velocity);

  insertDecelPoint(
    getEgoPosition(), decel_distance, p->yield_velocity, shifted_path.path, debug_data_.slow_pose);
}

void AvoidanceModule::insertPrepareVelocity(const bool avoidable, ShiftedPath & shifted_path) const
{
  const auto & data = avoidance_data_;

  if (data.target_objects.empty()) {
    return;
  }

  if (!!data.stop_target_object) {
    if (data.stop_target_object.get().reason != AvoidanceDebugFactor::TOO_LARGE_JERK) {
      return;
    }
  }

  if (data.avoiding_now) {
    return;
  }

  if (avoidable) {
    return;
  }

  const auto decel_distance = getFeasibleDecelDistance(0.0);

  insertDecelPoint(getEgoPosition(), decel_distance, 0.0, shifted_path.path, debug_data_.slow_pose);
}

std::shared_ptr<AvoidanceDebugMsgArray> AvoidanceModule::get_debug_msg_array() const
{
  debug_data_.avoidance_debug_msg_array.header.stamp = clock_->now();
  return std::make_shared<AvoidanceDebugMsgArray>(debug_data_.avoidance_debug_msg_array);
}

void AvoidanceModule::acceptVisitor(const std::shared_ptr<SceneModuleVisitor> & visitor) const
{
  if (visitor) {
    visitor->visitAvoidanceModule(this);
  }
}

void SceneModuleVisitor::visitAvoidanceModule(const AvoidanceModule * module) const
{
  avoidance_visitor_ = module->get_debug_msg_array();
}
}  // namespace behavior_path_planner

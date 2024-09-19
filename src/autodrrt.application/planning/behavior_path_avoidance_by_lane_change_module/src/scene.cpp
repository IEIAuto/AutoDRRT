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

#include "behavior_path_avoidance_by_lane_change_module/scene.hpp"

#include "behavior_path_avoidance_module/utils.hpp"
#include "behavior_path_planner_common/utils/drivable_area_expansion/static_drivable_area.hpp"
#include "behavior_path_planner_common/utils/path_safety_checker/objects_filtering.hpp"
#include "behavior_path_planner_common/utils/path_utils.hpp"
#include "behavior_path_planner_common/utils/utils.hpp"

#include <boost/geometry/algorithms/centroid.hpp>
#include <boost/geometry/strategies/cartesian/centroid_bashein_detmer.hpp>

#include <utility>

namespace behavior_path_planner
{
namespace
{
lanelet::BasicLineString3d toLineString3d(const std::vector<Point> & bound)
{
  lanelet::BasicLineString3d ret{};
  std::for_each(
    bound.begin(), bound.end(), [&](const auto & p) { ret.emplace_back(p.x, p.y, p.z); });
  return ret;
}
}  // namespace
AvoidanceByLaneChange::AvoidanceByLaneChange(
  const std::shared_ptr<LaneChangeParameters> & parameters,
  std::shared_ptr<AvoidanceByLCParameters> avoidance_parameters)
: NormalLaneChange(parameters, LaneChangeModuleType::AVOIDANCE_BY_LANE_CHANGE, Direction::NONE),
  avoidance_parameters_(std::move(avoidance_parameters))
{
}

bool AvoidanceByLaneChange::specialRequiredCheck() const
{
  const auto & data = avoidance_data_;

  if (!status_.is_safe) {
    return false;
  }

  const auto & object_parameters = avoidance_parameters_->object_parameters;
  const auto is_more_than_threshold =
    std::any_of(object_parameters.begin(), object_parameters.end(), [&](const auto & p) {
      const auto & objects = avoidance_data_.target_objects;

      const size_t num = std::count_if(objects.begin(), objects.end(), [&p](const auto & object) {
        const auto target_class =
          utils::getHighestProbLabel(object.object.classification) == p.first;
        return target_class && object.avoid_required;
      });

      return num >= p.second.execute_num;
    });

  if (!is_more_than_threshold) {
    return false;
  }

  const auto & front_object = data.target_objects.front();

  const auto THRESHOLD = avoidance_parameters_->execute_object_longitudinal_margin;
  if (front_object.longitudinal < THRESHOLD) {
    return false;
  }

  const auto path = status_.lane_change_path.path;
  const auto to_lc_end = motion_utils::calcSignedArcLength(
    status_.lane_change_path.path.points, getEgoPose().position,
    status_.lane_change_path.info.shift_line.end.position);
  const auto execute_only_when_lane_change_finish_before_object =
    avoidance_parameters_->execute_only_when_lane_change_finish_before_object;
  const auto not_enough_distance = front_object.longitudinal < to_lc_end;

  if (execute_only_when_lane_change_finish_before_object && not_enough_distance) {
    return false;
  }

  return true;
}

bool AvoidanceByLaneChange::specialExpiredCheck() const
{
  return !specialRequiredCheck();
}

void AvoidanceByLaneChange::updateSpecialData()
{
  const auto p = std::dynamic_pointer_cast<AvoidanceParameters>(avoidance_parameters_);

  avoidance_debug_data_ = DebugData();
  avoidance_data_ = calcAvoidancePlanningData(avoidance_debug_data_);

  if (avoidance_data_.target_objects.empty()) {
    direction_ = Direction::NONE;
  } else {
    direction_ = utils::avoidance::isOnRight(avoidance_data_.target_objects.front())
                   ? Direction::LEFT
                   : Direction::RIGHT;
  }

  utils::avoidance::updateRegisteredObject(registered_objects_, avoidance_data_.target_objects, p);
  utils::avoidance::compensateDetectionLost(
    registered_objects_, avoidance_data_.target_objects, avoidance_data_.other_objects);

  std::sort(
    avoidance_data_.target_objects.begin(), avoidance_data_.target_objects.end(),
    [](auto a, auto b) { return a.longitudinal < b.longitudinal; });
}

AvoidancePlanningData AvoidanceByLaneChange::calcAvoidancePlanningData(
  AvoidanceDebugData & debug) const
{
  AvoidancePlanningData data;

  // reference pose
  data.reference_pose = getEgoPose();

  data.reference_path_rough = prev_module_path_;

  const auto resample_interval = avoidance_parameters_->resample_interval_for_planning;
  data.reference_path = utils::resamplePathWithSpline(data.reference_path_rough, resample_interval);

  data.current_lanelets = getCurrentLanes();

  // expand drivable lanes
  std::for_each(
    data.current_lanelets.begin(), data.current_lanelets.end(), [&](const auto & lanelet) {
      data.drivable_lanes.push_back(utils::avoidance::generateExpandDrivableLanes(
        lanelet, planner_data_, avoidance_parameters_));
    });

  // calc drivable bound
  const auto shorten_lanes =
    utils::cutOverlappedLanes(data.reference_path_rough, data.drivable_lanes);
  data.left_bound = toLineString3d(utils::calcBound(
    planner_data_->route_handler, shorten_lanes, avoidance_parameters_->use_hatched_road_markings,
    avoidance_parameters_->use_intersection_areas, true));
  data.right_bound = toLineString3d(utils::calcBound(
    planner_data_->route_handler, shorten_lanes, avoidance_parameters_->use_hatched_road_markings,
    avoidance_parameters_->use_intersection_areas, false));

  // get related objects from dynamic_objects, and then separates them as target objects and non
  // target objects
  fillAvoidanceTargetObjects(data, debug);

  return data;
}

void AvoidanceByLaneChange::fillAvoidanceTargetObjects(
  AvoidancePlanningData & data, DebugData & debug) const
{
  const auto p = std::dynamic_pointer_cast<AvoidanceParameters>(avoidance_parameters_);

  const auto [object_within_target_lane, object_outside_target_lane] =
    utils::path_safety_checker::separateObjectsByLanelets(
      *planner_data_->dynamic_object, data.current_lanelets,
      utils::path_safety_checker::isPolygonOverlapLanelet);

  // Assume that the maximum allocation for data.other object is the sum of
  // objects_within_target_lane and object_outside_target_lane. The maximum allocation for
  // data.target_objects is equal to object_within_target_lane
  {
    const auto other_objects_size =
      object_within_target_lane.objects.size() + object_outside_target_lane.objects.size();
    data.other_objects.reserve(other_objects_size);
    data.target_objects.reserve(object_within_target_lane.objects.size());
  }

  {
    const auto & objects = object_outside_target_lane.objects;
    std::transform(
      objects.cbegin(), objects.cend(), std::back_inserter(data.other_objects),
      [](const auto & object) {
        ObjectData other_object;
        other_object.object = object;
        other_object.reason = "OutOfTargetArea";
        return other_object;
      });
  }

  ObjectDataArray target_lane_objects;
  target_lane_objects.reserve(object_within_target_lane.objects.size());
  {
    const auto & objects = object_within_target_lane.objects;
    std::transform(
      objects.cbegin(), objects.cend(), std::back_inserter(target_lane_objects),
      [&](const auto & object) { return createObjectData(data, object); });
  }

  utils::avoidance::filterTargetObjects(target_lane_objects, data, debug, planner_data_, p);
}

ObjectData AvoidanceByLaneChange::createObjectData(
  const AvoidancePlanningData & data, const PredictedObject & object) const
{
  using boost::geometry::return_centroid;
  using motion_utils::findNearestIndex;
  using tier4_autoware_utils::calcDistance2d;

  const auto p = std::dynamic_pointer_cast<AvoidanceParameters>(avoidance_parameters_);

  const auto & path_points = data.reference_path.points;
  const auto & object_pose = object.kinematics.initial_pose_with_covariance.pose;
  const auto object_closest_index = findNearestIndex(path_points, object_pose.position);
  const auto object_closest_pose = path_points.at(object_closest_index).point.pose;
  const auto t = utils::getHighestProbLabel(object.classification);
  const auto & object_parameter = avoidance_parameters_->object_parameters.at(t);

  ObjectData object_data{};

  object_data.object = object;

  const auto lower = p->lower_distance_for_polygon_expansion;
  const auto upper = p->upper_distance_for_polygon_expansion;
  const auto clamp =
    std::clamp(calcDistance2d(getEgoPose(), object_pose) - lower, 0.0, upper) / upper;
  object_data.distance_factor = object_parameter.max_expand_ratio * clamp + 1.0;

  // Calc envelop polygon.
  utils::avoidance::fillObjectEnvelopePolygon(
    object_data, registered_objects_, object_closest_pose, p);

  // calc object centroid.
  object_data.centroid = return_centroid<Point2d>(object_data.envelope_poly);

  // Calc moving time.
  utils::avoidance::fillObjectMovingTime(object_data, stopped_objects_, p);

  // Calc lateral deviation from path to target object.
  object_data.lateral =
    tier4_autoware_utils::calcLateralDeviation(object_closest_pose, object_pose.position);

  // Find the footprint point closest to the path, set to object_data.overhang_distance.
  object_data.overhang_dist = utils::avoidance::calcEnvelopeOverhangDistance(
    object_data, data.reference_path, object_data.overhang_pose.position);

  // Check whether the the ego should avoid the object.
  const auto & vehicle_width = planner_data_->parameters.vehicle_width;
  utils::avoidance::fillAvoidanceNecessity(object_data, registered_objects_, vehicle_width, p);

  return object_data;
}
}  // namespace behavior_path_planner

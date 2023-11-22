// Copyright 2020 Tier IV, Inc.
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

#include <interpolation/spline_interpolation.hpp>
#include <lanelet2_extension/regulatory_elements/road_marking.hpp>
#include <lanelet2_extension/utility/query.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <rclcpp/rclcpp.hpp>
#include <scene_module/intersection/util.hpp>
#include <utilization/path_utilization.hpp>
#include <utilization/util.hpp>

#include <lanelet2_core/geometry/Polygon.h>
#include <lanelet2_core/primitives/BasicRegulatoryElements.h>

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace behavior_velocity_planner
{
namespace bg = boost::geometry;

namespace util
{
std::optional<size_t> insertPoint(
  const geometry_msgs::msg::Pose & in_pose,
  autoware_auto_planning_msgs::msg::PathWithLaneId * inout_path)
{
  static constexpr double dist_thr = 10.0;
  static constexpr double angle_thr = M_PI / 1.5;
  const auto closest_idx_opt =
    motion_utils::findNearestIndex(inout_path->points, in_pose, dist_thr, angle_thr);
  if (!closest_idx_opt) {
    return -1;
  }
  const size_t closest_idx = closest_idx_opt.get();
  // vector.insert(i) inserts element on the left side of v[i]
  // the velocity need to be zero order hold(from prior point)
  int insert_idx = closest_idx;
  autoware_auto_planning_msgs::msg::PathPointWithLaneId inserted_point =
    inout_path->points.at(closest_idx);
  if (planning_utils::isAheadOf(in_pose, inout_path->points.at(closest_idx).point.pose)) {
    ++insert_idx;
  } else {
    // copy with velocity from prior point
    const size_t prior_ind = closest_idx > 0 ? closest_idx - 1 : 0;
    inserted_point = inout_path->points.at(prior_ind);
  }
  inserted_point.point.pose = in_pose;

  auto it = inout_path->points.begin() + insert_idx;
  inout_path->points.insert(it, inserted_point);

  return insert_idx;
}

bool hasLaneIds(
  const autoware_auto_planning_msgs::msg::PathPointWithLaneId & p, const std::set<int> & ids)
{
  for (const auto & pid : p.lane_ids) {
    if (ids.find(pid) != ids.end()) {
      return true;
    }
  }
  return false;
}

std::optional<std::pair<size_t, size_t>> findLaneIdsInterval(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & p, const std::set<int> & ids)
{
  bool found = false;
  size_t start = 0;
  size_t end = p.points.size() > 0 ? p.points.size() - 1 : 0;
  if (start == end) {
    // there is only one point in the path
    return std::nullopt;
  }
  for (size_t i = 0; i < p.points.size(); ++i) {
    if (hasLaneIds(p.points.at(i), ids)) {
      if (!found) {
        // found interval for the first time
        found = true;
        start = i;
      }
    } else if (found) {
      // prior point was in the interval. interval ended
      end = i;
      break;
    }
  }
  start = start > 0 ? start - 1 : 0;  // the idx of last point before the interval
  return found ? std::make_optional(std::make_pair(start, end)) : std::nullopt;
}

std::optional<size_t> getDuplicatedPointIdx(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path,
  const geometry_msgs::msg::Point & point)
{
  for (size_t i = 0; i < path.points.size(); i++) {
    const auto & p = path.points.at(i).point.pose.position;

    constexpr double min_dist = 0.001;
    if (tier4_autoware_utils::calcDistance2d(p, point) < min_dist) {
      return i;
    }
  }

  return std::nullopt;
}

std::optional<std::pair<size_t, lanelet::CompoundPolygon3d>> getFirstPointInsidePolygons(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path,
  const std::pair<size_t, size_t> lane_interval,
  const std::vector<lanelet::CompoundPolygon3d> & polygons)
{
  for (size_t i = lane_interval.first; i <= lane_interval.second; ++i) {
    bool is_in_lanelet = false;
    auto p = path.points.at(i).point.pose.position;
    for (const auto & polygon : polygons) {
      const auto polygon_2d = lanelet::utils::to2D(polygon);
      is_in_lanelet = bg::within(to_bg2d(p), polygon_2d);
      if (is_in_lanelet) {
        return std::make_optional<std::pair<size_t, lanelet::CompoundPolygon3d>>(i, polygon);
      }
    }
    if (is_in_lanelet) {
      break;
    }
  }
  return std::nullopt;
}

static std::optional<size_t> getFirstPointInsidePolygon(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path,
  const std::pair<size_t, size_t> lane_interval, const lanelet::CompoundPolygon3d & polygon)
{
  for (size_t i = lane_interval.first; i <= lane_interval.second; ++i) {
    bool is_in_lanelet = false;
    auto p = path.points.at(i).point.pose.position;
    const auto polygon_2d = lanelet::utils::to2D(polygon);
    is_in_lanelet = bg::within(to_bg2d(p), polygon_2d);
    if (is_in_lanelet) {
      return std::make_optional<size_t>(i);
    }
    if (is_in_lanelet) {
      break;
    }
  }
  return std::nullopt;
}

std::optional<StopLineIdx> generateStopLine(
  const int lane_id, const lanelet::CompoundPolygon3d & detection_area,
  const std::shared_ptr<const PlannerData> & planner_data, const double stop_line_margin,
  autoware_auto_planning_msgs::msg::PathWithLaneId * original_path,
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path_ip, const double interval,
  const std::pair<size_t, size_t> lane_interval_ip, const rclcpp::Logger logger)
{
  /* set judge line dist */
  const double current_vel = planner_data->current_velocity->twist.linear.x;
  const double current_acc = planner_data->current_acceleration->accel.accel.linear.x;
  const double max_acc = planner_data->max_stop_acceleration_threshold;
  const double max_jerk = planner_data->max_stop_jerk_threshold;
  const double delay_response_time = planner_data->delay_response_time;
  const double pass_judge_line_dist = planning_utils::calcJudgeLineDistWithJerkLimit(
    current_vel, current_acc, max_acc, max_jerk, delay_response_time);

  const int stop_line_margin_idx_dist = std::ceil(stop_line_margin / interval);
  const int base2front_idx_dist =
    std::ceil(planner_data->vehicle_info_.max_longitudinal_offset_m / interval);
  const int pass_judge_idx_dist = std::ceil(pass_judge_line_dist / interval);

  /* generate stop points */
  util::StopLineIdx idxs;

  // If a stop_line tag is defined on lanelet_map, use it.
  // else generate a stop_line behind the intersection of path and detection area
  // stop point index on interpolated(ip) path.
  size_t stop_idx_ip = 0;
  if (getStopLineIndexFromMap(
        path_ip, lane_interval_ip, lane_id, planner_data, &stop_idx_ip, 10.0, logger)) {
    stop_idx_ip =
      static_cast<size_t>(std::max<int>(static_cast<int>(stop_idx_ip) - base2front_idx_dist, 0));
  } else {
    // find the index of the first point that intersects with detection_areas
    const auto first_inside_detection_idx_ip_opt =
      getFirstPointInsidePolygon(path_ip, lane_interval_ip, detection_area);
    // if path is not intersecting with detection_area, skip
    if (!first_inside_detection_idx_ip_opt) {
      RCLCPP_DEBUG(
        logger, "Path is not intersecting with detection_area, not generating stop_line");
      return std::nullopt;
    }

    const auto first_inside_detection_idx_ip = first_inside_detection_idx_ip_opt.value();
    stop_idx_ip = static_cast<size_t>(std::max(
      static_cast<int>(first_inside_detection_idx_ip) - 1 - stop_line_margin_idx_dist -
        base2front_idx_dist,
      0));
  }
  if (stop_idx_ip == 0) {
    RCLCPP_DEBUG(logger, "stop line is at path[0], ignore planning\n===== plan end =====");
    return std::nullopt;
  }

  {
    /* insert stop_point */
    const auto & insert_point = path_ip.points.at(stop_idx_ip).point.pose;
    const auto duplicate_idx_opt =
      util::getDuplicatedPointIdx(*original_path, insert_point.position);
    if (duplicate_idx_opt) {
      idxs.collision_stop_line = duplicate_idx_opt.value();
    } else {
      const auto insert_idx_opt = util::insertPoint(insert_point, original_path);
      if (!insert_idx_opt) {
        RCLCPP_WARN(logger, "insertPoint failed for stop line");
        return std::nullopt;
      }
      idxs.collision_stop_line = insert_idx_opt.value();
    }
  }

  const bool has_prior_stopline = std::any_of(
    original_path->points.begin(), original_path->points.begin() + idxs.collision_stop_line,
    [](const auto & p) { return std::fabs(p.point.longitudinal_velocity_mps) < 0.1; });

  /* insert judge point */
  const size_t pass_judge_idx_ip = static_cast<size_t>(std::min(
    static_cast<int>(path_ip.points.size()) - 1,
    std::max<int>(static_cast<int>(stop_idx_ip) - pass_judge_idx_dist, 0)));
  /* if another stop point exist before intersection stop_line, disable judge_line. */
  if (has_prior_stopline || pass_judge_idx_ip == stop_idx_ip) {
    idxs.pass_judge_line = idxs.collision_stop_line;
  } else {
    const auto & insert_point = path_ip.points.at(pass_judge_idx_ip).point.pose;
    const auto duplicate_idx_opt =
      util::getDuplicatedPointIdx(*original_path, insert_point.position);
    if (duplicate_idx_opt) {
      idxs.pass_judge_line = duplicate_idx_opt.value();
    } else {
      const auto insert_idx_opt = util::insertPoint(insert_point, original_path);
      if (!insert_idx_opt) {
        RCLCPP_WARN(logger, "insertPoint failed to pass judge line");
        return std::nullopt;
      }
      idxs.pass_judge_line = insert_idx_opt.value();
      idxs.collision_stop_line =
        std::min<size_t>(idxs.collision_stop_line + 1, original_path->points.size() - 1);
    }
  }

  RCLCPP_DEBUG(
    logger,
    "generateStopLine() : stop_idx = %ld, pass_judge_idx = %ld"
    ", has_prior_stopline = %d",
    idxs.collision_stop_line, idxs.pass_judge_line, has_prior_stopline);

  return std::make_optional<StopLineIdx>(idxs);
}

std::optional<size_t> generateStuckStopLine(
  const lanelet::CompoundPolygon3d & conflicting_area,
  const std::shared_ptr<const PlannerData> & planner_data, const double stop_line_margin,
  const bool use_stuck_stopline, autoware_auto_planning_msgs::msg::PathWithLaneId * original_path,
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path_ip, const double interval,
  const std::pair<size_t, size_t> lane_interval_ip, const rclcpp::Logger logger)
{
  size_t stuck_stop_line_idx_ip = 0;
  const auto [lane_interval_ip_start, lane_interval_ip_end] = lane_interval_ip;
  if (use_stuck_stopline) {
    stuck_stop_line_idx_ip = lane_interval_ip_start;
  } else {
    const auto stuck_stop_line_idx_ip_opt =
      util::getFirstPointInsidePolygon(path_ip, lane_interval_ip, conflicting_area);
    if (!stuck_stop_line_idx_ip_opt) {
      RCLCPP_DEBUG(
        logger,
        "Path is not intersecting with conflicting area, not generating stuck_stop_line. start = "
        "%ld, end = %ld",
        lane_interval_ip_start, lane_interval_ip_end);
      return std::nullopt;
    }
    stuck_stop_line_idx_ip = stuck_stop_line_idx_ip_opt.value();
  }

  const int stop_line_margin_idx_dist = std::ceil(stop_line_margin / interval);
  const int base2front_idx_dist =
    std::ceil(planner_data->vehicle_info_.max_longitudinal_offset_m / interval);
  size_t stuck_stop_line_idx = 0;
  const size_t insert_idx_ip = static_cast<size_t>(std::max(
    static_cast<int>(stuck_stop_line_idx_ip) - 1 - stop_line_margin_idx_dist - base2front_idx_dist,
    0));
  const auto & insert_point = path_ip.points.at(insert_idx_ip).point.pose;
  const auto duplicate_idx_opt = util::getDuplicatedPointIdx(*original_path, insert_point.position);
  if (duplicate_idx_opt) {
    stuck_stop_line_idx = duplicate_idx_opt.value();
    return std::make_optional<size_t>(stuck_stop_line_idx);
  } else {
    const auto insert_idx_opt = util::insertPoint(insert_point, original_path);
    if (!insert_idx_opt) {
      RCLCPP_WARN(logger, "insertPoint failed for stuck stop line");
      return std::nullopt;
    }
    return std::make_optional<size_t>(insert_idx_opt.value());
  }
}

bool getStopLineIndexFromMap(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path,
  const std::pair<size_t, size_t> lane_interval, const int lane_id,
  const std::shared_ptr<const PlannerData> & planner_data, size_t * stop_idx_ip,
  const double dist_thr, const rclcpp::Logger logger)
{
  lanelet::ConstLanelet lanelet =
    planner_data->route_handler_->getLaneletMapPtr()->laneletLayer.get(lane_id);
  const auto road_markings = lanelet.regulatoryElementsAs<lanelet::autoware::RoadMarking>();
  lanelet::ConstLineStrings3d stop_line;
  for (const auto & road_marking : road_markings) {
    const std::string type =
      road_marking->roadMarking().attributeOr(lanelet::AttributeName::Type, "none");
    if (type == lanelet::AttributeValueString::StopLine) {
      stop_line.push_back(road_marking->roadMarking());
      break;  // only one stop_line exists.
    }
  }
  if (stop_line.empty()) {
    return false;
  }

  const auto p_start = stop_line.front().front();
  const auto p_end = stop_line.front().back();
  const LineString2d extended_stop_line =
    planning_utils::extendLine(p_start, p_end, planner_data->stop_line_extend_length);

  for (size_t i = lane_interval.first; i < lane_interval.second; i++) {
    const auto & p_front = path.points.at(i).point.pose.position;
    const auto & p_back = path.points.at(i + 1).point.pose.position;

    const LineString2d path_segment = {{p_front.x, p_front.y}, {p_back.x, p_back.y}};
    std::vector<Point2d> collision_points;
    bg::intersection(extended_stop_line, path_segment, collision_points);

    if (collision_points.empty()) {
      continue;
    }

    *stop_idx_ip = i;

    RCLCPP_DEBUG(logger, "found collision point");

    return true;
  }

  geometry_msgs::msg::Pose stop_point_from_map;
  stop_point_from_map.position.x = 0.5 * (p_start.x() + p_end.x());
  stop_point_from_map.position.y = 0.5 * (p_start.y() + p_end.y());
  stop_point_from_map.position.z = 0.5 * (p_start.z() + p_end.z());

  const auto stop_idx_ip_opt =
    motion_utils::findNearestIndex(path.points, stop_point_from_map, static_cast<double>(dist_thr));
  if (!stop_idx_ip_opt) {
    RCLCPP_DEBUG(logger, "found stop line, but not found stop index");
    return false;
  }
  *stop_idx_ip = stop_idx_ip_opt.get();

  RCLCPP_DEBUG(logger, "found stop line and stop index");

  return true;
}

IntersectionLanelets getObjectiveLanelets(
  lanelet::LaneletMapConstPtr lanelet_map_ptr, lanelet::routing::RoutingGraphPtr routing_graph_ptr,
  const int lane_id, const std::set<int> & assoc_ids,
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path,
  const std::pair<size_t, size_t> lane_interval, const double detection_area_length,
  const bool tl_arrow_solid_on)
{
  const auto & assigned_lanelet = lanelet_map_ptr->laneletLayer.get(lane_id);
  const auto turn_direction = assigned_lanelet.attributeOr("turn_direction", "else");

  // retrieve a stopline associated with a traffic light
  bool has_traffic_light = false;
  if (const auto tl_reg_elems = assigned_lanelet.regulatoryElementsAs<lanelet::TrafficLight>();
      tl_reg_elems.size() != 0) {
    const auto tl_reg_elem = tl_reg_elems.front();
    const auto stop_line_opt = tl_reg_elem->stopLine();
    if (!!stop_line_opt) has_traffic_light = true;
  }

  // for low priority lane
  // If ego_lane has right of way (i.e. is high priority),
  // ignore yieldLanelets (i.e. low priority lanes)
  lanelet::ConstLanelets yield_lanelets{};
  const auto right_of_ways = assigned_lanelet.regulatoryElementsAs<lanelet::RightOfWay>();
  for (const auto & right_of_way : right_of_ways) {
    if (lanelet::utils::contains(right_of_way->rightOfWayLanelets(), assigned_lanelet)) {
      for (const auto & yield_lanelet : right_of_way->yieldLanelets()) {
        yield_lanelets.push_back(yield_lanelet);
        for (const auto & previous_lanelet : routing_graph_ptr->previous(yield_lanelet)) {
          yield_lanelets.push_back(previous_lanelet);
        }
      }
    }
  }

  // get all following lanes of previous lane
  lanelet::ConstLanelets ego_lanelets{};
  for (const auto & previous_lanelet : routing_graph_ptr->previous(assigned_lanelet)) {
    ego_lanelets.push_back(previous_lanelet);
    for (const auto & following_lanelet : routing_graph_ptr->following(previous_lanelet)) {
      if (lanelet::utils::contains(ego_lanelets, following_lanelet)) {
        continue;
      }
      ego_lanelets.push_back(following_lanelet);
    }
  }

  // get conflicting lanes on assigned lanelet
  const auto & conflicting_lanelets =
    lanelet::utils::getConflictingLanelets(routing_graph_ptr, assigned_lanelet);

  // final objective lanelets
  lanelet::ConstLanelets detection_lanelets;
  lanelet::ConstLanelets conflicting_ex_ego_lanelets;
  // conflicting lanes is necessary to get stop_line for stuck vehicle
  for (auto && conflicting_lanelet : conflicting_lanelets) {
    if (!lanelet::utils::contains(ego_lanelets, conflicting_lanelet))
      conflicting_ex_ego_lanelets.push_back(conflicting_lanelet);
  }

  // exclude yield lanelets and ego lanelets from detection_lanelets
  if (turn_direction == std::string("straight") && has_traffic_light) {
    // if assigned lanelet is "straight" with traffic light, detection area is not necessary
  } else {
    // otherwise we need to know the priority from RightOfWay
    for (const auto & conflicting_lanelet : conflicting_lanelets) {
      if (
        lanelet::utils::contains(yield_lanelets, conflicting_lanelet) ||
        lanelet::utils::contains(ego_lanelets, conflicting_lanelet)) {
        continue;
      }
      detection_lanelets.push_back(conflicting_lanelet);
    }
  }

  // get possible lanelet path that reaches conflicting_lane longer than given length
  // if traffic light arrow is active, this process is unnecessary
  lanelet::ConstLanelets detection_and_preceding_lanelets;
  if (!tl_arrow_solid_on) {
    const double length = detection_area_length;
    std::set<lanelet::Id> detection_ids;
    for (const auto & ll : detection_lanelets) {
      // Preceding lanes does not include detection_lane so add them at the end
      const auto & inserted = detection_ids.insert(ll.id());
      if (inserted.second) detection_and_preceding_lanelets.push_back(ll);
      // get preceding lanelets without ego_lanelets
      // to prevent the detection area from including the ego lanes and its' preceding lanes.
      const auto lanelet_sequences = lanelet::utils::query::getPrecedingLaneletSequences(
        routing_graph_ptr, ll, length, ego_lanelets);
      for (const auto & ls : lanelet_sequences) {
        for (const auto & l : ls) {
          const auto & inserted = detection_ids.insert(l.id());
          if (inserted.second) detection_and_preceding_lanelets.push_back(l);
        }
      }
    }
  }

  IntersectionLanelets result;
  if (!tl_arrow_solid_on) {
    result.attention = std::move(detection_and_preceding_lanelets);
  } else {
    result.attention = std::move(detection_lanelets);
  }
  result.conflicting = std::move(conflicting_ex_ego_lanelets);
  result.adjacent = planning_utils::getConstLaneletsFromIds(lanelet_map_ptr, assoc_ids);
  // compoundPolygon3d
  result.attention_area = getPolygon3dFromLanelets(result.attention);
  result.conflicting_area = getPolygon3dFromLanelets(result.conflicting);
  result.adjacent_area = getPolygon3dFromLanelets(result.adjacent);

  // find the first conflicting/detection area polygon intersecting the path
  {
    auto first = util::getFirstPointInsidePolygons(path, lane_interval, result.conflicting_area);
    if (first) {
      result.first_conflicting_area = first.value().second;
    }
  }
  {
    auto first = util::getFirstPointInsidePolygons(path, lane_interval, result.attention_area);
    if (first) {
      result.first_detection_area = first.value().second;
    }
  }

  return result;
}

std::vector<lanelet::CompoundPolygon3d> getPolygon3dFromLanelets(
  const lanelet::ConstLanelets & ll_vec, double clip_length)
{
  std::vector<lanelet::CompoundPolygon3d> p_vec;
  for (const auto & ll : ll_vec) {
    const double path_length = lanelet::utils::getLaneletLength3d({ll});
    const auto polygon3d =
      lanelet::utils::getPolygonFromArcLength({ll}, path_length - clip_length, path_length);
    p_vec.push_back(polygon3d);
  }
  return p_vec;
}

std::vector<lanelet::CompoundPolygon3d> getPolygon3dFromLanelets(
  const lanelet::ConstLanelets & ll_vec)
{
  std::vector<lanelet::CompoundPolygon3d> polys;
  for (auto && ll : ll_vec) {
    polys.push_back(ll.polygon3d());
  }
  return polys;
}

std::vector<int> getLaneletIdsFromLanelets(lanelet::ConstLanelets ll)
{
  std::vector<int> id_list;
  for (const auto & l : ll) id_list.push_back(l.id());
  return id_list;
}

geometry_msgs::msg::Pose toPose(const geometry_msgs::msg::Point & p)
{
  geometry_msgs::msg::Pose pose;
  pose.position = p;
  return pose;
}

static std::string getTurnDirection(lanelet::ConstLanelet lane)
{
  return lane.attributeOr("turn_direction", "else");
}

bool isOverTargetIndex(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path, const int closest_idx,
  const geometry_msgs::msg::Pose & current_pose, const int target_idx)
{
  if (closest_idx == target_idx) {
    const geometry_msgs::msg::Pose target_pose = path.points.at(target_idx).point.pose;
    return planning_utils::isAheadOf(current_pose, target_pose);
  }
  return static_cast<bool>(closest_idx > target_idx);
}

bool isBeforeTargetIndex(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path, const int closest_idx,
  const geometry_msgs::msg::Pose & current_pose, const int target_idx)
{
  if (closest_idx == target_idx) {
    const geometry_msgs::msg::Pose target_pose = path.points.at(target_idx).point.pose;
    return planning_utils::isAheadOf(target_pose, current_pose);
  }
  return static_cast<bool>(target_idx > closest_idx);
}

static std::vector<int> getAllAdjacentLanelets(
  const lanelet::routing::RoutingGraphPtr routing_graph, lanelet::ConstLanelet lane)
{
  std::set<int> results;

  results.insert(lane.id());

  auto it = routing_graph->adjacentRight(lane);
  // take all lane on the right side
  while (!!it) {
    results.insert(it.get().id());
    it = routing_graph->adjacentRight(it.get());
  }
  // take all lane on the left side
  it = routing_graph->adjacentLeft(lane);
  while (!!it) {
    results.insert(it.get().id());
    it = routing_graph->adjacentLeft(it.get());
  }

  return std::vector<int>(results.begin(), results.end());
}

lanelet::ConstLanelets extendedAdjacentDirectionLanes(
  lanelet::LaneletMapConstPtr map, const lanelet::routing::RoutingGraphPtr routing_graph,
  lanelet::ConstLanelet lane)
{
  // some of the intersections are not well-formed, and "adjacent" turning
  // lanelets are not sharing the LineStrings
  const std::string turn_direction = getTurnDirection(lane);
  if (turn_direction != "left" && turn_direction != "right" && turn_direction != "straight")
    return {};

  std::set<int> previous_lanelet_ids;
  for (auto && previous_lanelet : routing_graph->previous(lane)) {
    previous_lanelet_ids.insert(previous_lanelet.id());
  }

  std::set<int> besides_previous_lanelet_ids;
  for (auto && previous_lanelet_id : previous_lanelet_ids) {
    lanelet::ConstLanelet previous_lanelet = map->laneletLayer.get(previous_lanelet_id);
    for (auto && beside_lanelet : getAllAdjacentLanelets(routing_graph, previous_lanelet)) {
      besides_previous_lanelet_ids.insert(beside_lanelet);
    }
  }

  std::set<int> following_turning_lanelets;
  following_turning_lanelets.insert(lane.id());
  for (auto && besides_previous_lanelet_id : besides_previous_lanelet_ids) {
    lanelet::ConstLanelet besides_previous_lanelet =
      map->laneletLayer.get(besides_previous_lanelet_id);
    for (auto && following_lanelet : routing_graph->following(besides_previous_lanelet)) {
      // if this has {"turn_direction", "${turn_direction}"}, take this
      if (getTurnDirection(following_lanelet) == turn_direction)
        following_turning_lanelets.insert(following_lanelet.id());
    }
  }
  lanelet::ConstLanelets ret{};
  for (auto && id : following_turning_lanelets) {
    ret.push_back(map->laneletLayer.get(id));
  }
  return ret;
}

std::optional<Polygon2d> getIntersectionArea(
  lanelet::ConstLanelet assigned_lane, lanelet::LaneletMapConstPtr lanelet_map_ptr)
{
  const std::string area_id_str = assigned_lane.attributeOr("intersection_area", "else");
  if (area_id_str == "else") return std::nullopt;

  const int area_id = std::atoi(area_id_str.c_str());
  const auto poly_3d = lanelet_map_ptr->polygonLayer.get(area_id);
  Polygon2d poly{};
  for (const auto & p : poly_3d) poly.outer().emplace_back(p.x(), p.y());
  return std::make_optional(poly);
}

bool hasAssociatedTrafficLight(lanelet::ConstLanelet lane)
{
  std::optional<int> tl_id = std::nullopt;
  for (auto && tl_reg_elem : lane.regulatoryElementsAs<lanelet::TrafficLight>()) {
    tl_id = tl_reg_elem->id();
    break;
  }
  return tl_id.has_value();
}

bool isTrafficLightArrowActivated(
  lanelet::ConstLanelet lane,
  const std::map<int, autoware_auto_perception_msgs::msg::TrafficSignalStamped> & tl_infos)
{
  const auto & turn_direction = lane.attributeOr("turn_direction", "else");
  std::optional<int> tl_id = std::nullopt;
  for (auto && tl_reg_elem : lane.regulatoryElementsAs<lanelet::TrafficLight>()) {
    tl_id = tl_reg_elem->id();
    break;
  }
  if (!tl_id) {
    // this lane has no traffic light
    return false;
  }
  const auto tl_info_it = tl_infos.find(tl_id.value());
  if (tl_info_it == tl_infos.end()) {
    // the info of this traffic light is not available
    return false;
  }
  const auto & tl_info = tl_info_it->second;
  for (auto && tl_light : tl_info.signal.lights) {
    if (tl_light.color != autoware_auto_perception_msgs::msg::TrafficLight::GREEN) continue;
    if (tl_light.status != autoware_auto_perception_msgs::msg::TrafficLight::SOLID_ON) continue;
    if (
      turn_direction == std::string("left") &&
      tl_light.shape == autoware_auto_perception_msgs::msg::TrafficLight::LEFT_ARROW)
      return true;
    if (
      turn_direction == std::string("right") &&
      tl_light.shape == autoware_auto_perception_msgs::msg::TrafficLight::RIGHT_ARROW)
      return true;
  }
  return false;
}

}  // namespace util
}  // namespace behavior_velocity_planner

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

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <behavior_velocity_planner_common/planner_data.hpp>
#include <behavior_velocity_planner_common/utilization/util.hpp>
#include <vehicle_info_util/vehicle_info_util.hpp>

#include <autoware_auto_perception_msgs/msg/shape.hpp>
#include <autoware_auto_planning_msgs/msg/path_point.hpp>
#include <tier4_debug_msgs/msg/float32_stamped.hpp>

#include <string>
#include <vector>

namespace behavior_velocity_planner
{
namespace run_out_utils
{
namespace bg = boost::geometry;
using autoware_auto_perception_msgs::msg::ObjectClassification;
using autoware_auto_perception_msgs::msg::PredictedObjects;
using autoware_auto_perception_msgs::msg::Shape;
using autoware_auto_planning_msgs::msg::PathPoint;
using autoware_auto_planning_msgs::msg::PathWithLaneId;
using tier4_autoware_utils::Box2d;
using tier4_autoware_utils::LineString2d;
using tier4_autoware_utils::Point2d;
using tier4_autoware_utils::Polygon2d;
using tier4_debug_msgs::msg::Float32Stamped;
using vehicle_info_util::VehicleInfo;
using PathPointsWithLaneId = std::vector<autoware_auto_planning_msgs::msg::PathPointWithLaneId>;
struct CommonParam
{
  double normal_min_jerk;  // min jerk limit for mild stop [m/sss]
  double normal_min_acc;   // min deceleration limit for mild stop [m/ss]
  double limit_min_jerk;   // min jerk limit [m/sss]
  double limit_min_acc;    // min deceleration limit [m/ss]
};
struct RunOutParam
{
  std::string detection_method;
  bool use_partition_lanelet;
  bool specify_decel_jerk;
  double stop_margin;
  double passing_margin;
  double deceleration_jerk;
  float detection_distance;
  float detection_span;
  float min_vel_ego_kmph;
};

struct VehicleParam
{
  float base_to_front;
  float base_to_rear;
  float width;
  float wheel_tread;
  double right_overhang;
  double left_overhang;
};

struct DetectionArea
{
  float margin_ahead;
  float margin_behind;
};

struct MandatoryArea
{
  float decel_jerk;
};

struct ApproachingParam
{
  bool enable;
  float margin;
  float limit_vel_kmph;
};

struct StateParam
{
  float stop_thresh;
  float stop_time_thresh;
  float disable_approach_dist;
  float keep_approach_duration;
};

struct SlowDownLimit
{
  bool enable;
  float max_jerk;
  float max_acc;
};

struct Smoother
{
  double start_jerk;
};

struct DynamicObstacleParam
{
  bool use_mandatory_area;
  bool assume_fixed_velocity;

  float min_vel_kmph;
  float max_vel_kmph;

  // parameter to convert points to dynamic obstacle
  float std_dev_multiplier;
  float diameter;             // [m]
  float height;               // [m]
  float max_prediction_time;  // [sec]
  float time_step;            // [sec]
  float points_interval;      // [m]
};

struct IgnoreMomentaryDetection
{
  bool enable;
  double time_threshold;
};

struct PlannerParam
{
  CommonParam common;
  RunOutParam run_out;
  VehicleParam vehicle_param;
  DetectionArea detection_area;
  MandatoryArea mandatory_area;
  ApproachingParam approaching;
  StateParam state_param;
  DynamicObstacleParam dynamic_obstacle;
  SlowDownLimit slow_down_limit;
  Smoother smoother;
  IgnoreMomentaryDetection ignore_momentary_detection;
};

enum class DetectionMethod {
  Object = 0,
  ObjectWithoutPath,
  Points,
  Unknown,
};

struct PoseWithRange
{
  geometry_msgs::msg::Pose pose_min;
  geometry_msgs::msg::Pose pose_max;
};

// since we use the minimum and maximum velocity,
// define the PredictedPath without time_step
struct PredictedPath
{
  std::vector<geometry_msgs::msg::Pose> path;
  float confidence;
};

// abstracted obstacle information
struct DynamicObstacle
{
  geometry_msgs::msg::Pose pose;
  std::vector<geometry_msgs::msg::Point> collision_points;
  geometry_msgs::msg::Point nearest_collision_point;
  float min_velocity_mps;
  float max_velocity_mps;
  std::vector<ObjectClassification> classifications;
  Shape shape;
  std::vector<PredictedPath> predicted_paths;
};

struct DynamicObstacleData
{
  PredictedObjects predicted_objects;
  PathWithLaneId path;
  Polygons2d detection_area;
  Polygons2d mandatory_detection_area;
};

Polygon2d createBoostPolyFromMsg(const std::vector<geometry_msgs::msg::Point> & input_poly);

std::uint8_t getHighestProbLabel(const std::vector<ObjectClassification> & classification);

std::vector<geometry_msgs::msg::Pose> getHighestConfidencePath(
  const std::vector<PredictedPath> & predicted_paths);

// apply linear interpolation to position
geometry_msgs::msg::Pose lerpByPose(
  const geometry_msgs::msg::Pose & p1, const geometry_msgs::msg::Pose & p2, const float t);

std::vector<geometry_msgs::msg::Point> findLateralSameSidePoints(
  const std::vector<geometry_msgs::msg::Point> & points, const geometry_msgs::msg::Pose & base_pose,
  const geometry_msgs::msg::Point & target_point);

bool isSamePoint(const geometry_msgs::msg::Point & p1, const geometry_msgs::msg::Point & p2);

// if path points have the same point as target_point, return the index
std::optional<size_t> haveSamePoint(
  const PathPointsWithLaneId & path_points, const geometry_msgs::msg::Point & target_point);

// insert path velocity which doesn't exceed original velocity
void insertPathVelocityFromIndexLimited(
  const size_t & start_idx, const float velocity_mps, PathPointsWithLaneId & path_points);

void insertPathVelocityFromIndex(
  const size_t & start_idx, const float velocity_mps, PathPointsWithLaneId & path_points);

std::optional<size_t> findFirstStopPointIdx(PathPointsWithLaneId & path_points);

LineString2d createLineString2d(const lanelet::BasicPolygon2d & poly);

std::vector<DynamicObstacle> excludeObstaclesOutSideOfLine(
  const std::vector<DynamicObstacle> & dynamic_obstacles, const PathPointsWithLaneId & path_points,
  const lanelet::BasicPolygon2d & partition);

PathPointsWithLaneId decimatePathPoints(
  const PathPointsWithLaneId & input_path_points, const float step);

// trim path from self_pose to trim_distance
PathWithLaneId trimPathFromSelfPose(
  const PathWithLaneId & input, const geometry_msgs::msg::Pose & self_pose,
  const double trim_distance);

// create polygon for passing lines and deceleration line calculated by stopping jerk
// note that this polygon is not closed
std::optional<std::vector<geometry_msgs::msg::Point>> createDetectionAreaPolygon(
  const std::vector<std::vector<geometry_msgs::msg::Point>> & passing_lines,
  const size_t deceleration_line_idx);

// extend path to the pose of goal
PathWithLaneId extendPath(const PathWithLaneId & input, const double extend_distance);
PathPoint createExtendPathPoint(const double extend_distance, const PathPoint & goal_point);

DetectionMethod toEnum(const std::string & detection_method);

Polygons2d createDetectionAreaPolygon(
  const PathWithLaneId & path, const PlannerData & planner_data,
  const PlannerParam & planner_param);

Polygons2d createMandatoryDetectionAreaPolygon(
  const PathWithLaneId & path, const PlannerData & planner_data,
  const PlannerParam & planner_param);
}  // namespace run_out_utils
}  // namespace behavior_velocity_planner
#endif  // UTILS_HPP_

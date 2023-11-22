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

#include "obstacle_avoidance_planner/node.hpp"

#include "interpolation/spline_interpolation_points_2d.hpp"
#include "obstacle_avoidance_planner/debug_marker.hpp"
#include "obstacle_avoidance_planner/utils/geometry_utils.hpp"
#include "obstacle_avoidance_planner/utils/trajectory_utils.hpp"
#include "rclcpp/time.hpp"

#include <chrono>
#include <limits>

namespace obstacle_avoidance_planner
{
namespace
{
template <class T>
std::vector<T> concatVectors(const std::vector<T> & prev_vector, const std::vector<T> & next_vector)
{
  std::vector<T> concatenated_vector;
  concatenated_vector.insert(concatenated_vector.end(), prev_vector.begin(), prev_vector.end());
  concatenated_vector.insert(concatenated_vector.end(), next_vector.begin(), next_vector.end());
  return concatenated_vector;
}

StringStamped createStringStamped(const rclcpp::Time & now, const std::string & data)
{
  StringStamped msg;
  msg.stamp = now;
  msg.data = data;
  return msg;
}

void setZeroVelocityAfterStopPoint(std::vector<TrajectoryPoint> & traj_points)
{
  const auto opt_zero_vel_idx = motion_utils::searchZeroVelocityIndex(traj_points);
  if (opt_zero_vel_idx) {
    for (size_t i = opt_zero_vel_idx.get(); i < traj_points.size(); ++i) {
      traj_points.at(i).longitudinal_velocity_mps = 0.0;
    }
  }
}

bool hasZeroVelocity(const TrajectoryPoint & traj_point)
{
  constexpr double zero_vel = 0.0001;
  return std::abs(traj_point.longitudinal_velocity_mps) < zero_vel;
}
}  // namespace

ObstacleAvoidancePlanner::ObstacleAvoidancePlanner(const rclcpp::NodeOptions & node_options)
: Node("obstacle_avoidance_planner", node_options),
  vehicle_info_(vehicle_info_util::VehicleInfoUtil(*this).getVehicleInfo()),
  debug_data_ptr_(std::make_shared<DebugData>()),
  time_keeper_ptr_(std::make_shared<TimeKeeper>())
{
  // interface publisher
  traj_pub_ = create_publisher<Trajectory>("~/output/path", 1);
  virtual_wall_pub_ = create_publisher<MarkerArray>("~/virtual_wall", 1);

  // interface subscriber
  path_sub_ = create_subscription<Path>(
    "~/input/path", 1, std::bind(&ObstacleAvoidancePlanner::onPath, this, std::placeholders::_1));
  odom_sub_ = create_subscription<Odometry>(
    "~/input/odometry", 1, [this](const Odometry::SharedPtr msg) { ego_state_ptr_ = msg; });

  // debug publisher
  debug_extended_traj_pub_ = create_publisher<Trajectory>("~/debug/extended_traj", 1);
  debug_markers_pub_ = create_publisher<MarkerArray>("~/debug/marker", 1);
  debug_calculation_time_pub_ = create_publisher<StringStamped>("~/debug/calculation_time", 1);

  {  // parameters
    // parameter for option
    enable_outside_drivable_area_stop_ =
      declare_parameter<bool>("option.enable_outside_drivable_area_stop");
    enable_smoothing_ = declare_parameter<bool>("option.enable_smoothing");
    enable_skip_optimization_ = declare_parameter<bool>("option.enable_skip_optimization");
    enable_reset_prev_optimization_ =
      declare_parameter<bool>("option.enable_reset_prev_optimization");
    use_footprint_polygon_for_outside_drivable_area_check_ =
      declare_parameter<bool>("option.use_footprint_polygon_for_outside_drivable_area_check");

    // parameter for debug marker
    enable_pub_debug_marker_ = declare_parameter<bool>("option.debug.enable_pub_debug_marker");

    // parameter for debug info
    enable_debug_info_ = declare_parameter<bool>("option.debug.enable_debug_info");
    time_keeper_ptr_->enable_calculation_time_info =
      declare_parameter<bool>("option.debug.enable_calculation_time_info");

    // parameters for ego nearest search
    ego_nearest_param_ = EgoNearestParam(this);

    // parameters for trajectory
    traj_param_ = TrajectoryParam(this);
  }

  // create core algorithm pointers with parameter declaration
  replan_checker_ptr_ = std::make_shared<ReplanChecker>(this, ego_nearest_param_);
  eb_path_smoother_ptr_ = std::make_shared<EBPathSmoother>(
    this, enable_debug_info_, ego_nearest_param_, traj_param_, time_keeper_ptr_);
  mpt_optimizer_ptr_ = std::make_shared<MPTOptimizer>(
    this, enable_debug_info_, ego_nearest_param_, vehicle_info_, traj_param_, debug_data_ptr_,
    time_keeper_ptr_);

  // reset planners
  // NOTE: This function must be called after core algorithms (e.g. mpt_optimizer_) have been
  // initialized.
  initializePlanning();

  // set parameter callback
  // NOTE: This function must be called after core algorithms (e.g. mpt_optimizer_) have been
  // initialized.
  set_param_res_ = this->add_on_set_parameters_callback(
    std::bind(&ObstacleAvoidancePlanner::onParam, this, std::placeholders::_1));
}

rcl_interfaces::msg::SetParametersResult ObstacleAvoidancePlanner::onParam(
  const std::vector<rclcpp::Parameter> & parameters)
{
  using tier4_autoware_utils::updateParam;

  // parameters for option
  updateParam<bool>(
    parameters, "option.enable_outside_drivable_area_stop", enable_outside_drivable_area_stop_);
  updateParam<bool>(parameters, "option.enable_smoothing", enable_smoothing_);
  updateParam<bool>(parameters, "option.enable_skip_optimization", enable_skip_optimization_);
  updateParam<bool>(
    parameters, "option.enable_reset_prev_optimization", enable_reset_prev_optimization_);
  updateParam<bool>(
    parameters, "option.use_footprint_polygon_for_outside_drivable_area_check",
    use_footprint_polygon_for_outside_drivable_area_check_);

  // parameters for debug marker
  updateParam<bool>(parameters, "option.debug.enable_pub_debug_marker", enable_pub_debug_marker_);

  // parameters for debug info
  updateParam<bool>(parameters, "option.debug.enable_debug_info", enable_debug_info_);
  updateParam<bool>(
    parameters, "option.debug.enable_calculation_time_info",
    time_keeper_ptr_->enable_calculation_time_info);

  // parameters for ego nearest search
  ego_nearest_param_.onParam(parameters);

  // parameters for trajectory
  traj_param_.onParam(parameters);

  // parameters for core algorithms
  replan_checker_ptr_->onParam(parameters);
  eb_path_smoother_ptr_->onParam(parameters);
  mpt_optimizer_ptr_->onParam(parameters);

  // reset planners
  initializePlanning();

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  return result;
}

void ObstacleAvoidancePlanner::initializePlanning()
{
  RCLCPP_INFO(get_logger(), "Initialize planning");

  eb_path_smoother_ptr_->initialize(enable_debug_info_, traj_param_);
  mpt_optimizer_ptr_->initialize(enable_debug_info_, traj_param_);

  resetPreviousData();
}

void ObstacleAvoidancePlanner::resetPreviousData()
{
  eb_path_smoother_ptr_->resetPreviousData();
  mpt_optimizer_ptr_->resetPreviousData();

  prev_optimized_traj_points_ptr_ = nullptr;
}

void ObstacleAvoidancePlanner::onPath(const Path::SharedPtr path_ptr)
{
  time_keeper_ptr_->init();
  time_keeper_ptr_->tic(__func__);

  // check if data is ready and valid
  if (!isDataReady(*path_ptr, *get_clock())) {
    return;
  }

  // 0. return if path is backward
  // TODO(murooka): support backward path
  const auto is_driving_forward = driving_direction_checker_.isDrivingForward(path_ptr->points);
  if (!is_driving_forward) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "Backward path is NOT supported. Just converting path to trajectory");

    const auto traj_points = trajectory_utils::convertToTrajectoryPoints(path_ptr->points);
    const auto output_traj_msg = trajectory_utils::createTrajectory(path_ptr->header, traj_points);
    traj_pub_->publish(output_traj_msg);
    return;
  }

  // 1. create planner data
  const auto planner_data = createPlannerData(*path_ptr);

  // 2. generate optimized trajectory
  const auto optimized_traj_points = generateOptimizedTrajectory(planner_data);

  // 3. extend trajectory to connect the optimized trajectory and the following path smoothly
  auto full_traj_points = extendTrajectory(planner_data.traj_points, optimized_traj_points);

  // 4. set zero velocity after stop point
  setZeroVelocityAfterStopPoint(full_traj_points);

  // 5. publish debug data
  publishDebugData(planner_data.header);

  time_keeper_ptr_->toc(__func__, "");
  *time_keeper_ptr_ << "========================================";
  time_keeper_ptr_->endLine();

  // publish calculation_time
  // NOTE: This function must be called after measuring onPath calculation time
  const auto calculation_time_msg = createStringStamped(now(), time_keeper_ptr_->getLog());
  debug_calculation_time_pub_->publish(calculation_time_msg);

  const auto output_traj_msg =
    trajectory_utils::createTrajectory(path_ptr->header, full_traj_points);
  traj_pub_->publish(output_traj_msg);
}

bool ObstacleAvoidancePlanner::isDataReady(const Path & path, rclcpp::Clock clock) const
{
  if (!ego_state_ptr_) {
    RCLCPP_INFO_SKIPFIRST_THROTTLE(get_logger(), clock, 5000, "Waiting for ego pose and twist.");
    return false;
  }

  if (path.points.size() < 2) {
    RCLCPP_INFO_SKIPFIRST_THROTTLE(get_logger(), clock, 5000, "Path points size is less than 1.");
    return false;
  }

  if (path.left_bound.empty() || path.right_bound.empty()) {
    RCLCPP_INFO_SKIPFIRST_THROTTLE(
      get_logger(), clock, 5000, "Left or right bound in path is empty.");
    return false;
  }

  return true;
}

PlannerData ObstacleAvoidancePlanner::createPlannerData(const Path & path) const
{
  // create planner data
  PlannerData planner_data;
  planner_data.header = path.header;
  planner_data.traj_points = trajectory_utils::convertToTrajectoryPoints(path.points);
  planner_data.left_bound = path.left_bound;
  planner_data.right_bound = path.right_bound;
  planner_data.ego_pose = ego_state_ptr_->pose.pose;
  planner_data.ego_vel = ego_state_ptr_->twist.twist.linear.x;

  debug_data_ptr_->ego_pose = planner_data.ego_pose;
  return planner_data;
}

std::vector<TrajectoryPoint> ObstacleAvoidancePlanner::generateOptimizedTrajectory(
  const PlannerData & planner_data)
{
  time_keeper_ptr_->tic(__func__);

  const auto & input_traj_points = planner_data.traj_points;

  // 1. calculate trajectory with EB and MPT
  //    NOTE: This function may return previously optimized trajectory points.
  //          Also, velocity on some points will not be updated for a logic purpose.
  auto optimized_traj_points = optimizeTrajectory(planner_data);

  // 2. update velocity
  //    NOTE: When optimization failed or is skipped, velocity in trajectory points must
  //          be updated since velocity in input trajectory (path) may change.
  applyInputVelocity(optimized_traj_points, input_traj_points, planner_data.ego_pose);

  // 3. insert zero velocity when trajectory is over drivable area
  insertZeroVelocityOutsideDrivableArea(planner_data, optimized_traj_points);

  // 4. publish debug marker
  publishDebugMarkerOfOptimization(optimized_traj_points);

  time_keeper_ptr_->toc(__func__, " ");
  return optimized_traj_points;
}

std::vector<TrajectoryPoint> ObstacleAvoidancePlanner::optimizeTrajectory(
  const PlannerData & planner_data)
{
  time_keeper_ptr_->tic(__func__);
  const auto & p = planner_data;

  // 1. check if replan (= optimization) is required
  const bool reset_prev_optimization = replan_checker_ptr_->isResetRequired(planner_data);
  if (enable_reset_prev_optimization_ || reset_prev_optimization) {
    // NOTE: always replan when resetting previous optimization
    resetPreviousData();
  } else {
    // check replan when not resetting previous optimization
    const bool is_replan_required = replan_checker_ptr_->isReplanRequired(now());
    if (!is_replan_required) {
      return getPrevOptimizedTrajectory(p.traj_points);
    }
  }

  if (enable_skip_optimization_) {
    return p.traj_points;
  }

  // 2. smooth trajectory with elastic band
  const auto eb_traj =
    enable_smoothing_ ? eb_path_smoother_ptr_->getEBTrajectory(planner_data) : p.traj_points;
  if (!eb_traj) {
    return getPrevOptimizedTrajectory(p.traj_points);
  }

  // 3. make trajectory kinematically-feasible and collision-free (= inside the drivable area)
  //    with model predictive trajectory
  const auto mpt_traj = mpt_optimizer_ptr_->getModelPredictiveTrajectory(planner_data, *eb_traj);
  if (!mpt_traj) {
    return getPrevOptimizedTrajectory(p.traj_points);
  }

  // 4. make prev trajectories
  prev_optimized_traj_points_ptr_ = std::make_shared<std::vector<TrajectoryPoint>>(*mpt_traj);

  time_keeper_ptr_->toc(__func__, "    ");
  return *mpt_traj;
}

std::vector<TrajectoryPoint> ObstacleAvoidancePlanner::getPrevOptimizedTrajectory(
  const std::vector<TrajectoryPoint> & traj_points) const
{
  if (prev_optimized_traj_points_ptr_) {
    return *prev_optimized_traj_points_ptr_;
  }

  return traj_points;
}

void ObstacleAvoidancePlanner::applyInputVelocity(
  std::vector<TrajectoryPoint> & output_traj_points,
  const std::vector<TrajectoryPoint> & input_traj_points,
  const geometry_msgs::msg::Pose & ego_pose) const
{
  time_keeper_ptr_->tic(__func__);

  // crop forward for faster calculation
  const auto forward_cropped_input_traj_points = [&]() {
    const double optimized_traj_length = mpt_optimizer_ptr_->getTrajectoryLength();
    constexpr double margin_traj_length = 10.0;

    const size_t ego_seg_idx =
      trajectory_utils::findEgoSegmentIndex(input_traj_points, ego_pose, ego_nearest_param_);
    return trajectory_utils::cropForwardPoints(
      input_traj_points, ego_pose.position, ego_seg_idx,
      optimized_traj_length + margin_traj_length);
  }();

  // update velocity
  size_t input_traj_start_idx = 0;
  for (size_t i = 0; i < output_traj_points.size(); i++) {
    // crop backward for efficient calculation
    const auto cropped_input_traj_points = std::vector<TrajectoryPoint>{
      forward_cropped_input_traj_points.begin() + input_traj_start_idx,
      forward_cropped_input_traj_points.end()};

    const size_t nearest_seg_idx = trajectory_utils::findEgoSegmentIndex(
      cropped_input_traj_points, output_traj_points.at(i).pose, ego_nearest_param_);
    input_traj_start_idx = nearest_seg_idx;

    // calculate velocity with zero order hold
    const double velocity = cropped_input_traj_points.at(nearest_seg_idx).longitudinal_velocity_mps;
    output_traj_points.at(i).longitudinal_velocity_mps = velocity;
  }

  // insert stop point explicitly
  const auto stop_idx = motion_utils::searchZeroVelocityIndex(forward_cropped_input_traj_points);
  if (stop_idx) {
    const auto input_stop_pose = forward_cropped_input_traj_points.at(stop_idx.get()).pose;
    const size_t stop_seg_idx = trajectory_utils::findEgoSegmentIndex(
      output_traj_points, input_stop_pose, ego_nearest_param_);

    // calculate and insert stop pose on output trajectory
    trajectory_utils::insertStopPoint(output_traj_points, input_stop_pose, stop_seg_idx);
  }

  time_keeper_ptr_->toc(__func__, "    ");
}

void ObstacleAvoidancePlanner::insertZeroVelocityOutsideDrivableArea(
  const PlannerData & planner_data, std::vector<TrajectoryPoint> & optimized_traj_points) const
{
  time_keeper_ptr_->tic(__func__);

  if (!enable_outside_drivable_area_stop_) {
    return;
  }

  if (optimized_traj_points.empty()) {
    return;
  }

  // 1. calculate ego_index nearest to optimized_traj_points
  const size_t ego_idx = trajectory_utils::findEgoIndex(
    optimized_traj_points, planner_data.ego_pose, ego_nearest_param_);

  // 2. calculate an end point to check being outside the drivable area
  // NOTE: Some terminal trajectory points tend to be outside the drivable area when
  //       they have high curvature.
  //       Therefore, these points should be ignored to check if being outside the drivable area
  constexpr int num_points_ignore_drivable_area = 5;
  const int end_idx = std::min(
    static_cast<int>(optimized_traj_points.size()) - 1,
    mpt_optimizer_ptr_->getNumberOfPoints() - num_points_ignore_drivable_area);

  // 3. assign zero velocity to the first point being outside the drivable area
  const auto first_outside_idx = [&]() -> std::optional<size_t> {
    for (size_t i = ego_idx; i < static_cast<size_t>(end_idx); ++i) {
      auto & traj_point = optimized_traj_points.at(i);

      // check if the footprint is outside the drivable area
      const bool is_outside = geometry_utils::isOutsideDrivableAreaFromRectangleFootprint(
        traj_point.pose, planner_data.left_bound, planner_data.right_bound, vehicle_info_,
        use_footprint_polygon_for_outside_drivable_area_check_);

      if (is_outside) {
        publishVirtualWall(traj_point.pose);
        return i;
      }
    }
    return std::nullopt;
  }();

  if (first_outside_idx) {
    for (size_t i = *first_outside_idx; i < optimized_traj_points.size(); ++i) {
      optimized_traj_points.at(i).longitudinal_velocity_mps = 0.0;
    }
  }

  time_keeper_ptr_->toc(__func__, "    ");
}

void ObstacleAvoidancePlanner::publishVirtualWall(const geometry_msgs::msg::Pose & stop_pose) const
{
  time_keeper_ptr_->tic(__func__);

  const auto virtual_wall_marker = motion_utils::createStopVirtualWallMarker(
    stop_pose, "outside drivable area", now(), 0, vehicle_info_.max_longitudinal_offset_m);

  virtual_wall_pub_->publish(virtual_wall_marker);
  time_keeper_ptr_->toc(__func__, "      ");
}

void ObstacleAvoidancePlanner::publishDebugMarkerOfOptimization(
  const std::vector<TrajectoryPoint> & traj_points) const
{
  if (!enable_pub_debug_marker_) {
    return;
  }

  time_keeper_ptr_->tic(__func__);

  // debug marker
  time_keeper_ptr_->tic("getDebugMarker");
  const auto debug_marker = getDebugMarker(*debug_data_ptr_, traj_points, vehicle_info_);
  time_keeper_ptr_->toc("getDebugMarker", "      ");

  time_keeper_ptr_->tic("publishDebugMarker");
  debug_markers_pub_->publish(debug_marker);
  time_keeper_ptr_->toc("publishDebugMarker", "      ");

  time_keeper_ptr_->toc(__func__, "    ");
}

std::vector<TrajectoryPoint> ObstacleAvoidancePlanner::extendTrajectory(
  const std::vector<TrajectoryPoint> & traj_points,
  const std::vector<TrajectoryPoint> & optimized_traj_points) const
{
  time_keeper_ptr_->tic(__func__);

  const auto & joint_start_pose = optimized_traj_points.back().pose;

  // calculate end idx of optimized points on path points
  const size_t joint_start_traj_seg_idx =
    trajectory_utils::findEgoSegmentIndex(traj_points, joint_start_pose, ego_nearest_param_);

  // crop trajectory for extension
  constexpr double joint_traj_length_for_smoothing = 5.0;
  const auto joint_end_traj_point_idx = trajectory_utils::getPointIndexAfter(
    traj_points, joint_start_pose.position, joint_start_traj_seg_idx,
    joint_traj_length_for_smoothing);

  // calculate full trajectory points
  const auto full_traj_points = [&]() {
    if (!joint_end_traj_point_idx) {
      return optimized_traj_points;
    }
    const auto extended_traj_points = std::vector<TrajectoryPoint>{
      traj_points.begin() + *joint_end_traj_point_idx, traj_points.end()};
    return concatVectors(optimized_traj_points, extended_traj_points);
  }();

  // resample trajectory points
  auto resampled_traj_points = trajectory_utils::resampleTrajectoryPoints(
    full_traj_points, traj_param_.output_delta_arc_length);

  // update velocity on joint
  for (size_t i = joint_start_traj_seg_idx + 1; i <= joint_end_traj_point_idx; ++i) {
    if (hasZeroVelocity(traj_points.at(i))) {
      if (i != 0 && !hasZeroVelocity(traj_points.at(i - 1))) {
        // Here is when current point is 0 velocity, but previous point is not 0 velocity.
        const auto & input_stop_pose = traj_points.at(i).pose;
        const size_t stop_seg_idx = trajectory_utils::findEgoSegmentIndex(
          resampled_traj_points, input_stop_pose, ego_nearest_param_);

        // calculate and insert stop pose on output trajectory
        trajectory_utils::insertStopPoint(resampled_traj_points, input_stop_pose, stop_seg_idx);
      }
    }
  }

  // debug_data_ptr_->extended_traj_points =
  //   extended_traj_points ? *extended_traj_points : std::vector<TrajectoryPoint>();
  time_keeper_ptr_->toc(__func__, "  ");
  return resampled_traj_points;
}

void ObstacleAvoidancePlanner::publishDebugData(const Header & header) const
{
  time_keeper_ptr_->tic(__func__);

  // publish trajectories
  const auto debug_extended_traj =
    trajectory_utils::createTrajectory(header, debug_data_ptr_->extended_traj_points);
  debug_extended_traj_pub_->publish(debug_extended_traj);

  time_keeper_ptr_->toc(__func__, "  ");
}
}  // namespace obstacle_avoidance_planner

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(obstacle_avoidance_planner::ObstacleAvoidancePlanner)

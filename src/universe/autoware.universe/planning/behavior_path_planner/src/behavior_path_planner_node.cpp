// Copyright 2021-2023 Tier IV, Inc.
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

#include "behavior_path_planner/behavior_path_planner_node.hpp"

#include "behavior_path_planner/marker_util/debug_utilities.hpp"
#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/util/drivable_area_expansion/map_utils.hpp"

#include <tier4_autoware_utils/tier4_autoware_utils.hpp>
#include <vehicle_info_util/vehicle_info_util.hpp>

#include <tier4_planning_msgs/msg/path_change_module_id.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{
rclcpp::SubscriptionOptions createSubscriptionOptions(rclcpp::Node * node_ptr)
{
  rclcpp::CallbackGroup::SharedPtr callback_group =
    node_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  auto sub_opt = rclcpp::SubscriptionOptions();
  sub_opt.callback_group = callback_group;

  return sub_opt;
}
}  // namespace

namespace behavior_path_planner
{
using tier4_planning_msgs::msg::PathChangeModuleId;
using vehicle_info_util::VehicleInfoUtil;

BehaviorPathPlannerNode::BehaviorPathPlannerNode(const rclcpp::NodeOptions & node_options)
: Node("behavior_path_planner", node_options)
{
  using std::placeholders::_1;
  using std::chrono_literals::operator""ms;

  // data_manager
  {
    planner_data_ = std::make_shared<PlannerData>();
    planner_data_->parameters = getCommonParam();
    planner_data_->drivable_area_expansion_parameters.init(*this);
  }

  // publisher
  path_publisher_ = create_publisher<PathWithLaneId>("~/output/path", 1);
  turn_signal_publisher_ =
    create_publisher<TurnIndicatorsCommand>("~/output/turn_indicators_cmd", 1);
  hazard_signal_publisher_ = create_publisher<HazardLightsCommand>("~/output/hazard_lights_cmd", 1);
  modified_goal_publisher_ = create_publisher<PoseWithUuidStamped>("~/output/modified_goal", 1);
  debug_avoidance_msg_array_publisher_ =
    create_publisher<AvoidanceDebugMsgArray>("~/debug/avoidance_debug_message_array", 1);
  debug_lane_change_msg_array_publisher_ =
    create_publisher<LaneChangeDebugMsgArray>("~/debug/lane_change_debug_message_array", 1);

  if (planner_data_->parameters.visualize_maximum_drivable_area) {
    debug_maximum_drivable_area_publisher_ =
      create_publisher<MarkerArray>("~/maximum_drivable_area", 1);
  }

  bound_publisher_ = create_publisher<MarkerArray>("~/debug/bound", 1);

  // subscriber
  velocity_subscriber_ = create_subscription<Odometry>(
    "~/input/odometry", 1, std::bind(&BehaviorPathPlannerNode::onOdometry, this, _1),
    createSubscriptionOptions(this));
  acceleration_subscriber_ = create_subscription<AccelWithCovarianceStamped>(
    "~/input/accel", 1, std::bind(&BehaviorPathPlannerNode::onAcceleration, this, _1),
    createSubscriptionOptions(this));
  perception_subscriber_ = create_subscription<PredictedObjects>(
    "~/input/perception", 1, std::bind(&BehaviorPathPlannerNode::onPerception, this, _1),
    createSubscriptionOptions(this));
  occupancy_grid_subscriber_ = create_subscription<OccupancyGrid>(
    "~/input/occupancy_grid_map", 1, std::bind(&BehaviorPathPlannerNode::onOccupancyGrid, this, _1),
    createSubscriptionOptions(this));
  costmap_subscriber_ = create_subscription<OccupancyGrid>(
    "~/input/costmap", 1, std::bind(&BehaviorPathPlannerNode::onCostMap, this, _1),
    createSubscriptionOptions(this));
  lateral_offset_subscriber_ = this->create_subscription<LateralOffset>(
    "~/input/lateral_offset", 1, std::bind(&BehaviorPathPlannerNode::onLateralOffset, this, _1),
    createSubscriptionOptions(this));
  operation_mode_subscriber_ = create_subscription<OperationModeState>(
    "/system/operation_mode/state", 1,
    std::bind(&BehaviorPathPlannerNode::onOperationMode, this, _1),
    createSubscriptionOptions(this));
  scenario_subscriber_ = create_subscription<Scenario>(
    "~/input/scenario", 1,
    [this](const Scenario::ConstSharedPtr msg) {
      current_scenario_ = std::make_shared<Scenario>(*msg);
    },
    createSubscriptionOptions(this));

  // route_handler
  auto qos_transient_local = rclcpp::QoS{1}.transient_local();
  vector_map_subscriber_ = create_subscription<HADMapBin>(
    "~/input/vector_map", qos_transient_local, std::bind(&BehaviorPathPlannerNode::onMap, this, _1),
    createSubscriptionOptions(this));
  route_subscriber_ = create_subscription<LaneletRoute>(
    "~/input/route", qos_transient_local, std::bind(&BehaviorPathPlannerNode::onRoute, this, _1),
    createSubscriptionOptions(this));

  // set parameters
  {
    avoidance_param_ptr_ = std::make_shared<AvoidanceParameters>(getAvoidanceParam());
    lane_change_param_ptr_ = std::make_shared<LaneChangeParameters>(getLaneChangeParam());
    lane_following_param_ptr_ = std::make_shared<LaneFollowingParameters>(getLaneFollowingParam());
    pull_out_param_ptr_ = std::make_shared<PullOutParameters>(getPullOutParam());
    pull_over_param_ptr_ = std::make_shared<PullOverParameters>(getPullOverParam());
    side_shift_param_ptr_ = std::make_shared<SideShiftParameters>(getSideShiftParam());
  }

  m_set_param_res = this->add_on_set_parameters_callback(
    std::bind(&BehaviorPathPlannerNode::onSetParam, this, std::placeholders::_1));

#ifdef USE_OLD_ARCHITECTURE
  // behavior tree manager
  {
    RCLCPP_INFO(get_logger(), "use behavior tree.");

    const std::string path_candidate_name_space = "/planning/path_candidate/";
    mutex_bt_.lock();

    bt_manager_ = std::make_shared<BehaviorTreeManager>(*this, getBehaviorTreeManagerParam());

    auto side_shift_module =
      std::make_shared<SideShiftModule>("SideShift", *this, side_shift_param_ptr_);
    bt_manager_->registerSceneModule(side_shift_module);

    auto avoidance_module =
      std::make_shared<AvoidanceModule>("Avoidance", *this, avoidance_param_ptr_);
    path_candidate_publishers_.emplace(
      "Avoidance", create_publisher<Path>(path_candidate_name_space + "avoidance", 1));
    bt_manager_->registerSceneModule(avoidance_module);

    auto lane_following_module =
      std::make_shared<LaneFollowingModule>("LaneFollowing", *this, lane_following_param_ptr_);
    bt_manager_->registerSceneModule(lane_following_module);

    auto ext_request_lane_change_right_module =
      std::make_shared<ExternalRequestLaneChangeRightModule>(
        "ExternalRequestLaneChangeRight", *this, lane_change_param_ptr_);
    path_candidate_publishers_.emplace(
      "ExternalRequestLaneChangeRight",
      create_publisher<Path>(path_candidate_name_space + "ext_request_lane_change_right", 1));
    bt_manager_->registerSceneModule(ext_request_lane_change_right_module);

    auto ext_request_lane_change_left_module =
      std::make_shared<ExternalRequestLaneChangeLeftModule>(
        "ExternalRequestLaneChangeLeft", *this, lane_change_param_ptr_);
    path_candidate_publishers_.emplace(
      "ExternalRequestLaneChangeLeft",
      create_publisher<Path>(path_candidate_name_space + "ext_request_lane_change_left", 1));
    bt_manager_->registerSceneModule(ext_request_lane_change_left_module);

    auto lane_change_module =
      std::make_shared<LaneChangeModule>("LaneChange", *this, lane_change_param_ptr_);
    path_candidate_publishers_.emplace(
      "LaneChange", create_publisher<Path>(path_candidate_name_space + "lane_change", 1));
    bt_manager_->registerSceneModule(lane_change_module);

    auto pull_over_module =
      std::make_shared<PullOverModule>("PullOver", *this, pull_over_param_ptr_);
    path_candidate_publishers_.emplace(
      "PullOver", create_publisher<Path>(path_candidate_name_space + "pull_over", 1));
    bt_manager_->registerSceneModule(pull_over_module);

    auto pull_out_module = std::make_shared<PullOutModule>("PullOut", *this, pull_out_param_ptr_);
    path_candidate_publishers_.emplace(
      "PullOut", create_publisher<Path>(path_candidate_name_space + "pull_out", 1));
    bt_manager_->registerSceneModule(pull_out_module);

    bt_manager_->createBehaviorTree();

    mutex_bt_.unlock();
  }
#else
  {
    RCLCPP_INFO(get_logger(), "not use behavior tree.");

    const std::string path_candidate_name_space = "/planning/path_candidate/";
    const std::string path_reference_name_space = "/planning/path_reference/";

    mutex_bt_.lock();

    const auto & p = planner_data_->parameters;
    planner_manager_ =
      std::make_shared<PlannerManager>(*this, lane_following_param_ptr_, p.verbose);

    if (p.config_pull_out.enable_module) {
      auto manager = std::make_shared<PullOutModuleManager>(
        this, "pull_out", p.config_pull_out, pull_out_param_ptr_);
      planner_manager_->registerSceneModuleManager(manager);
      path_candidate_publishers_.emplace(
        "pull_out", create_publisher<Path>(path_candidate_name_space + "pull_out", 1));
      path_reference_publishers_.emplace(
        "pull_out", create_publisher<Path>(path_reference_name_space + "pull_out", 1));
    }

    if (p.config_pull_over.enable_module) {
      auto manager = std::make_shared<PullOverModuleManager>(
        this, "pull_over", p.config_pull_over, pull_over_param_ptr_);
      planner_manager_->registerSceneModuleManager(manager);
      path_candidate_publishers_.emplace(
        "pull_over", create_publisher<Path>(path_candidate_name_space + "pull_over", 1));
      path_reference_publishers_.emplace(
        "pull_over", create_publisher<Path>(path_reference_name_space + "pull_over", 1));
    }

    if (p.config_side_shift.enable_module) {
      auto manager = std::make_shared<SideShiftModuleManager>(
        this, "side_shift", p.config_side_shift, side_shift_param_ptr_);
      planner_manager_->registerSceneModuleManager(manager);
      path_candidate_publishers_.emplace(
        "side_shift", create_publisher<Path>(path_candidate_name_space + "side_shift", 1));
      path_reference_publishers_.emplace(
        "side_shift", create_publisher<Path>(path_reference_name_space + "side_shift", 1));
    }

    if (p.config_lane_change.enable_module) {
      const std::string module_topic = "lane_change";
      auto manager = std::make_shared<LaneChangeModuleManager>(
        this, module_topic, p.config_lane_change, lane_change_param_ptr_);
      planner_manager_->registerSceneModuleManager(manager);
      path_candidate_publishers_.emplace(
        module_topic, create_publisher<Path>(path_candidate_name_space + module_topic, 1));
      path_reference_publishers_.emplace(
        module_topic, create_publisher<Path>(path_reference_name_space + module_topic, 1));
    }

    if (p.config_avoidance.enable_module) {
      auto manager = std::make_shared<AvoidanceModuleManager>(
        this, "avoidance", p.config_avoidance, avoidance_param_ptr_);
      planner_manager_->registerSceneModuleManager(manager);
      path_candidate_publishers_.emplace(
        "avoidance", create_publisher<Path>(path_candidate_name_space + "avoidance", 1));
      path_reference_publishers_.emplace(
        "avoidance", create_publisher<Path>(path_reference_name_space + "avoidance", 1));
    }

    mutex_bt_.unlock();
  }
#endif

  // turn signal decider
  {
    const double turn_signal_intersection_search_distance =
      planner_data_->parameters.turn_signal_intersection_search_distance;
    const double turn_signal_intersection_angle_threshold_deg =
      planner_data_->parameters.turn_signal_intersection_angle_threshold_deg;
    const double turn_signal_search_time = planner_data_->parameters.turn_signal_search_time;
    turn_signal_decider_.setParameters(
      planner_data_->parameters.base_link2front, turn_signal_intersection_search_distance,
      turn_signal_search_time, turn_signal_intersection_angle_threshold_deg);
  }

  steering_factor_interface_ptr_ = std::make_unique<SteeringFactorInterface>(this, "intersection");

  // Start timer
  {
    const auto planning_hz = declare_parameter("planning_hz", 10.0);
    const auto period_ns = rclcpp::Rate(planning_hz).period();
    timer_ = rclcpp::create_timer(
      this, get_clock(), period_ns, std::bind(&BehaviorPathPlannerNode::run, this));
  }
}

BehaviorPathPlannerParameters BehaviorPathPlannerNode::getCommonParam()
{
  BehaviorPathPlannerParameters p{};

  p.verbose = declare_parameter<bool>("verbose");

  {
    const std::string ns = "pull_out.";
    p.config_pull_out.enable_module = declare_parameter<bool>(ns + "enable_module");
    p.config_pull_out.enable_simultaneous_execution =
      declare_parameter<bool>(ns + "enable_simultaneous_execution");
    p.config_pull_out.priority = declare_parameter<int>(ns + "priority");
    p.config_pull_out.max_module_size = declare_parameter<int>(ns + "max_module_size");
  }

  {
    const std::string ns = "pull_over.";
    p.config_pull_over.enable_module = declare_parameter<bool>(ns + "enable_module");
    p.config_pull_over.enable_simultaneous_execution =
      declare_parameter<bool>(ns + "enable_simultaneous_execution");
    p.config_pull_over.priority = declare_parameter<int>(ns + "priority");
    p.config_pull_over.max_module_size = declare_parameter<int>(ns + "max_module_size");
  }

  {
    const std::string ns = "side_shift.";
    p.config_side_shift.enable_module = declare_parameter<bool>(ns + "enable_module");
    p.config_side_shift.enable_simultaneous_execution =
      declare_parameter<bool>(ns + "enable_simultaneous_execution");
    p.config_side_shift.priority = declare_parameter<int>(ns + "priority");
    p.config_side_shift.max_module_size = declare_parameter<int>(ns + "max_module_size");
  }

  {
    const std::string ns = "lane_change.";
    p.config_lane_change.enable_module = declare_parameter<bool>(ns + "enable_module");
    p.config_lane_change.enable_simultaneous_execution =
      declare_parameter<bool>(ns + "enable_simultaneous_execution");
    p.config_lane_change.priority = declare_parameter<int>(ns + "priority");
    p.config_lane_change.max_module_size = declare_parameter<int>(ns + "max_module_size");
  }

  {
    const std::string ns = "avoidance.";
    p.config_avoidance.enable_module = declare_parameter<bool>(ns + "enable_module");
    p.config_avoidance.enable_simultaneous_execution =
      declare_parameter<bool>(ns + "enable_simultaneous_execution");
    p.config_avoidance.priority = declare_parameter<int>(ns + "priority");
    p.config_avoidance.max_module_size = declare_parameter<int>(ns + "max_module_size");
  }

  // vehicle info
  const auto vehicle_info = VehicleInfoUtil(*this).getVehicleInfo();
  p.vehicle_info = vehicle_info;
  p.vehicle_width = vehicle_info.vehicle_width_m;
  p.vehicle_length = vehicle_info.vehicle_length_m;
  p.wheel_tread = vehicle_info.wheel_tread_m;
  p.wheel_base = vehicle_info.wheel_base_m;
  p.front_overhang = vehicle_info.front_overhang_m;
  p.rear_overhang = vehicle_info.rear_overhang_m;
  p.left_over_hang = vehicle_info.left_overhang_m;
  p.right_over_hang = vehicle_info.right_overhang_m;
  p.base_link2front = vehicle_info.max_longitudinal_offset_m;
  p.base_link2rear = p.rear_overhang;

  // NOTE: backward_path_length is used not only calculating path length but also calculating the
  // size of a drivable area.
  //       The drivable area has to cover not the base link but the vehicle itself. Therefore
  //       rear_overhang must be added to backward_path_length. In addition, because of the
  //       calculation of the drivable area in the obstacle_avoidance_planner package, the drivable
  //       area has to be a little longer than the backward_path_length parameter by adding
  //       min_backward_offset.
  constexpr double min_backward_offset = 1.0;
  const double backward_offset = vehicle_info.rear_overhang_m + min_backward_offset;

  // ROS parameters
  p.backward_path_length = declare_parameter("backward_path_length", 5.0) + backward_offset;
  p.forward_path_length = declare_parameter("forward_path_length", 100.0);
  p.backward_length_buffer_for_end_of_lane =
    declare_parameter("lane_change.backward_length_buffer_for_end_of_lane", 5.0);
  p.backward_length_buffer_for_end_of_pull_over =
    declare_parameter("backward_length_buffer_for_end_of_pull_over", 5.0);
  p.backward_length_buffer_for_end_of_pull_out =
    declare_parameter("backward_length_buffer_for_end_of_pull_out", 5.0);
  p.minimum_lane_change_length = declare_parameter("lane_change.minimum_lane_change_length", 8.0);
  p.minimum_lane_change_prepare_distance =
    declare_parameter("lane_change.minimum_lane_change_prepare_distance", 2.0);

  p.minimum_pull_over_length = declare_parameter("minimum_pull_over_length", 15.0);
  p.refine_goal_search_radius_range = declare_parameter("refine_goal_search_radius_range", 7.5);
  p.turn_signal_intersection_search_distance =
    declare_parameter("turn_signal_intersection_search_distance", 30.0);
  p.turn_signal_intersection_angle_threshold_deg =
    declare_parameter("turn_signal_intersection_angle_threshold_deg", 15.0);
  p.turn_signal_minimum_search_distance =
    declare_parameter("turn_signal_minimum_search_distance", 10.0);
  p.turn_signal_search_time = declare_parameter("turn_signal_search_time", 3.0);
  p.turn_signal_shift_length_threshold =
    declare_parameter("turn_signal_shift_length_threshold", 0.3);
  p.turn_signal_on_swerving = declare_parameter("turn_signal_on_swerving", true);

  p.path_interval = declare_parameter<double>("path_interval");
  p.visualize_maximum_drivable_area = declare_parameter("visualize_maximum_drivable_area", true);
  p.ego_nearest_dist_threshold = declare_parameter<double>("ego_nearest_dist_threshold");
  p.ego_nearest_yaw_threshold = declare_parameter<double>("ego_nearest_yaw_threshold");

  p.lateral_distance_max_threshold = declare_parameter("lateral_distance_max_threshold", 2.0);
  p.longitudinal_distance_min_threshold =
    declare_parameter("longitudinal_distance_min_threshold", 3.0);

  p.expected_front_deceleration = declare_parameter("expected_front_deceleration", -0.5);
  p.expected_rear_deceleration = declare_parameter("expected_rear_deceleration", -1.0);

  p.expected_front_deceleration_for_abort =
    declare_parameter("expected_front_deceleration_for_abort", -2.0);
  p.expected_rear_deceleration_for_abort =
    declare_parameter("expected_rear_deceleration_for_abort", -2.5);

  p.rear_vehicle_reaction_time = declare_parameter("rear_vehicle_reaction_time", 2.0);
  p.rear_vehicle_safety_time_margin = declare_parameter("rear_vehicle_safety_time_margin", 2.0);

  if (p.backward_length_buffer_for_end_of_lane < 1.0) {
    RCLCPP_WARN_STREAM(
      get_logger(), "Lane change buffer must be more than 1 meter. Modifying the buffer.");
  }
  return p;
}

SideShiftParameters BehaviorPathPlannerNode::getSideShiftParam()
{
  const auto dp = [this](const std::string & str, auto def_val) {
    std::string name = "side_shift." + str;
    return this->declare_parameter(name, def_val);
  };

  SideShiftParameters p{};
  p.min_distance_to_start_shifting = dp("min_distance_to_start_shifting", 5.0);
  p.time_to_start_shifting = dp("time_to_start_shifting", 1.0);
  p.shifting_lateral_jerk = dp("shifting_lateral_jerk", 0.2);
  p.min_shifting_distance = dp("min_shifting_distance", 5.0);
  p.min_shifting_speed = dp("min_shifting_speed", 5.56);
  p.drivable_area_right_bound_offset = dp("drivable_area_right_bound_offset", 0.0);
  p.shift_request_time_limit = dp("shift_request_time_limit", 1.0);
  p.drivable_area_left_bound_offset = dp("drivable_area_left_bound_offset", 0.0);
  p.drivable_area_types_to_skip =
    dp("drivable_area_types_to_skip", std::vector<std::string>({"road_border"}));

  return p;
}

AvoidanceParameters BehaviorPathPlannerNode::getAvoidanceParam()
{
  AvoidanceParameters p{};
  // general params
  {
    std::string ns = "avoidance.";
    p.resample_interval_for_planning =
      declare_parameter<double>(ns + "resample_interval_for_planning");
    p.resample_interval_for_output = declare_parameter<double>(ns + "resample_interval_for_output");
    p.detection_area_right_expand_dist =
      declare_parameter<double>(ns + "detection_area_right_expand_dist");
    p.detection_area_left_expand_dist =
      declare_parameter<double>(ns + "detection_area_left_expand_dist");
    p.drivable_area_right_bound_offset =
      declare_parameter<double>(ns + "drivable_area_right_bound_offset");
    p.drivable_area_left_bound_offset =
      declare_parameter<double>(ns + "drivable_area_left_bound_offset");
    p.drivable_area_types_to_skip =
      declare_parameter<std::vector<std::string>>(ns + "drivable_area_types_to_skip");
    p.object_envelope_buffer = declare_parameter<double>(ns + "object_envelope_buffer");
    p.enable_bound_clipping = declare_parameter<bool>(ns + "enable_bound_clipping");
    p.enable_avoidance_over_same_direction =
      declare_parameter<bool>(ns + "enable_avoidance_over_same_direction");
    p.enable_avoidance_over_opposite_direction =
      declare_parameter<bool>(ns + "enable_avoidance_over_opposite_direction");
    p.enable_update_path_when_object_is_gone =
      declare_parameter<bool>(ns + "enable_update_path_when_object_is_gone");
    p.enable_force_avoidance_for_stopped_vehicle =
      declare_parameter<bool>(ns + "enable_force_avoidance_for_stopped_vehicle");
    p.enable_safety_check = declare_parameter<bool>(ns + "enable_safety_check");
    p.enable_yield_maneuver = declare_parameter<bool>(ns + "enable_yield_maneuver");
    p.disable_path_update = declare_parameter<bool>(ns + "disable_path_update");
    p.publish_debug_marker = declare_parameter<bool>(ns + "publish_debug_marker");
    p.print_debug_info = declare_parameter<bool>(ns + "print_debug_info");
  }

  // target object
  {
    std::string ns = "avoidance.target_object.";
    p.avoid_car = declare_parameter<bool>(ns + "car");
    p.avoid_truck = declare_parameter<bool>(ns + "truck");
    p.avoid_bus = declare_parameter<bool>(ns + "bus");
    p.avoid_trailer = declare_parameter<bool>(ns + "trailer");
    p.avoid_unknown = declare_parameter<bool>(ns + "unknown");
    p.avoid_bicycle = declare_parameter<bool>(ns + "bicycle");
    p.avoid_motorcycle = declare_parameter<bool>(ns + "motorcycle");
    p.avoid_pedestrian = declare_parameter<bool>(ns + "pedestrian");
  }

  // target filtering
  {
    std::string ns = "avoidance.target_filtering.";
    p.threshold_speed_object_is_stopped =
      declare_parameter<double>(ns + "threshold_speed_object_is_stopped");
    p.threshold_time_object_is_moving =
      declare_parameter<double>(ns + "threshold_time_object_is_moving");
    p.threshold_time_force_avoidance_for_stopped_vehicle =
      declare_parameter<double>(ns + "threshold_time_force_avoidance_for_stopped_vehicle");
    p.object_check_force_avoidance_clearance =
      declare_parameter<double>(ns + "object_check_force_avoidance_clearance");
    p.object_check_forward_distance =
      declare_parameter<double>(ns + "object_check_forward_distance");
    p.object_check_backward_distance =
      declare_parameter<double>(ns + "object_check_backward_distance");
    p.object_check_goal_distance = declare_parameter<double>(ns + "object_check_goal_distance");
    p.threshold_distance_object_is_on_center =
      declare_parameter<double>(ns + "threshold_distance_object_is_on_center");
    p.object_check_shiftable_ratio = declare_parameter<double>(ns + "object_check_shiftable_ratio");
    p.object_check_min_road_shoulder_width =
      declare_parameter<double>(ns + "object_check_min_road_shoulder_width");
    p.object_last_seen_threshold = declare_parameter<double>(ns + "object_last_seen_threshold");
  }

  // safety check
  {
    std::string ns = "avoidance.safety_check.";
    p.safety_check_backward_distance =
      declare_parameter<double>(ns + "safety_check_backward_distance");
    p.safety_check_time_horizon = declare_parameter<double>(ns + "safety_check_time_horizon");
    p.safety_check_idling_time = declare_parameter<double>(ns + "safety_check_idling_time");
    p.safety_check_accel_for_rss = declare_parameter<double>(ns + "safety_check_accel_for_rss");
    p.safety_check_hysteresis_factor =
      declare_parameter<double>(ns + "safety_check_hysteresis_factor");
  }

  // avoidance maneuver (lateral)
  {
    std::string ns = "avoidance.avoidance.lateral.";
    p.lateral_collision_margin = declare_parameter<double>(ns + "lateral_collision_margin");
    p.lateral_collision_safety_buffer =
      declare_parameter<double>(ns + "lateral_collision_safety_buffer");
    p.lateral_passable_safety_buffer =
      declare_parameter<double>(ns + "lateral_passable_safety_buffer");
    p.road_shoulder_safety_margin = declare_parameter<double>(ns + "road_shoulder_safety_margin");
    p.avoidance_execution_lateral_threshold =
      declare_parameter<double>(ns + "avoidance_execution_lateral_threshold");
    p.max_right_shift_length = declare_parameter<double>(ns + "max_right_shift_length");
    p.max_left_shift_length = declare_parameter<double>(ns + "max_left_shift_length");
  }

  // avoidance maneuver (longitudinal)
  {
    std::string ns = "avoidance.avoidance.longitudinal.";
    p.prepare_time = declare_parameter<double>(ns + "prepare_time");
    p.longitudinal_collision_safety_buffer =
      declare_parameter<double>(ns + "longitudinal_collision_safety_buffer");
    p.min_prepare_distance = declare_parameter<double>(ns + "min_prepare_distance");
    p.min_avoidance_distance = declare_parameter<double>(ns + "min_avoidance_distance");
    p.min_nominal_avoidance_speed = declare_parameter<double>(ns + "min_nominal_avoidance_speed");
    p.min_sharp_avoidance_speed = declare_parameter<double>(ns + "min_sharp_avoidance_speed");
  }

  // yield
  {
    std::string ns = "avoidance.yield.";
    p.yield_velocity = declare_parameter<double>(ns + "yield_velocity");
  }

  // stop
  {
    std::string ns = "avoidance.stop.";
    p.stop_min_distance = declare_parameter<double>(ns + "min_distance");
    p.stop_max_distance = declare_parameter<double>(ns + "max_distance");
  }

  // constraints
  {
    std::string ns = "avoidance.constraints.";
    p.use_constraints_for_decel = declare_parameter<bool>(ns + "use_constraints_for_decel");
  }

  // constraints (longitudinal)
  {
    std::string ns = "avoidance.constraints.longitudinal.";
    p.nominal_deceleration = declare_parameter<double>(ns + "nominal_deceleration");
    p.nominal_jerk = declare_parameter<double>(ns + "nominal_jerk");
    p.max_deceleration = declare_parameter<double>(ns + "max_deceleration");
    p.max_jerk = declare_parameter<double>(ns + "max_jerk");
    p.min_avoidance_speed_for_acc_prevention =
      declare_parameter<double>(ns + "min_avoidance_speed_for_acc_prevention");
    p.max_avoidance_acceleration = declare_parameter<double>(ns + "max_avoidance_acceleration");
  }

  // constraints (lateral)
  {
    std::string ns = "avoidance.constraints.lateral.";
    p.nominal_lateral_jerk = declare_parameter<double>(ns + "nominal_lateral_jerk");
    p.max_lateral_jerk = declare_parameter<double>(ns + "max_lateral_jerk");
  }

  // velocity matrix
  {
    std::string ns = "avoidance.target_velocity_matrix.";
    p.col_size = declare_parameter<int>(ns + "col_size");
    p.target_velocity_matrix = declare_parameter<std::vector<double>>(ns + "matrix");
  }

  return p;
}

LaneFollowingParameters BehaviorPathPlannerNode::getLaneFollowingParam()
{
  LaneFollowingParameters p{};
  p.drivable_area_right_bound_offset =
    declare_parameter("lane_following.drivable_area_right_bound_offset", 0.0);
  p.drivable_area_left_bound_offset =
    declare_parameter("lane_following.drivable_area_left_bound_offset", 0.0);
  p.drivable_area_types_to_skip = declare_parameter(
    "lane_following.drivable_area_types_to_skip", std::vector<std::string>({"road_border"}));
  p.lane_change_prepare_duration =
    declare_parameter("lane_following.lane_change_prepare_duration", 2.0);
  return p;
}

LaneChangeParameters BehaviorPathPlannerNode::getLaneChangeParam()
{
  LaneChangeParameters p{};
  const auto parameter = [](std::string && name) { return "lane_change." + name; };

  // trajectory generation
  p.lane_change_prepare_duration =
    declare_parameter<double>(parameter("lane_change_prepare_duration"));
  p.lane_changing_safety_check_duration =
    declare_parameter<double>(parameter("lane_changing_safety_check_duration"));
  p.lane_changing_lateral_jerk = declare_parameter<double>(parameter("lane_changing_lateral_jerk"));
  p.lane_changing_lateral_acc = declare_parameter<double>(parameter("lane_changing_lateral_acc"));
  p.lane_change_finish_judge_buffer =
    declare_parameter<double>(parameter("lane_change_finish_judge_buffer"));
  p.minimum_lane_change_velocity =
    declare_parameter<double>(parameter("minimum_lane_change_velocity"));
  p.prediction_time_resolution = declare_parameter<double>(parameter("prediction_time_resolution"));
  p.maximum_deceleration = declare_parameter<double>(parameter("maximum_deceleration"));
  p.lane_change_sampling_num = declare_parameter<int>(parameter("lane_change_sampling_num"));

  // collision check
  p.enable_collision_check_at_prepare_phase =
    declare_parameter<bool>(parameter("enable_collision_check_at_prepare_phase"));
  p.prepare_phase_ignore_target_speed_thresh =
    declare_parameter<double>(parameter("prepare_phase_ignore_target_speed_thresh"));
  p.use_predicted_path_outside_lanelet =
    declare_parameter<bool>(parameter("use_predicted_path_outside_lanelet"));
  p.use_all_predicted_path = declare_parameter<bool>(parameter("use_all_predicted_path"));

  // abort
  p.enable_cancel_lane_change = declare_parameter<bool>(parameter("enable_cancel_lane_change"));
  p.enable_abort_lane_change = declare_parameter<bool>(parameter("enable_abort_lane_change"));

  p.abort_delta_time = declare_parameter<double>(parameter("abort_delta_time"));
  p.abort_max_lateral_jerk = declare_parameter<double>(parameter("abort_max_lateral_jerk"));

  // drivable area expansion
  p.drivable_area_right_bound_offset =
    declare_parameter<double>(parameter("drivable_area_right_bound_offset"));
  p.drivable_area_left_bound_offset =
    declare_parameter<double>(parameter("drivable_area_left_bound_offset"));
  p.drivable_area_types_to_skip =
    declare_parameter<std::vector<std::string>>(parameter("drivable_area_types_to_skip"));

  // debug marker
  p.publish_debug_marker = declare_parameter<bool>(parameter("publish_debug_marker"));

  // validation of parameters
  if (p.lane_change_sampling_num < 1) {
    RCLCPP_FATAL_STREAM(
      get_logger(), "lane_change_sampling_num must be positive integer. Given parameter: "
                      << p.lane_change_sampling_num << std::endl
                      << "Terminating the program...");
    exit(EXIT_FAILURE);
  }
  if (p.maximum_deceleration < 0.0) {
    RCLCPP_FATAL_STREAM(
      get_logger(), "maximum_deceleration cannot be negative value. Given parameter: "
                      << p.maximum_deceleration << std::endl
                      << "Terminating the program...");
    exit(EXIT_FAILURE);
  }

  if (p.abort_delta_time < 1.0) {
    RCLCPP_FATAL_STREAM(
      get_logger(), "abort_delta_time: " << p.abort_delta_time << ", is too short.\n"
                                         << "Terminating the program...");
    exit(EXIT_FAILURE);
  }

  const auto lc_buffer =
    this->get_parameter("lane_change.backward_length_buffer_for_end_of_lane").get_value<double>();
  if (lc_buffer < p.lane_change_finish_judge_buffer + 1.0) {
    p.lane_change_finish_judge_buffer = lc_buffer - 1;
    RCLCPP_WARN_STREAM(
      get_logger(), "lane change buffer is less than finish buffer. Modifying the value to "
                      << p.lane_change_finish_judge_buffer << "....");
  }

  return p;
}

PullOverParameters BehaviorPathPlannerNode::getPullOverParam()
{
  PullOverParameters p;

  {
    std::string ns = "pull_over.";
    p.request_length = declare_parameter<double>(ns + "request_length");
    p.th_stopped_velocity = declare_parameter<double>(ns + "th_stopped_velocity");
    p.th_arrived_distance = declare_parameter<double>(ns + "th_arrived_distance");
    p.th_stopped_time = declare_parameter<double>(ns + "th_stopped_time");
    p.margin_from_boundary = declare_parameter<double>(ns + "margin_from_boundary");
    p.decide_path_distance = declare_parameter<double>(ns + "decide_path_distance");
    p.maximum_deceleration = declare_parameter<double>(ns + "maximum_deceleration");
    // goal research
    p.enable_goal_research = declare_parameter<bool>(ns + "enable_goal_research");
    p.search_priority = declare_parameter<std::string>(ns + "search_priority");
    p.forward_goal_search_length = declare_parameter<double>(ns + "forward_goal_search_length");
    p.backward_goal_search_length = declare_parameter<double>(ns + "backward_goal_search_length");
    p.goal_search_interval = declare_parameter<double>(ns + "goal_search_interval");
    p.longitudinal_margin = declare_parameter<double>(ns + "longitudinal_margin");
    p.max_lateral_offset = declare_parameter<double>(ns + "max_lateral_offset");
    p.lateral_offset_interval = declare_parameter<double>(ns + "lateral_offset_interval");
    p.ignore_distance_from_lane_start =
      declare_parameter<double>(ns + "ignore_distance_from_lane_start");
    // occupancy grid map
    p.use_occupancy_grid = declare_parameter<bool>(ns + "use_occupancy_grid");
    p.use_occupancy_grid_for_longitudinal_margin =
      declare_parameter<bool>(ns + "use_occupancy_grid_for_longitudinal_margin");
    p.occupancy_grid_collision_check_margin =
      declare_parameter<double>(ns + "occupancy_grid_collision_check_margin");
    p.theta_size = declare_parameter<int>(ns + "theta_size");
    p.obstacle_threshold = declare_parameter<int>(ns + "obstacle_threshold");
    // object recognition
    p.use_object_recognition = declare_parameter<bool>(ns + "use_object_recognition");
    p.object_recognition_collision_check_margin =
      declare_parameter<double>(ns + "object_recognition_collision_check_margin");
    // shift path
    p.enable_shift_parking = declare_parameter<bool>(ns + "enable_shift_parking");
    p.pull_over_sampling_num = declare_parameter<int>(ns + "pull_over_sampling_num");
    p.maximum_lateral_jerk = declare_parameter<double>(ns + "maximum_lateral_jerk");
    p.minimum_lateral_jerk = declare_parameter<double>(ns + "minimum_lateral_jerk");
    p.deceleration_interval = declare_parameter<double>(ns + "deceleration_interval");
    p.pull_over_velocity = declare_parameter<double>(ns + "pull_over_velocity");
    p.pull_over_minimum_velocity = declare_parameter<double>(ns + "pull_over_minimum_velocity");
    p.after_pull_over_straight_distance =
      declare_parameter<double>(ns + "after_pull_over_straight_distance");
    // parallel parking
    p.enable_arc_forward_parking = declare_parameter<bool>(ns + "enable_arc_forward_parking");
    p.enable_arc_backward_parking = declare_parameter<bool>(ns + "enable_arc_backward_parking");
    p.after_forward_parking_straight_distance =
      declare_parameter<double>(ns + "after_forward_parking_straight_distance");
    p.after_backward_parking_straight_distance =
      declare_parameter<double>(ns + "after_backward_parking_straight_distance");
    p.forward_parking_velocity = declare_parameter<double>(ns + "forward_parking_velocity");
    p.backward_parking_velocity = declare_parameter<double>(ns + "backward_parking_velocity");
    p.forward_parking_lane_departure_margin =
      declare_parameter<double>(ns + "forward_parking_lane_departure_margin");
    p.backward_parking_lane_departure_margin =
      declare_parameter<double>(ns + "backward_parking_lane_departure_margin");
    p.arc_path_interval = declare_parameter<double>(ns + "arc_path_interval");
    p.pull_over_max_steer_angle =
      declare_parameter<double>(ns + "pull_over_max_steer_angle");  // 20deg
                                                                    // freespace parking
    p.enable_freespace_parking = declare_parameter<bool>(ns + "enable_freespace_parking");
    // hazard
    p.hazard_on_threshold_distance = declare_parameter<double>(ns + "hazard_on_threshold_distance");
    p.hazard_on_threshold_velocity = declare_parameter<double>(ns + "hazard_on_threshold_velocity");
    // safety with dynamic objects. Not used now.
    p.pull_over_duration = declare_parameter<double>(ns + "pull_over_duration");
    p.pull_over_prepare_duration = declare_parameter<double>(ns + "pull_over_prepare_duration");
    p.min_stop_distance = declare_parameter<double>(ns + "min_stop_distance");
    p.stop_time = declare_parameter<double>(ns + "stop_time");
    p.hysteresis_buffer_distance = declare_parameter<double>(ns + "hysteresis_buffer_distance");
    p.prediction_time_resolution = declare_parameter<double>(ns + "prediction_time_resolution");
    p.enable_collision_check_at_prepare_phase =
      declare_parameter<bool>(ns + "enable_collision_check_at_prepare_phase");
    p.use_predicted_path_outside_lanelet =
      declare_parameter<bool>(ns + "use_predicted_path_outside_lanelet");
    p.use_all_predicted_path = declare_parameter<bool>(ns + "use_all_predicted_path");
    // drivable area
    p.drivable_area_right_bound_offset =
      declare_parameter<double>(ns + "drivable_area_right_bound_offset");
    p.drivable_area_left_bound_offset =
      declare_parameter<double>(ns + "drivable_area_left_bound_offset");
    p.drivable_area_types_to_skip =
      declare_parameter<std::vector<std::string>>(ns + "drivable_area_types_to_skip");
    // debug
    p.print_debug_info = declare_parameter<bool>(ns + "print_debug_info");

    // validation of parameters
    if (p.pull_over_sampling_num < 1) {
      RCLCPP_FATAL_STREAM(
        get_logger(), "pull_over_sampling_num must be positive integer. Given parameter: "
                        << p.pull_over_sampling_num << std::endl
                        << "Terminating the program...");
      exit(EXIT_FAILURE);
    }
    if (p.maximum_deceleration < 0.0) {
      RCLCPP_FATAL_STREAM(
        get_logger(), "maximum_deceleration cannot be negative value. Given parameter: "
                        << p.maximum_deceleration << std::endl
                        << "Terminating the program...");
      exit(EXIT_FAILURE);
    }
  }

  {
    std::string ns = "pull_over.freespace_parking.";
    // search configs
    p.algorithm = declare_parameter<std::string>(ns + "planning_algorithm");
    p.freespace_parking_velocity = declare_parameter<double>(ns + "velocity");
    p.vehicle_shape_margin = declare_parameter<double>(ns + "vehicle_shape_margin");
    p.common_parameters.time_limit = declare_parameter<double>(ns + "time_limit");
    p.common_parameters.minimum_turning_radius =
      declare_parameter<double>(ns + "minimum_turning_radius");
    p.common_parameters.maximum_turning_radius =
      declare_parameter<double>(ns + "maximum_turning_radius");
    p.common_parameters.turning_radius_size = declare_parameter<int>(ns + "turning_radius_size");
    p.common_parameters.maximum_turning_radius = std::max(
      p.common_parameters.maximum_turning_radius, p.common_parameters.minimum_turning_radius);
    p.common_parameters.turning_radius_size = std::max(p.common_parameters.turning_radius_size, 1);

    p.common_parameters.theta_size = declare_parameter<int>(ns + "theta_size");
    p.common_parameters.angle_goal_range = declare_parameter<double>(ns + "angle_goal_range");

    p.common_parameters.curve_weight = declare_parameter<double>(ns + "curve_weight");
    p.common_parameters.reverse_weight = declare_parameter<double>(ns + "reverse_weight");
    p.common_parameters.lateral_goal_range = declare_parameter<double>(ns + "lateral_goal_range");
    p.common_parameters.longitudinal_goal_range =
      declare_parameter<double>(ns + "longitudinal_goal_range");

    // costmap configs
    p.common_parameters.obstacle_threshold = declare_parameter<int>(ns + "obstacle_threshold");
  }

  {
    std::string ns = "pull_over.freespace_parking.astar.";
    p.astar_parameters.only_behind_solutions =
      declare_parameter<bool>(ns + "only_behind_solutions");
    p.astar_parameters.use_back = declare_parameter<bool>(ns + "use_back");
    p.astar_parameters.distance_heuristic_weight =
      declare_parameter<double>(ns + "distance_heuristic_weight");
  }

  {
    std::string ns = "pull_over.freespace_parking.rrtstar.";
    p.rrt_star_parameters.enable_update = declare_parameter<bool>(ns + "enable_update");
    p.rrt_star_parameters.use_informed_sampling =
      declare_parameter<bool>(ns + "use_informed_sampling");
    p.rrt_star_parameters.max_planning_time = declare_parameter<double>(ns + "max_planning_time");
    p.rrt_star_parameters.neighbor_radius = declare_parameter<double>(ns + "neighbor_radius");
    p.rrt_star_parameters.margin = declare_parameter<double>(ns + "margin");
  }

  return p;
}

PullOutParameters BehaviorPathPlannerNode::getPullOutParam()
{
  const auto dp = [this](const std::string & str, auto def_val) {
    std::string name = "pull_out." + str;
    return this->declare_parameter(name, def_val);
  };

  PullOutParameters p;

  p.th_arrived_distance = dp("th_arrived_distance", 1.0);
  p.th_stopped_velocity = dp("th_stopped_velocity", 0.01);
  p.th_stopped_time = dp("th_stopped_time", 1.0);
  p.collision_check_margin = dp("collision_check_margin", 1.0);
  p.collision_check_distance_from_end = dp("collision_check_distance_from_end", 3.0);
  // shift pull out
  p.enable_shift_pull_out = dp("enable_shift_pull_out", true);
  p.shift_pull_out_velocity = dp("shift_pull_out_velocity", 8.3);
  p.pull_out_sampling_num = dp("pull_out_sampling_num", 4);
  p.minimum_shift_pull_out_distance = dp("minimum_shift_pull_out_distance", 20.0);
  p.maximum_lateral_jerk = dp("maximum_lateral_jerk", 3.0);
  p.minimum_lateral_jerk = dp("minimum_lateral_jerk", 1.0);
  p.deceleration_interval = dp("deceleration_interval", 10.0);
  // geometric pull out
  p.enable_geometric_pull_out = dp("enable_geometric_pull_out", true);
  p.divide_pull_out_path = dp("divide_pull_out_path", false);
  p.geometric_pull_out_velocity = dp("geometric_pull_out_velocity", 1.0);
  p.arc_path_interval = dp("arc_path_interval", 1.0);
  p.lane_departure_margin = dp("lane_departure_margin", 0.2);
  p.backward_velocity = dp("backward_velocity", -0.3);
  p.pull_out_max_steer_angle = dp("pull_out_max_steer_angle", 0.26);  // 15deg
  // search start pose backward
  p.search_priority =
    dp("search_priority", "efficient_path");  // "efficient_path" or "short_back_distance"
  p.enable_back = dp("enable_back", true);
  p.max_back_distance = dp("max_back_distance", 30.0);
  p.backward_search_resolution = dp("backward_search_resolution", 2.0);
  p.backward_path_update_duration = dp("backward_path_update_duration", 3.0);
  p.ignore_distance_from_lane_end = dp("ignore_distance_from_lane_end", 15.0);
  // drivable area
  p.drivable_area_right_bound_offset = dp("drivable_area_right_bound_offset", 0.0);
  p.drivable_area_left_bound_offset = dp("drivable_area_left_bound_offset", 0.0);
  p.drivable_area_types_to_skip =
    dp("drivable_area_types_to_skip", std::vector<std::string>({"road_border"}));

  // validation of parameters
  if (p.pull_out_sampling_num < 1) {
    RCLCPP_FATAL_STREAM(
      get_logger(), "pull_out_sampling_num must be positive integer. Given parameter: "
                      << p.pull_out_sampling_num << std::endl
                      << "Terminating the program...");
    exit(EXIT_FAILURE);
  }

  return p;
}

#ifdef USE_OLD_ARCHITECTURE
BehaviorTreeManagerParam BehaviorPathPlannerNode::getBehaviorTreeManagerParam()
{
  BehaviorTreeManagerParam p{};
  p.bt_tree_config_path = declare_parameter("bt_tree_config_path", "default");
  p.groot_zmq_publisher_port = declare_parameter("groot_zmq_publisher_port", 1666);
  p.groot_zmq_server_port = declare_parameter("groot_zmq_server_port", 1667);
  return p;
}
#endif

// wait until mandatory data is ready
bool BehaviorPathPlannerNode::isDataReady()
{
  const auto missing = [this](const auto & name) {
    RCLCPP_INFO_SKIPFIRST_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for %s", name);
    return false;
  };

  if (!current_scenario_) {
    return missing("scenario_topic");
  }

  if (!route_ptr_) {
    return missing("route");
  }

  if (!map_ptr_) {
    return missing("map");
  }

  if (!planner_data_->dynamic_object) {
    return missing("dynamic_object");
  }

  if (!planner_data_->self_odometry) {
    return missing("self_odometry");
  }

  if (!planner_data_->self_acceleration) {
    return missing("self_acceleration");
  }

  if (!planner_data_->operation_mode) {
    return missing("operation_mode");
  }

  return true;
}

std::shared_ptr<PlannerData> BehaviorPathPlannerNode::createLatestPlannerData()
{
  const std::lock_guard<std::mutex> lock(mutex_pd_);

  // update map
  if (has_received_map_) {
    planner_data_->route_handler->setMap(*map_ptr_);
    has_received_map_ = false;
  }

  // update route
  const bool is_first_time = !(planner_data_->route_handler->isHandlerReady());
  if (has_received_route_) {
    planner_data_->route_handler->setRoute(*route_ptr_);
    // Reset behavior tree when new route is received,
    // so that the each modules do not have to care about the "route jump".
    if (!is_first_time) {
      RCLCPP_DEBUG(get_logger(), "new route is received. reset behavior tree.");
#ifdef USE_OLD_ARCHITECTURE
      bt_manager_->resetBehaviorTree();
#else
      planner_manager_->reset();
#endif
    }

    has_received_route_ = false;
  }

  return std::make_shared<PlannerData>(*planner_data_);
}

void BehaviorPathPlannerNode::run()
{
  if (!isDataReady()) {
    return;
  }

  RCLCPP_DEBUG(get_logger(), "----- BehaviorPathPlannerNode start -----");
  mutex_bt_.lock();  // for bt_manager_

  // behavior_path_planner runs only in LANE DRIVING scenario.
  if (current_scenario_->current_scenario != Scenario::LANEDRIVING) {
    mutex_bt_.unlock();  // for bt_manager_
    return;
  }

  // create latest planner data
  const auto planner_data = createLatestPlannerData();

#ifndef USE_OLD_ARCHITECTURE
  if (planner_data->operation_mode->mode != OperationModeState::AUTONOMOUS) {
    planner_manager_->resetRootLanelet(planner_data);
  }
#endif

  // run behavior planner
#ifdef USE_OLD_ARCHITECTURE
  const auto output = bt_manager_->run(planner_data);
#else
  const auto output = planner_manager_->run(planner_data);
#endif

  // path handling
  const auto path = getPath(output, planner_data);

  // update planner data
  planner_data_->prev_output_path = path;

  // compute turn signal
  computeTurnSignal(planner_data, *path, output);

  // publish drivable bounds
  publish_bounds(*path);

  const size_t target_idx = planner_data->findEgoIndex(path->points);
  util::clipPathLength(*path, target_idx, planner_data_->parameters);

  if (!path->points.empty()) {
    path_publisher_->publish(*path);
  } else {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000, "behavior path output is empty! Stop publish.");
  }

#ifdef USE_OLD_ARCHITECTURE
  publishPathCandidate(bt_manager_->getSceneModules());
#else
  publishPathCandidate(planner_manager_->getSceneModuleManagers());
  publishPathReference(planner_manager_->getSceneModuleManagers());
#endif

  publishSceneModuleDebugMsg();

  if (output.modified_goal) {
    PoseWithUuidStamped modified_goal = *(output.modified_goal);
    modified_goal.header.stamp = path->header.stamp;
    modified_goal_publisher_->publish(modified_goal);
  }

  if (planner_data->parameters.visualize_maximum_drivable_area) {
    const auto maximum_drivable_area =
      marker_utils::createFurthestLineStringMarkerArray(util::getMaximumDrivableArea(planner_data));
    debug_maximum_drivable_area_publisher_->publish(maximum_drivable_area);
  }

#ifndef USE_OLD_ARCHITECTURE
  planner_manager_->print();
  planner_manager_->publishDebugMarker();
#endif

  mutex_bt_.unlock();
  RCLCPP_DEBUG(get_logger(), "----- behavior path planner end -----\n\n");
}

void BehaviorPathPlannerNode::computeTurnSignal(
  const std::shared_ptr<PlannerData> planner_data, const PathWithLaneId & path,
  const BehaviorModuleOutput & output)
{
  TurnIndicatorsCommand turn_signal;
  HazardLightsCommand hazard_signal;
  if (output.turn_signal_info.hazard_signal.command == HazardLightsCommand::ENABLE) {
    turn_signal.command = TurnIndicatorsCommand::DISABLE;
    hazard_signal.command = output.turn_signal_info.hazard_signal.command;
  } else {
    turn_signal = turn_signal_decider_.getTurnSignal(planner_data, path, output.turn_signal_info);
    hazard_signal.command = HazardLightsCommand::DISABLE;
  }
  turn_signal.stamp = get_clock()->now();
  hazard_signal.stamp = get_clock()->now();
  turn_signal_publisher_->publish(turn_signal);
  hazard_signal_publisher_->publish(hazard_signal);

  publish_steering_factor(turn_signal);
}

void BehaviorPathPlannerNode::publish_steering_factor(const TurnIndicatorsCommand & turn_signal)
{
  const auto [intersection_flag, approaching_intersection_flag] =
    turn_signal_decider_.getIntersectionTurnSignalFlag();
  if (intersection_flag || approaching_intersection_flag) {
    const uint16_t steering_factor_direction = std::invoke([&turn_signal]() {
      if (turn_signal.command == TurnIndicatorsCommand::ENABLE_LEFT) {
        return SteeringFactor::LEFT;
      }
      return SteeringFactor::RIGHT;
    });

    const auto [intersection_pose, intersection_distance] =
      turn_signal_decider_.getIntersectionPoseAndDistance();
    const uint16_t steering_factor_state = std::invoke([&intersection_flag]() {
      if (intersection_flag) {
        return SteeringFactor::TURNING;
      }
      return SteeringFactor::TRYING;
    });

    steering_factor_interface_ptr_->updateSteeringFactor(
      {intersection_pose, intersection_pose}, {intersection_distance, intersection_distance},
      SteeringFactor::INTERSECTION, steering_factor_direction, steering_factor_state, "");
  } else {
    steering_factor_interface_ptr_->clearSteeringFactors();
  }
  steering_factor_interface_ptr_->publishSteeringFactor(get_clock()->now());
}

void BehaviorPathPlannerNode::publish_bounds(const PathWithLaneId & path)
{
  constexpr double scale_x = 0.2;
  constexpr double scale_y = 0.2;
  constexpr double scale_z = 0.2;
  constexpr double color_r = 0.0 / 256.0;
  constexpr double color_g = 148.0 / 256.0;
  constexpr double color_b = 205.0 / 256.0;
  constexpr double color_a = 0.999;

  const auto current_time = path.header.stamp;
  auto left_marker = tier4_autoware_utils::createDefaultMarker(
    "map", current_time, "left_bound", 0L, Marker::LINE_STRIP,
    tier4_autoware_utils::createMarkerScale(scale_x, scale_y, scale_z),
    tier4_autoware_utils::createMarkerColor(color_r, color_g, color_b, color_a));
  for (const auto lb : path.left_bound) {
    left_marker.points.push_back(lb);
  }

  auto right_marker = tier4_autoware_utils::createDefaultMarker(
    "map", current_time, "right_bound", 0L, Marker::LINE_STRIP,
    tier4_autoware_utils::createMarkerScale(scale_x, scale_y, scale_z),
    tier4_autoware_utils::createMarkerColor(color_r, color_g, color_b, color_a));
  for (const auto rb : path.right_bound) {
    right_marker.points.push_back(rb);
  }

  MarkerArray msg;
  msg.markers.push_back(left_marker);
  msg.markers.push_back(right_marker);
  bound_publisher_->publish(msg);
}

void BehaviorPathPlannerNode::publishSceneModuleDebugMsg()
{
#ifdef USE_OLD_ARCHITECTURE
  const auto debug_messages_data_ptr = bt_manager_->getAllSceneModuleDebugMsgData();

  const auto avoidance_debug_message = debug_messages_data_ptr->getAvoidanceModuleDebugMsg();
  if (avoidance_debug_message) {
    debug_avoidance_msg_array_publisher_->publish(*avoidance_debug_message);
  }

  const auto lane_change_debug_message = debug_messages_data_ptr->getLaneChangeModuleDebugMsg();
  if (lane_change_debug_message) {
    debug_lane_change_msg_array_publisher_->publish(*lane_change_debug_message);
  }
#endif
}

#ifdef USE_OLD_ARCHITECTURE
void BehaviorPathPlannerNode::publishPathCandidate(
  const std::vector<std::shared_ptr<SceneModuleInterface>> & scene_modules)
{
  for (auto & module : scene_modules) {
    if (path_candidate_publishers_.count(module->name()) != 0) {
      path_candidate_publishers_.at(module->name())
        ->publish(convertToPath(module->getPathCandidate(), module->isExecutionReady()));
    }
  }
}
#else
void BehaviorPathPlannerNode::publishPathCandidate(
  const std::vector<std::shared_ptr<SceneModuleManagerInterface>> & managers)
{
  for (auto & manager : managers) {
    if (path_candidate_publishers_.count(manager->getModuleName()) == 0) {
      continue;
    }

    if (manager->getSceneModules().empty()) {
      path_candidate_publishers_.at(manager->getModuleName())
        ->publish(convertToPath(nullptr, false));
      continue;
    }

    for (auto & module : manager->getSceneModules()) {
      path_candidate_publishers_.at(module->name())
        ->publish(convertToPath(module->getPathCandidate(), module->isExecutionReady()));
    }
  }
}

void BehaviorPathPlannerNode::publishPathReference(
  const std::vector<std::shared_ptr<SceneModuleManagerInterface>> & managers)
{
  for (auto & manager : managers) {
    if (path_reference_publishers_.count(manager->getModuleName()) == 0) {
      continue;
    }

    if (manager->getSceneModules().empty()) {
      path_reference_publishers_.at(manager->getModuleName())
        ->publish(convertToPath(nullptr, false));
      continue;
    }

    for (auto & module : manager->getSceneModules()) {
      path_reference_publishers_.at(module->name())
        ->publish(convertToPath(module->getPathReference(), true));
    }
  }
}
#endif

Path BehaviorPathPlannerNode::convertToPath(
  const std::shared_ptr<PathWithLaneId> & path_candidate_ptr, const bool is_ready)
{
  Path output;
  output.header = planner_data_->route_handler->getRouteHeader();
  output.header.stamp = this->now();

  if (!path_candidate_ptr) {
    return output;
  }

  output = util::toPath(*path_candidate_ptr);
  // header is replaced by the input one, so it is substituted again
  output.header = planner_data_->route_handler->getRouteHeader();
  output.header.stamp = this->now();

  if (!is_ready) {
    for (auto & point : output.points) {
      point.longitudinal_velocity_mps = 0.0;
    }
  }

  return output;
}

PathWithLaneId::SharedPtr BehaviorPathPlannerNode::getPath(
  const BehaviorModuleOutput & bt_output, const std::shared_ptr<PlannerData> planner_data)
{
  // TODO(Horibe) do some error handling when path is not available.

  auto path = bt_output.path ? bt_output.path : planner_data->prev_output_path;
  path->header = planner_data->route_handler->getRouteHeader();
  path->header.stamp = this->now();
  RCLCPP_DEBUG(
    get_logger(), "BehaviorTreeManager: output is %s.", bt_output.path ? "FOUND" : "NOT FOUND");

  PathWithLaneId connected_path;
#ifdef USE_OLD_ARCHITECTURE
  const auto module_status_ptr_vec = bt_manager_->getModulesStatus();
#else
  const auto module_status_ptr_vec = planner_manager_->getSceneModuleStatus();
#endif
  if (skipSmoothGoalConnection(module_status_ptr_vec)) {
    connected_path = *path;
  } else {
    connected_path = modifyPathForSmoothGoalConnection(*path);
  }

  const auto resampled_path = util::resamplePathWithSpline(
    connected_path, planner_data_->parameters.path_interval,
    keepInputPoints(module_status_ptr_vec));
  return std::make_shared<PathWithLaneId>(resampled_path);
}

bool BehaviorPathPlannerNode::skipSmoothGoalConnection(
  const std::vector<std::shared_ptr<SceneModuleStatus>> & statuses) const
{
#ifdef USE_OLD_ARCHITECTURE
  const auto target_module = "PullOver";
#else
  const auto target_module = "pull_over";
#endif

  const auto target_status = ModuleStatus::RUNNING;

  for (auto & status : statuses) {
    if (status->is_waiting_approval || status->status == target_status) {
      if (target_module == status->module_name) {
        return true;
      }
    }
  }
  return false;
}

// This is a temporary process until motion planning can take the terminal pose into account
bool BehaviorPathPlannerNode::keepInputPoints(
  const std::vector<std::shared_ptr<SceneModuleStatus>> & statuses) const
{
#ifdef USE_OLD_ARCHITECTURE
  const std::vector<std::string> target_modules = {"PullOver", "Avoidance"};
#else
  const std::vector<std::string> target_modules = {"pull_over", "avoidance"};
#endif

  const auto target_status = ModuleStatus::RUNNING;

  for (auto & status : statuses) {
    if (status->is_waiting_approval || status->status == target_status) {
      if (
        std::find(target_modules.begin(), target_modules.end(), status->module_name) !=
        target_modules.end()) {
        return true;
      }
    }
  }
  return false;
}

void BehaviorPathPlannerNode::onOdometry(const Odometry::ConstSharedPtr msg)
{
  const std::lock_guard<std::mutex> lock(mutex_pd_);
  planner_data_->self_odometry = msg;
}
void BehaviorPathPlannerNode::onAcceleration(const AccelWithCovarianceStamped::ConstSharedPtr msg)
{
  const std::lock_guard<std::mutex> lock(mutex_pd_);
  planner_data_->self_acceleration = msg;
}
void BehaviorPathPlannerNode::onPerception(const PredictedObjects::ConstSharedPtr msg)
{
  const std::lock_guard<std::mutex> lock(mutex_pd_);
  planner_data_->dynamic_object = msg;
}
void BehaviorPathPlannerNode::onOccupancyGrid(const OccupancyGrid::ConstSharedPtr msg)
{
  const std::lock_guard<std::mutex> lock(mutex_pd_);
  planner_data_->occupancy_grid = msg;
}
void BehaviorPathPlannerNode::onCostMap(const OccupancyGrid::ConstSharedPtr msg)
{
  const std::lock_guard<std::mutex> lock(mutex_pd_);
  planner_data_->costmap = msg;
}
void BehaviorPathPlannerNode::onMap(const HADMapBin::ConstSharedPtr msg)
{
  const std::lock_guard<std::mutex> lock(mutex_pd_);
  map_ptr_ = msg;
  has_received_map_ = true;
}
void BehaviorPathPlannerNode::onRoute(const LaneletRoute::ConstSharedPtr msg)
{
  const std::lock_guard<std::mutex> lock(mutex_pd_);
  route_ptr_ = msg;
  has_received_route_ = true;
}
void BehaviorPathPlannerNode::onOperationMode(const OperationModeState::ConstSharedPtr msg)
{
  const std::lock_guard<std::mutex> lock(mutex_pd_);
  planner_data_->operation_mode = msg;
}
void BehaviorPathPlannerNode::onLateralOffset(const LateralOffset::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_pd_);

  if (!planner_data_->lateral_offset) {
    planner_data_->lateral_offset = msg;
    return;
  }

  const auto & new_offset = msg->lateral_offset;
  const auto & old_offset = planner_data_->lateral_offset->lateral_offset;

  // offset is not changed.
  if (std::abs(old_offset - new_offset) < 1e-4) {
    return;
  }

  planner_data_->lateral_offset = msg;
}

SetParametersResult BehaviorPathPlannerNode::onSetParam(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;

  if (!lane_change_param_ptr_ && !avoidance_param_ptr_) {
    result.successful = false;
    result.reason = "param not initialized";
    return result;
  }

  result.successful = true;
  result.reason = "success";

  try {
    update_param(
      parameters, "avoidance.publish_debug_marker", avoidance_param_ptr_->publish_debug_marker);
    update_param(
      parameters, "lane_change.publish_debug_marker", lane_change_param_ptr_->publish_debug_marker);
    // Drivable area expansion parameters
    using drivable_area_expansion::DrivableAreaExpansionParameters;
    update_param(
      parameters, DrivableAreaExpansionParameters::ENABLED_PARAM,
      planner_data_->drivable_area_expansion_parameters.enabled);
    update_param(
      parameters, DrivableAreaExpansionParameters::AVOID_DYN_OBJECTS_PARAM,
      planner_data_->drivable_area_expansion_parameters.avoid_dynamic_objects);
    update_param(
      parameters, DrivableAreaExpansionParameters::EXPANSION_METHOD_PARAM,
      planner_data_->drivable_area_expansion_parameters.expansion_method);
    update_param(
      parameters, DrivableAreaExpansionParameters::AVOID_LINESTRING_TYPES_PARAM,
      planner_data_->drivable_area_expansion_parameters.avoid_linestring_types);
    update_param(
      parameters, DrivableAreaExpansionParameters::AVOID_LINESTRING_DIST_PARAM,
      planner_data_->drivable_area_expansion_parameters.avoid_linestring_dist);
    update_param(
      parameters, DrivableAreaExpansionParameters::EGO_EXTRA_OFFSET_FRONT,
      planner_data_->drivable_area_expansion_parameters.ego_extra_front_offset);
    update_param(
      parameters, DrivableAreaExpansionParameters::EGO_EXTRA_OFFSET_REAR,
      planner_data_->drivable_area_expansion_parameters.ego_extra_rear_offset);
    update_param(
      parameters, DrivableAreaExpansionParameters::EGO_EXTRA_OFFSET_LEFT,
      planner_data_->drivable_area_expansion_parameters.ego_extra_left_offset);
    update_param(
      parameters, DrivableAreaExpansionParameters::EGO_EXTRA_OFFSET_RIGHT,
      planner_data_->drivable_area_expansion_parameters.ego_extra_right_offset);
    update_param(
      parameters, DrivableAreaExpansionParameters::DYN_OBJECTS_EXTRA_OFFSET_FRONT,
      planner_data_->drivable_area_expansion_parameters.dynamic_objects_extra_front_offset);
    update_param(
      parameters, DrivableAreaExpansionParameters::DYN_OBJECTS_EXTRA_OFFSET_REAR,
      planner_data_->drivable_area_expansion_parameters.dynamic_objects_extra_rear_offset);
    update_param(
      parameters, DrivableAreaExpansionParameters::DYN_OBJECTS_EXTRA_OFFSET_LEFT,
      planner_data_->drivable_area_expansion_parameters.dynamic_objects_extra_left_offset);
    update_param(
      parameters, DrivableAreaExpansionParameters::DYN_OBJECTS_EXTRA_OFFSET_RIGHT,
      planner_data_->drivable_area_expansion_parameters.dynamic_objects_extra_right_offset);
    update_param(
      parameters, DrivableAreaExpansionParameters::MAX_EXP_DIST_PARAM,
      planner_data_->drivable_area_expansion_parameters.max_expansion_distance);
    update_param(
      parameters, DrivableAreaExpansionParameters::MAX_PATH_ARC_LENGTH_PARAM,
      planner_data_->drivable_area_expansion_parameters.max_path_arc_length);
    update_param(
      parameters, DrivableAreaExpansionParameters::EXTRA_ARC_LENGTH_PARAM,
      planner_data_->drivable_area_expansion_parameters.extra_arc_length);
    update_param(
      parameters, DrivableAreaExpansionParameters::COMPENSATE_PARAM,
      planner_data_->drivable_area_expansion_parameters.compensate_uncrossable_lines);
    update_param(
      parameters, DrivableAreaExpansionParameters::EXTRA_COMPENSATE_PARAM,
      planner_data_->drivable_area_expansion_parameters.compensate_extra_dist);
  } catch (const rclcpp::exceptions::InvalidParameterTypeException & e) {
    result.successful = false;
    result.reason = e.what();
  }

  return result;
}

PathWithLaneId BehaviorPathPlannerNode::modifyPathForSmoothGoalConnection(
  const PathWithLaneId & path) const
{
  const auto goal = planner_data_->route_handler->getGoalPose();
  const auto goal_lane_id = planner_data_->route_handler->getGoalLaneId();

  Pose refined_goal{};
  {
    lanelet::ConstLanelet goal_lanelet;
    if (planner_data_->route_handler->getGoalLanelet(&goal_lanelet)) {
      refined_goal = util::refineGoal(goal, goal_lanelet);
    } else {
      refined_goal = goal;
    }
  }

  auto refined_path = util::refinePathForGoal(
    planner_data_->parameters.refine_goal_search_radius_range, M_PI * 0.5, path, refined_goal,
    goal_lane_id);
  refined_path.header.frame_id = "map";
  refined_path.header.stamp = this->now();

  return refined_path;
}
}  // namespace behavior_path_planner

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(behavior_path_planner::BehaviorPathPlannerNode)

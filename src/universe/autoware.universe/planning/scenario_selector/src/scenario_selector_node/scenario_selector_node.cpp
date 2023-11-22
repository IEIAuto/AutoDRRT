// // Copyright 2020 Tier IV, Inc.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// #include "scenario_selector/scenario_selector_node.hpp"

// #include <lanelet2_extension/utility/message_conversion.hpp>
// #include <lanelet2_extension/utility/query.hpp>

// #include <lanelet2_core/geometry/BoundingBox.h>
// #include <lanelet2_core/geometry/Lanelet.h>
// #include <lanelet2_core/geometry/LineString.h>
// #include <lanelet2_core/geometry/Point.h>
// #include <lanelet2_core/geometry/Polygon.h>

// #include <deque>
// #include <memory>
// #include <string>
// #include <utility>
// #include <vector>

// namespace
// {
// template <class T>
// void onData(const T & data, T * buffer)
// {
//   *buffer = data;
// }

// using AutowareState = autoware_auto_system_msgs::msg::AutowareState;
// using Engage = autoware_auto_vehicle_msgs::msg::Engage;
// using PoseStamped = geometry_msgs::msg::PoseStamped;
// Engage createEngageMessage()
// {
//   auto msg = Engage();
//   msg.engage = true;
//   return msg;
// }
// // std::shared_ptr<lanelet::ConstPolygon3d> findNearestParkinglot(
// //   const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
// //   const lanelet::BasicPoint2d & current_position)
// // {
// //   const auto all_parking_lots = lanelet::utils::query::getAllParkingLots(lanelet_map_ptr);

// //   const auto linked_parking_lot = std::make_shared<lanelet::ConstPolygon3d>();
// //   const auto result = lanelet::utils::query::getLinkedParkingLot(
// //     current_position, all_parking_lots, linked_parking_lot.get());

// //   if (result) {
// //     return linked_parking_lot;
// //   } else {
// //     return {};
// //   }
// // }

// geometry_msgs::msg::PoseStamped::ConstSharedPtr getCurrentPose(
//   const tf2_ros::Buffer & tf_buffer, const rclcpp::Logger & logger)
// {
//   geometry_msgs::msg::TransformStamped tf_current_pose;

//   try {
//     tf_current_pose = tf_buffer.lookupTransform("map", "base_link", tf2::TimePointZero);
//   } catch (tf2::TransformException & ex) {
//     RCLCPP_ERROR(logger, "%s", ex.what());
//     return nullptr;
//   }

//   geometry_msgs::msg::PoseStamped::SharedPtr p(new geometry_msgs::msg::PoseStamped());
//   p->header = tf_current_pose.header;
//   p->pose.orientation = tf_current_pose.transform.rotation;
//   p->pose.position.x = tf_current_pose.transform.translation.x;
//   p->pose.position.y = tf_current_pose.transform.translation.y;
//   p->pose.position.z = tf_current_pose.transform.translation.z;

//   return geometry_msgs::msg::PoseStamped::ConstSharedPtr(p);
// }

// // bool isInLane(
// //   const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
// //   const geometry_msgs::msg::Point & current_pos)
// // {
// //   const auto & p = current_pos;
// //   const lanelet::Point3d search_point(lanelet::InvalId, p.x, p.y, p.z);

// //   std::vector<std::pair<double, lanelet::Lanelet>> nearest_lanelets =
// //     lanelet::geometry::findNearest(lanelet_map_ptr->laneletLayer, search_point.basicPoint2d(), 1);

// //   if (nearest_lanelets.empty()) {
// //     return false;
// //   }

// //   const auto nearest_lanelet = nearest_lanelets.front().second;

// //   return lanelet::geometry::within(search_point, nearest_lanelet.polygon3d());
// // }

// // bool isInParkingLot(
// //   const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
// //   const geometry_msgs::msg::Pose & current_pose)
// // {
// //   const auto & p = current_pose.position;
// //   const lanelet::Point3d search_point(lanelet::InvalId, p.x, p.y, p.z);

// //   const auto nearest_parking_lot =
// //     findNearestParkinglot(lanelet_map_ptr, search_point.basicPoint2d());

// //   if (!nearest_parking_lot) {
// //     return false;
// //   }

// //   return lanelet::geometry::within(search_point, nearest_parking_lot->basicPolygon());
// // }



// bool isNearTrajectoryEnd(
//   const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr trajectory,
//   const geometry_msgs::msg::Pose & current_pose, const double th_dist)
// {
//   if (!trajectory || trajectory->points.empty()) {
//     return false;
//   }

//   const auto & p1 = current_pose.position;
//   const auto & p2 = trajectory->points.back().pose.position;

//   const auto dist = std::hypot(p1.x - p2.x, p1.y - p2.y);

//   return dist < th_dist;
// }

// bool isStopped(
//   const std::deque<geometry_msgs::msg::TwistStamped::ConstSharedPtr> & twist_buffer,
//   const double th_stopped_velocity_mps)
// {
//   for (const auto & twist : twist_buffer) {
//     if (std::abs(twist->twist.linear.x) > th_stopped_velocity_mps) {
//       return false;
//     }
//   }
//   return true;
// }

// }  // namespace

// size_t ScenarioSelectorNode::arrivedWhichGoal(const geometry_msgs::msg::Pose & current_pose, const double th_dist)
//   {
//     double min_dis = 100.0;
//     double min_ang = 100.0;
//     size_t min_index = 0;
//     for(size_t index = 0; index < this->pre_seted_pose_vector.size(); index++)
//     {
//       const auto & p1 = current_pose.position;
//       const auto & p2 = this->pre_seted_pose_vector[index].pose.position;
//       const auto dist = std::hypot(p1.x - p2.x, p1.y - p2.y);
//       const auto & o1 = current_pose.orientation;
//       const auto & o2 = this->pre_seted_pose_vector[index].pose.orientation;
//       double dotProduct = o1.x * o2.x + o1.y * o2.y + o1.z * o2.z + o1.w * o2.w;
//       double angleDifference = 2.0 * std::acos(std::abs(dotProduct));
//       RCLCPP_INFO(this->get_logger(), "goal %ld dis is %f, ang is %f",index,dist,angleDifference);
//       if(dist <= th_dist*1.5 && angleDifference <= 0.3)
//       {
//         if(min_dis > dist)
//         {
//           min_dis = dist;
//           if(min_ang > angleDifference)
//           {
//             min_ang = angleDifference;
//             min_index = index;
//           }
          
//         }
//       }
//     }
//     if(std::fabs(min_dis - 100.00) < 0.0001)
//     {
//       return 60;
//     }
//     else{
//       return min_index;
//     }
    
//   }


// autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr
// ScenarioSelectorNode::getScenarioTrajectory(const std::string & scenario)
// {
//   if (scenario == tier4_planning_msgs::msg::Scenario::LANEDRIVING) {
//     return lane_driving_trajectory_;
//   }
//   if (scenario == tier4_planning_msgs::msg::Scenario::PARKING) {
//     return parking_trajectory_;
//   }
//   RCLCPP_ERROR_STREAM(this->get_logger(), "invalid scenario argument: " << scenario);
//   return lane_driving_trajectory_;
// }

// // std::string ScenarioSelectorNode::selectScenarioByPosition()
// // {
// //   return tier4_planning_msgs::msg::Scenario::PARKING;
// //   const auto is_in_lane = isInLane(lanelet_map_ptr_, current_pose_->pose.position);
// //   const auto is_goal_in_lane = isInLane(lanelet_map_ptr_, route_->goal_pose.position);
// //   const auto is_in_parking_lot = isInParkingLot(lanelet_map_ptr_, current_pose_->pose);

// //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::EMPTY) {
// //     if (is_in_lane && is_goal_in_lane) {
// //       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
// //     } else if (is_in_parking_lot) {
// //       return tier4_planning_msgs::msg::Scenario::PARKING;
// //     } else {
// //       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
// //     }
// //   }

// //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::LANEDRIVING) {
// //     if (is_in_parking_lot && !is_goal_in_lane) {
// //       return tier4_planning_msgs::msg::Scenario::PARKING;
// //     }
// //   }

// //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::PARKING) {
// //     if (is_parking_completed_ && is_in_lane) {
// //       is_parking_completed_ = false;
// //       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
// //     }
// //   }

// //   return current_scenario_;
// // }

// std::string ScenarioSelectorNode::selectScenarioByPosition()
// {
//   // return tier4_planning_msgs::msg::Scenario::PARKING;
//   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::EMPTY) {
//       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
//   }

//   size_t goal_index = arrivedWhichGoal(current_pose_->pose, th_arrived_distance_m_);
//   RCLCPP_INFO(this->get_logger(),"goal_index is %ld",goal_index);
//   switch (goal_index)
//   {
//     case 0:
//     this->drive_stage = goal_index;
//     return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
    
//     break;

//     case 1:
//     this->drive_stage = goal_index;
//     // this->change_freespace_param(0);
//     return tier4_planning_msgs::msg::Scenario::PARKING;
    
//     break;

//     case 2:
//     this->drive_stage = goal_index;
//     return tier4_planning_msgs::msg::Scenario::PARKING;
    
//     break;

//     case 3:
//     this->drive_stage = goal_index;
//     // this->change_freespace_param(3);
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//     break;
//     case 4:
//     this->drive_stage = goal_index;
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//     break;
//     case 5:
//     this->drive_stage = goal_index;
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//      break;
//     case 6:
//     this->drive_stage = goal_index;
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//     break;
//     case 7:
//     this->drive_stage = goal_index;
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//     break;
//     case 8:
//     this->drive_stage = goal_index;
//     // this->change_pid_param();
    
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//     break;
//     case 9:
//     this->drive_stage = goal_index;
    
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//     case 10:
//     this->change_freespace_param(0);
//     this->drive_stage = goal_index;
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//     break;

//     case 11:
//     // this->change_freespace_param_width(0.1);
//     this->drive_stage = goal_index;
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//     break;

//     case 12:
//     this->drive_stage = goal_index;
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//     break;

//     case 13:
//     this->drive_stage = goal_index;
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//     break;
//     case 14:
//     this->drive_stage = goal_index;
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//     break;
    

//     default:
//     return tier4_planning_msgs::msg::Scenario::PARKING;
//       break;
//   }
//   return current_scenario_;
// }

// void ScenarioSelectorNode::updateCurrentScenario()
// {

//   const auto prev_scenario = current_scenario_;
//   // RCLCPP_INFO_STREAM(this->get_logger(), "updateCurrentScenario");
//   const auto scenario_trajectory = getScenarioTrajectory(current_scenario_);
//   const auto is_near_trajectory_end =
//     isNearTrajectoryEnd(scenario_trajectory, current_pose_->pose, th_arrived_distance_m_);

//   const auto is_stopped = isStopped(twist_buffer_, th_stopped_velocity_mps_);

//   if (is_near_trajectory_end && is_stopped) {
//     current_scenario_ = selectScenarioByPosition();
//   }

//   if (current_scenario_ != prev_scenario) {
   
//     RCLCPP_INFO_STREAM(
//       this->get_logger(), "scenario changed: " << prev_scenario << " -> " << current_scenario_);
//     this->is_published_goal = false;
//     this->is_engaged_goal = false;
//   }

//   //just for test 

  
//   if (is_special_position)
//   {
//       this->drive_stage = this->drive_stage + 1;
//       RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing goal pose. stage add one. the current stage is %d", this->drive_stage);
//       goal_pos_publisher->publish(pre_seted_pose_vector[this->drive_stage + 1]);
//       engage_publisher->publish(createEngageMessage());
//       is_special_position=false;
    
//   }
//   // engage_publisher->publish(createEngageMessage());
// }

// bool ScenarioSelectorNode::inSpecialPosition()
// {
//   for(size_t index = 0; index < this->pre_seted_special_pose_vector.size(); index++)
//     {
//       const auto & p1 = current_pose_->pose.position;
//       const auto & p2 = this->pre_seted_special_pose_vector[index].pose.position;
//       const auto dist = std::hypot(p1.x - p2.x, p1.y - p2.y);
//       const auto & o1 = current_pose_->pose.orientation;
//       const auto & o2 = this->pre_seted_special_pose_vector[index].pose.orientation;
//       double dotProduct = o1.x * o2.x + o1.y * o2.y + o1.z * o2.z + o1.w * o2.w;
//       double angleDifference = 2.0 * std::acos(std::abs(dotProduct));
//       RCLCPP_INFO(this->get_logger(), "poseindex %ld dis is %f, ang is %f",index,dist,angleDifference);
//       if(dist <= 1.5 && angleDifference <= 0.5)
//       {
//         return true;
//       }
//     }
//   return false;
// }

// void ScenarioSelectorNode::onMap(
//   const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg)
// {
//   lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
//   lanelet::utils::conversion::fromBinMsg(
//     *msg, lanelet_map_ptr_, &traffic_rules_ptr_, &routing_graph_ptr_);
//   route_handler_ = std::make_shared<route_handler::RouteHandler>(*msg);
// }

// void ScenarioSelectorNode::onRoute(
//   const autoware_planning_msgs::msg::LaneletRoute::ConstSharedPtr msg)
// {
//   route_ = msg;
//   // current_scenario_ = tier4_planning_msgs::msg::Scenario::EMPTY;
// }

// void ScenarioSelectorNode::onOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
// {
//   auto twist = std::make_shared<geometry_msgs::msg::TwistStamped>();
//   twist->header = msg->header;
//   twist->twist = msg->twist.twist;

//   twist_ = twist;
//   twist_buffer_.push_back(twist);

//   // Delete old data in buffer
//   while (true) {
//     const auto time_diff =
//       rclcpp::Time(msg->header.stamp) - rclcpp::Time(twist_buffer_.front()->header.stamp);

//     if (time_diff.seconds() < th_stopped_time_sec_) {
//       break;
//     }

//     twist_buffer_.pop_front();
//   }
// }

// void ScenarioSelectorNode::onParkingState(const std_msgs::msg::Bool::ConstSharedPtr msg)
// {
//   is_parking_completed_ = msg->data;
// }

// void ScenarioSelectorNode::onSpecialPosition(const std_msgs::msg::Bool::ConstSharedPtr msg)
// {
//   is_special_position = msg->data;
// }

// bool ScenarioSelectorNode::isDataReady()
// {
//   if (!current_pose_) {
//     RCLCPP_INFO(this->get_logger(), "Waiting for current pose.");
//     return false;
//   }

//   if (!lanelet_map_ptr_) {
//     RCLCPP_INFO(this->get_logger(), "Waiting for lanelet map.");
//     return false;
//   }

//   // if (!route_) {
//   //   RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for route.");
//   //   return false;
//   // }

//   // if (!twist_) {
//   //   RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for twist.");
//   //   return false;
//   // }

//   // Check route handler is ready
//   route_handler_->setRoute(*route_);
//   if (!route_handler_->isHandlerReady()) {
//     RCLCPP_WARN_THROTTLE(
//       this->get_logger(), *this->get_clock(), 5000, "Waiting for route handler.");
//     return false;
//   }

//   return true;
// }

// void ScenarioSelectorNode::onTimer()
// {
//   // RCLCPP_INFO(this->get_logger(), "selectScenarioByPosition onTimer");
//   current_pose_ = getCurrentPose(tf_buffer_, this->get_logger());

//   // if (!isDataReady()) {
//   //   RCLCPP_INFO(
//   //     this->get_logger(), "DataNotReady");
//   //   return;
//   // }

//   // Initialize Scenario
//   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::EMPTY) {
//     current_scenario_ = selectScenarioByPosition();
//   }
//   // RCLCPP_INFO(this->get_logger(), "selectScenarioByPosition");
//   updateCurrentScenario();
//   tier4_planning_msgs::msg::Scenario scenario;
//   scenario.current_scenario = current_scenario_;

//   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::PARKING) {
//     scenario.activating_scenarios.push_back(current_scenario_);
//   }

//   pub_scenario_->publish(scenario);
// }




// void ScenarioSelectorNode::onLaneDrivingTrajectory(
//   const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
// {
//   lane_driving_trajectory_ = msg;

//   if (current_scenario_ != tier4_planning_msgs::msg::Scenario::LANEDRIVING) {
//     return;
//   }

//   publishTrajectory(msg);
// }

// void ScenarioSelectorNode::onParkingTrajectory(
//   const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
// {
//   parking_trajectory_ = msg;

//   if (current_scenario_ != tier4_planning_msgs::msg::Scenario::PARKING) {
//     return;
//   }

//   publishTrajectory(msg);
// }

// void ScenarioSelectorNode::publishTrajectory(
//   const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
// {
//   const auto now = this->now();
//   const auto delay_sec = (now - msg->header.stamp).seconds();
//   if (delay_sec <= th_max_message_delay_sec_) {
//     pub_trajectory_->publish(*msg);
//   } else {
//     RCLCPP_WARN_THROTTLE(
//       this->get_logger(), *this->get_clock(), std::chrono::milliseconds(1000).count(),
//       "trajectory is delayed: scenario = %s, delay = %f, th_max_message_delay = %f",
//       current_scenario_.c_str(), delay_sec, th_max_message_delay_sec_);
//   }
// }

// void ScenarioSelectorNode::goalCallback(const AutowareState& msg)
// {
//   if(this->drive_stage >= 50)
//     {
//       // RCLCPP_INFO(this->get_logger(), "Cannot arrive any goal");
//       return;
//     }
//   RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: current_scenario_ is %s. is_published_goal is %d. this->drive_stage is %d, is_engage_goal is %d",current_scenario_.c_str(),this->is_published_goal,this->drive_stage,this->is_engaged_goal);
//   RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: msg.state %d",msg.state);
  

//   if (!current_pose_) {
//     return;
//   }
  
//   switch (msg.state) {
//     case AutowareState::WAITING_FOR_ROUTE:
//       RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: current_state is WAITING_FOR_ROUTE");
//       if(this->drive_stage == -1){
//         RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: exit WAITING_FOR_ROUTE due to drive_stage is -1");
//         break;
//       }
//       if(!this->is_published_goal)
//       {
//         RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing goal pose.");
//         goal_pos_publisher->publish(pre_seted_pose_vector[this->drive_stage + 1]);
//         this->is_published_goal = true;
//       }
//       break;
//     case AutowareState::ARRIVED_GOAL:
//       this->is_published_goal = false;
//       this->is_engaged_goal = false;
//       RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: current_state is ARRIVED_GOAL");
//       if(!this->is_published_goal)
//       {
        
//         RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing goal pose.");
//         goal_pos_publisher->publish(pre_seted_pose_vector[this->drive_stage + 1]);
//         this->is_published_goal = true;
//       }
//       break;
    
//     case AutowareState::PLANNING:
//       RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Planning...");
//       RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: current_state is PLANNING");
//       if(this->drive_stage == -1){
//         RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: exit PLANNING due to drive_stage is -1");
//         break;
//       }
//       if(!this->is_engaged_goal)
//       {
        
//         RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing Planning engage message.");
//         engage_publisher->publish(createEngageMessage());
//         this->is_engaged_goal = true;
//       }
//       break;
//     case AutowareState::WAITING_FOR_ENGAGE:
//       RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: WAITING_FOR_ENGAGE.");
//       if(this->drive_stage == -1){
//         RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: exit WAITING_FOR_ENGAGE due to drive_stage is -1");
//         break;
//       }
//       if(!this->is_engaged_goal)
//       {
//         RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing engage message.");
//         engage_publisher->publish(createEngageMessage());
//         this->is_engaged_goal = true;
//       }
//       break;
//     default:
//       break;
//   }
// }
// void ScenarioSelectorNode::initPreSetedPose()
// {
//   auto msg = PoseStamped();

//  //start--0
// //   msg.header.frame_id = "map";
// //   msg.pose.position.x = 3818.806640625;
// //   msg.pose.position.y = 73772.3046875;
// //   msg.pose.position.z = 0.0;
// //   msg.pose.orientation.x = 0.0;
// //   msg.pose.orientation.y = 0.0;
// //   msg.pose.orientation.z = -0.9712304294215052;
// //   msg.pose.orientation.w = 0.23814166574902115;
// //   pre_seted_pose_vector.push_back(msg);

// // //box1--1
// //   msg.header.frame_id = "map";
// //   msg.pose.position.x = 3796.34033203125;
// //   msg.pose.position.y = 73761.75;
// //   msg.pose.position.z = 0.0;
// //   msg.pose.orientation.x = 0.0;
// //   msg.pose.orientation.y = 0.0;
// //   msg.pose.orientation.z = -0.9719930786142853;
// //   msg.pose.orientation.w = 0.23500947880016188;
// //   pre_seted_pose_vector.push_back(msg);

// // //box2--2
// //   msg.header.frame_id = "map";
// //   msg.pose.position.x = 3789.184326171875;
// //   msg.pose.position.y = 73755.6640625;
// //   msg.pose.position.z = 0.0;
// //   msg.pose.orientation.x = 0.0;
// //   msg.pose.orientation.y = 0.0;
// //   msg.pose.orientation.z = -0.9719930786142853;
// //   msg.pose.orientation.w = 0.23500947880016188;
// //   pre_seted_pose_vector.push_back(msg);


// // //box3--3
// //   msg.header.frame_id = "map";
// //   msg.pose.position.x = 3779.483642578125;
// //   msg.pose.position.y = 73753.0390625;
// //   msg.pose.position.z = 0.0;
// //   msg.pose.orientation.x = 0.0;
// //   msg.pose.orientation.y = 0.0;
// //   msg.pose.orientation.z = -0.962210886752998;
// //   msg.pose.orientation.w = 0.27230536060461474;
// //   pre_seted_pose_vector.push_back(msg);


//  //boxes--0
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3757.973388671875;
//   msg.pose.position.y = 73738.6484375;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = -0.9724300171876937;
//   msg.pose.orientation.w = 0.23319490061393244;
//   pre_seted_pose_vector.push_back(msg);

// // //boxes--0  ---------now
// //   msg.header.frame_id = "map";
// //   msg.pose.position.x = 3754.98876953125;
// //   msg.pose.position.y = 73737.453125;
// //   msg.pose.position.z = 0.0;
// //   msg.pose.orientation.x = 0.0;
// //   msg.pose.orientation.y = 0.0;
// //   msg.pose.orientation.z = -0.9724300171876937;
// //   msg.pose.orientation.w = 0.23319490061393244;
// //   pre_seted_pose_vector.push_back(msg);


// //honggang--5


//   // msg.header.frame_id = "map";
//   // msg.pose.position.x = 3745.2490234375;
//   // msg.pose.position.y = 73743.7109375;
//   // msg.pose.position.z = 0.0;
//   // msg.pose.orientation.x = 0.0;
//   // msg.pose.orientation.y = 0.0;
//   // msg.pose.orientation.z = 0.8693105746034401;
//   // msg.pose.orientation.w = 0.49426624898189925;
//   // pre_seted_pose_vector.push_back(msg);

//   // msg.header.frame_id = "map";
//   // msg.pose.position.x = 3756.67431640625;
//   // msg.pose.position.y = 73738.484375;
//   // msg.pose.position.z = 0.0;
//   // msg.pose.orientation.x = 0.0;
//   // msg.pose.orientation.y = 0.0;
//   // msg.pose.orientation.z = -0.9654916720794067;
//   // msg.pose.orientation.w = 0.2604339285602234;
//   // pre_seted_pose_vector.push_back(msg);
// //--1
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3745.2490234375;
//   msg.pose.position.y = 73743.7109375;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = 0.8693105746034401;
//   msg.pose.orientation.w = 0.49426624898189925;
//   pre_seted_pose_vector.push_back(msg);
  



//   //  L first 
//   // msg.header.frame_id = "map";
//   // msg.pose.position.x = 3749.715576171875;
//   // msg.pose.position.y = 73734.71875;
//   // msg.pose.position.z = 0.0;
//   // msg.pose.orientation.x = 0.0;
//   // msg.pose.orientation.y = 0.0;
//   // msg.pose.orientation.z = 0.7579653122371283;
//   // msg.pose.orientation.w = 0.6522948608147029;
//   // pre_seted_pose_vector.push_back(msg);
  
//   // //  L first replace
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3750.13525390625;
//   msg.pose.position.y = 73734.7890625;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = 0.7525766947068779;
//   msg.pose.orientation.w = 0.658504607868518;
//   pre_seted_pose_vector.push_back(msg);

  
  
//   // //  L second
//   // // msg.header.frame_id = "map";
//   // // msg.pose.position.x = 3749.709716796875;
//   // // msg.pose.position.y = 73735.859375;
//   // // msg.pose.position.z = 0.0;
//   // // msg.pose.orientation.x = 0.0;
//   // // msg.pose.orientation.y = 0.0;
//   // // msg.pose.orientation.z = 0.5708294720816296;
//   // // msg.pose.orientation.w = 0.8210686413467562;
//   // // pre_seted_pose_vector.push_back(msg);
  
//   //  //  L third
//   // // msg.header.frame_id = "map";
//   // // msg.pose.position.x = 3750.218505859375;
//   // // msg.pose.position.y = 73736.234375;
//   // // msg.pose.position.z = 0.0;
//   // // msg.pose.orientation.x = 0.0;
//   // // msg.pose.orientation.y = 0.0;
//   // // msg.pose.orientation.z = 0.43712374276198984;
//   // // msg.pose.orientation.w = 0.8994013750899815;
//   // // pre_seted_pose_vector.push_back(msg);

//    //  L third replace
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3750.20263671875;
//   msg.pose.position.y = 73736.03125;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = 0.5799025524453801;
//   msg.pose.orientation.w = 0.8146858472241513;
//   pre_seted_pose_vector.push_back(msg);

  
//   //  //  L third replace  ---
//   // msg.header.frame_id = "map";
//   // msg.pose.position.x = 3750.34814453125;
//   // msg.pose.position.y = 73736.6875;
//   // msg.pose.position.z = 0.0;
//   // msg.pose.orientation.x = 0.0;
//   // msg.pose.orientation.y = 0.0;
//   // msg.pose.orientation.z = 0.6017139825542301;
//   // msg.pose.orientation.w = 0.7987116395788455;
//   // pre_seted_pose_vector.push_back(msg);
  


//   // msg.header.frame_id = "map";
//   // msg.pose.position.x = 3751.2109375;
//   // msg.pose.position.y = 73737.1015625;
//   // msg.pose.position.z = 0.0;
//   // msg.pose.orientation.x = 0.0;
//   // msg.pose.orientation.y = 0.0;
//   // msg.pose.orientation.z = 0.3214301106708757;
//   // msg.pose.orientation.w = 0.9469333049133443;
//   // pre_seted_pose_vector.push_back(msg);

// //--2
//   // msg.header.frame_id = "map";
//   // msg.pose.position.x = 3749.689697265625;
//   // msg.pose.position.y = 73735.3203125;
//   // msg.pose.position.z = 0.0;
//   // msg.pose.orientation.x = 0.0;
//   // msg.pose.orientation.y = 0.0;
//   // msg.pose.orientation.z = 0.6678927619723282;
//   // msg.pose.orientation.w = 0.7442575216314411;
//   // pre_seted_pose_vector.push_back(msg);
// //--3
//   // msg.header.frame_id = "map";
//   // msg.pose.position.x = 3755.7021484375;
//   // msg.pose.position.y = 73739.1640625;
//   // msg.pose.position.z = 0.0;
//   // msg.pose.orientation.x = 0.0;
//   // msg.pose.orientation.y = 0.0;
//   // msg.pose.orientation.z = 0.24927759678599598;
//   // msg.pose.orientation.w = 0.9684320728582869;
//   // pre_seted_pose_vector.push_back(msg);

//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3754.760498046875;
//   msg.pose.position.y = 73738.625;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = 0.24487996608642515;
//   msg.pose.orientation.w = 0.9695534034850846;
//   pre_seted_pose_vector.push_back(msg);

//   //--3 + 
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3750.83154296875;
//   msg.pose.position.y = 73737.640625;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = 0.024630872112737776;
//   msg.pose.orientation.w = 0.9996966140479651;
//   pre_seted_pose_vector.push_back(msg);

//  // -3 replace
//   // msg.header.frame_id = "map";
//   // msg.pose.position.x = 3755.517333984375;
//   // msg.pose.position.y = 73738.7421875;
//   // msg.pose.position.z = 0.0;
//   // msg.pose.orientation.x = 0.0;
//   // msg.pose.orientation.y = 0.0;
//   // msg.pose.orientation.z = 0.26645980910882694;
//   // msg.pose.orientation.w = 0.9638460303023961;
//   // pre_seted_pose_vector.push_back(msg);



//   //--4
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3745.0830078125;
//   msg.pose.position.y = 73743.4453125;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = -0.5178196572501046;
//   msg.pose.orientation.w = 0.8554898027243716;
//   pre_seted_pose_vector.push_back(msg);
// //--5
// //honggang

// //drive 1
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3739.350830078125;
//   msg.pose.position.y = 73754.3515625;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = -0.49164540017165514;
//   msg.pose.orientation.w = 0.8707954986620298;
//   pre_seted_pose_vector.push_back(msg);
// //--6
// //drive 1 middle
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3727.3359375;
//   msg.pose.position.y = 73757.21875;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = 0.2408351489047307;
//   msg.pose.orientation.w = 0.9705660364200038;
//   pre_seted_pose_vector.push_back(msg);

//   //--7
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3718.353271484375;
//   msg.pose.position.y = 73750.734375;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = 0.6679334714495515;
//   msg.pose.orientation.w = 0.7442209871519018;
//   pre_seted_pose_vector.push_back(msg);
// //--8
// //drive 2
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3720.598388671875;
//   msg.pose.position.y = 73744.0859375;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = 0.8632819539170411;
//   msg.pose.orientation.w = 0.5047219710307603;
//   pre_seted_pose_vector.push_back(msg);

// // --9 
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3715.57568359375;
//   msg.pose.position.y = 73734.90625;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = 0.26171122506788413;
//   msg.pose.orientation.w = 0.9651462245035554;
//   pre_seted_pose_vector.push_back(msg);

//   //--9 
//   // msg.header.frame_id = "map";
//   // msg.pose.position.x = 3715.57568359375;
//   // msg.pose.position.y = 73734.90625;
//   // msg.pose.position.z = 0.0;
//   // msg.pose.orientation.x = 0.0;
//   // msg.pose.orientation.y = 0.0;
//   // msg.pose.orientation.z = 0.23593056980882635;
//   // msg.pose.orientation.w = 0.9717699142439441;
//   // pre_seted_pose_vector.push_back(msg);
  
  
//   //--10
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3706.668701171875;
//   msg.pose.position.y = 73730.421875;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = 0.2309412920875842;
//   msg.pose.orientation.w = 0.9729676868267091;
//   pre_seted_pose_vector.push_back(msg);


//   //--10
//   // msg.header.frame_id = "map";
//   // msg.pose.position.x = 3704.99658203125;
//   // msg.pose.position.y = 73737.609375;
//   // msg.pose.position.z = 0.0;
//   // msg.pose.orientation.x = 0.0;
//   // msg.pose.orientation.y = 0.0;
//   // msg.pose.orientation.z = -0.5021006243841415;
//   // msg.pose.orientation.w = 0.8648092061218215;
//   // pre_seted_pose_vector.push_back(msg);

// //--9 
//   // msg.header.frame_id = "map";
//   // msg.pose.position.x = 3708.572509765625;
//   // msg.pose.position.y = 73731.375;
//   // msg.pose.position.z = 0.0;
//   // msg.pose.orientation.x = 0.0;
//   // msg.pose.orientation.y = 0.0;
//   // msg.pose.orientation.z = 0.24365928570205184;
//   // msg.pose.orientation.w = 0.9698608933713978;
//   // pre_seted_pose_vector.push_back(msg);
// //--9

// // //drive3
// //   msg.header.frame_id = "map";
// //   msg.pose.position.x = 3717.45068359375;
// //   msg.pose.position.y = 73735.7109375;
// //   msg.pose.position.z = 0.0;
// //   msg.pose.orientation.x = 0.0;
// //   msg.pose.orientation.y = 0.0;
// //   msg.pose.orientation.z = 0.23903341739572623;
// //   msg.pose.orientation.w = 0.9710113415239394;
// //   pre_seted_pose_vector.push_back(msg);

// //drive4
// msg.header.frame_id = "map";
// msg.pose.position.x = 3701.373779296875;
// msg.pose.position.y = 73744.7890625;
// msg.pose.position.z = 0.0;
// msg.pose.orientation.x = 0.0;
// msg.pose.orientation.y = 0.0;
// msg.pose.orientation.z = -0.5372075731791744;
// msg.pose.orientation.w = 0.8434500716218726;
// pre_seted_pose_vector.push_back(msg);

//   for(auto pose : pre_seted_pose_vector)
//   {
//     RCLCPP_INFO(this->get_logger(), "pre_seted_pose_vector x is %f",pose.pose.position.x);
//   }



// //special pose
// msg.header.frame_id = "map";
// msg.pose.position.x = 3754.385009765625;
// msg.pose.position.y = 73739.8046875;
// msg.pose.position.z = 0.0;
// msg.pose.orientation.x = 0.0;
// msg.pose.orientation.y = 0.0;
// msg.pose.orientation.z = 0.1939932506522466;
// msg.pose.orientation.w = 0.9810028637579885;
// pre_seted_special_pose_vector.push_back(msg);

// }

// // std::string exec(const char* cmd) {
// //     std::array<char, 128> buffer;
// //     std::string result;
// //     std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
// //     if (!pipe) throw std::runtime_error("popen() failed!");
// //     while (!feof(pipe.get())) {
// //         if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
// //             result += buffer.data();
// //     }
// //     return result;
// // }


// void ScenarioSelectorNode::change_freespace_param(const int & num)
// {
//     // std::string result = exec("ros2 param set /planning/scenario_planning/parking/freespace_planner jump_if_size_less 1");
//     // RCLCPP_INFO(this->get_logger(), "Parameter set result is %s",result.c_str());
//   // // 修改节点的参数值
//     std::vector<rclcpp::Parameter> params;
//     params.push_back(rclcpp::Parameter("jump_if_size_less", num));

//     auto result = this->parameter_client_freespace->set_parameters_atomically(params);
//     if(result.successful)
//     {
//         RCLCPP_INFO(this->get_logger(), "Parameter set successfully");
//     }
//     else
//     {
//         RCLCPP_ERROR(this->get_logger(), "Failed to set parameter");
//     }
//   }

// void ScenarioSelectorNode::change_freespace_param_width(const double & num)
// {
//     // std::string result = exec("ros2 param set /planning/scenario_planning/parking/freespace_planner jump_if_size_less 1");
//     // RCLCPP_INFO(this->get_logger(), "Parameter set result is %s",result.c_str());
//   // // 修改节点的参数值
//     std::vector<rclcpp::Parameter> params;
//     params.push_back(rclcpp::Parameter("vehicle_shape_margin_width_m", num));

//     auto result = this->parameter_client_freespace->set_parameters_atomically(params);
//     if(result.successful)
//     {
//         RCLCPP_INFO(this->get_logger(), "vehicle_shape_margin_width_m Parameter set successfully");
//     }
//     else
//     {
//         RCLCPP_ERROR(this->get_logger(), "vehicle_shape_margin_width_m Failed to set parameter");
//     }
//   }




//   void ScenarioSelectorNode::change_pid_param()
// {
//   // // 修改节点的参数值
//   // 修改节点的参数值
//     std::vector<rclcpp::Parameter> params;
//     params.push_back(rclcpp::Parameter("stopping_state_stop_dist", 0.1));

//     auto result = this->parameter_client_pid->set_parameters_atomically(params);
//     if(result.successful)
//     {
//         RCLCPP_INFO(this->get_logger(), "Parameter set successfully");
//     }
//     else
//     {
//         RCLCPP_ERROR(this->get_logger(), "Failed to set parameter");
//     }
//   }

  
// ScenarioSelectorNode::ScenarioSelectorNode(const rclcpp::NodeOptions & node_options)
// : Node("scenario_selector", node_options),
//   tf_buffer_(this->get_clock()),
//   tf_listener_(tf_buffer_),
//   current_scenario_(tier4_planning_msgs::msg::Scenario::EMPTY),
//   update_rate_(this->declare_parameter<double>("update_rate", 10.0)),
//   th_max_message_delay_sec_(this->declare_parameter<double>("th_max_message_delay_sec", 1.0)),
//   th_arrived_distance_m_(this->declare_parameter<double>("th_arrived_distance_m", 1.0)),
//   th_stopped_time_sec_(this->declare_parameter<double>("th_stopped_time_sec", 1.0)),
//   th_stopped_velocity_mps_(this->declare_parameter<double>("th_stopped_velocity_mps", 0.01)),
//   is_parking_completed_(false),
//   drive_stage(-1),
//   is_published_goal(false),
//   is_special_position(false),
//   is_engaged_goal(false)
// {
  

    
//   // Input
//   sub_lane_driving_trajectory_ =
//     this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
//       "input/lane_driving/trajectory", rclcpp::QoS{1},
//       std::bind(&ScenarioSelectorNode::onLaneDrivingTrajectory, this, std::placeholders::_1));

//   sub_parking_trajectory_ = this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
//     "input/parking/trajectory", rclcpp::QoS{1},
//     std::bind(&ScenarioSelectorNode::onParkingTrajectory, this, std::placeholders::_1));

//   sub_lanelet_map_ = this->create_subscription<autoware_auto_mapping_msgs::msg::HADMapBin>(
//     "input/lanelet_map", rclcpp::QoS{1}.transient_local(),
//     std::bind(&ScenarioSelectorNode::onMap, this, std::placeholders::_1));
//   sub_route_ = this->create_subscription<autoware_planning_msgs::msg::LaneletRoute>(
//     "input/route", rclcpp::QoS{1}.transient_local(),
//     std::bind(&ScenarioSelectorNode::onRoute, this, std::placeholders::_1));
//   sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
//     "input/odometry", rclcpp::QoS{100},
//     std::bind(&ScenarioSelectorNode::onOdom, this, std::placeholders::_1));
//   sub_parking_state_ = this->create_subscription<std_msgs::msg::Bool>(
//     "is_parking_completed", rclcpp::QoS{100},
//     std::bind(&ScenarioSelectorNode::onParkingState, this, std::placeholders::_1));
  
//   sub_special_position = this->create_subscription<std_msgs::msg::Bool>(
//     "is_special_position", rclcpp::QoS{100},
//     std::bind(&ScenarioSelectorNode::onSpecialPosition, this, std::placeholders::_1));

//   // Output
//   pub_scenario_ =
//     this->create_publisher<tier4_planning_msgs::msg::Scenario>("output/scenario", rclcpp::QoS{1});
//   pub_trajectory_ = this->create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
//     "output/trajectory", rclcpp::QoS{1});


//   //ai_challange_control
//   // Publishers
//   engage_publisher =
//     this->create_publisher<Engage>("output/engage", 1);
//   goal_pos_publisher =
//     this->create_publisher<PoseStamped>("output/goal", 1);

//   // Subscribers
//   state_subscriber = this->create_subscription<AutowareState>(
//     "input/state", 1, std::bind(&ScenarioSelectorNode::goalCallback, this, std::placeholders::_1));

//   // Timer Callback
//   const auto period_ns = rclcpp::Rate(static_cast<double>(update_rate_)).period();

//   timer_ = rclcpp::create_timer(
//     this, get_clock(), period_ns, std::bind(&ScenarioSelectorNode::onTimer, this));
//   // 创建一个参数客户端来修改参数
//   auto freespace_manager = rclcpp::Node::make_shared("freespace_manager");
//   this->parameter_client_freespace = std::make_shared<rclcpp::SyncParametersClient>(freespace_manager,"/planning/scenario_planning/parking/freespace_planner");
//   auto pid_manager = rclcpp::Node::make_shared("pid_manager");
//   // 创建一个参数客户端来修改参数
//   this->parameter_client_pid = std::make_shared<rclcpp::SyncParametersClient>(pid_manager,"/control/trajectory_follower/controller_node_exe");

//   this->initPreSetedPose();
//   // Wait for first tf
//   while (rclcpp::ok()) {
//     try {
//       tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
//       break;
//     } catch (tf2::TransformException & ex) {
//       RCLCPP_DEBUG(this->get_logger(), "waiting for initial pose...");
//       rclcpp::sleep_for(std::chrono::milliseconds(100));
//     }
//   }
// }

// #include <rclcpp_components/register_node_macro.hpp>
// RCLCPP_COMPONENTS_REGISTER_NODE(ScenarioSelectorNode)













// // // Copyright 2020 Tier IV, Inc.
// // //
// // // Licensed under the Apache License, Version 2.0 (the "License");
// // // you may not use this file except in compliance with the License.
// // // You may obtain a copy of the License at
// // //
// // //     http://www.apache.org/licenses/LICENSE-2.0
// // //
// // // Unless required by applicable law or agreed to in writing, software
// // // distributed under the License is distributed on an "AS IS" BASIS,
// // // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // // See the License for the specific language governing permissions and
// // // limitations under the License.

// // #include "scenario_selector/scenario_selector_node.hpp"

// // #include <lanelet2_extension/utility/message_conversion.hpp>
// // #include <lanelet2_extension/utility/query.hpp>

// // #include <lanelet2_core/geometry/BoundingBox.h>
// // #include <lanelet2_core/geometry/Lanelet.h>
// // #include <lanelet2_core/geometry/LineString.h>
// // #include <lanelet2_core/geometry/Point.h>
// // #include <lanelet2_core/geometry/Polygon.h>

// // #include <deque>
// // #include <memory>
// // #include <string>
// // #include <utility>
// // #include <vector>

// // namespace
// // {
// // template <class T>
// // void onData(const T & data, T * buffer)
// // {
// //   *buffer = data;
// // }

// // std::shared_ptr<lanelet::ConstPolygon3d> findNearestParkinglot(
// //   const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
// //   const lanelet::BasicPoint2d & current_position)
// // {
// //   const auto all_parking_lots = lanelet::utils::query::getAllParkingLots(lanelet_map_ptr);

// //   const auto linked_parking_lot = std::make_shared<lanelet::ConstPolygon3d>();
// //   const auto result = lanelet::utils::query::getLinkedParkingLot(
// //     current_position, all_parking_lots, linked_parking_lot.get());

// //   if (result) {
// //     return linked_parking_lot;
// //   } else {
// //     return {};
// //   }
// // }

// // geometry_msgs::msg::PoseStamped::ConstSharedPtr getCurrentPose(
// //   const tf2_ros::Buffer & tf_buffer, const rclcpp::Logger & logger)
// // {
// //   geometry_msgs::msg::TransformStamped tf_current_pose;

// //   try {
// //     tf_current_pose = tf_buffer.lookupTransform("map", "base_link", tf2::TimePointZero);
// //   } catch (tf2::TransformException & ex) {
// //     RCLCPP_ERROR(logger, "%s", ex.what());
// //     return nullptr;
// //   }

// //   geometry_msgs::msg::PoseStamped::SharedPtr p(new geometry_msgs::msg::PoseStamped());
// //   p->header = tf_current_pose.header;
// //   p->pose.orientation = tf_current_pose.transform.rotation;
// //   p->pose.position.x = tf_current_pose.transform.translation.x;
// //   p->pose.position.y = tf_current_pose.transform.translation.y;
// //   p->pose.position.z = tf_current_pose.transform.translation.z;

// //   return geometry_msgs::msg::PoseStamped::ConstSharedPtr(p);
// // }

// // bool isInLane(
// //   const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
// //   const geometry_msgs::msg::Point & current_pos)
// // {
// //   const auto & p = current_pos;
// //   const lanelet::Point3d search_point(lanelet::InvalId, p.x, p.y, p.z);

// //   std::vector<std::pair<double, lanelet::Lanelet>> nearest_lanelets =
// //     lanelet::geometry::findNearest(lanelet_map_ptr->laneletLayer, search_point.basicPoint2d(), 1);

// //   if (nearest_lanelets.empty()) {
// //     return false;
// //   }

// //   const auto nearest_lanelet = nearest_lanelets.front().second;

// //   return lanelet::geometry::within(search_point, nearest_lanelet.polygon3d());
// // }

// // bool isInParkingLot(
// //   const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
// //   const geometry_msgs::msg::Pose & current_pose)
// // {
// //   const auto & p = current_pose.position;
// //   const lanelet::Point3d search_point(lanelet::InvalId, p.x, p.y, p.z);

// //   const auto nearest_parking_lot =
// //     findNearestParkinglot(lanelet_map_ptr, search_point.basicPoint2d());

// //   if (!nearest_parking_lot) {
// //     return false;
// //   }

// //   return lanelet::geometry::within(search_point, nearest_parking_lot->basicPolygon());
// // }

// // bool isNearTrajectoryEnd(
// //   const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr trajectory,
// //   const geometry_msgs::msg::Pose & current_pose, const double th_dist)
// // {
// //   if (!trajectory || trajectory->points.empty()) {
// //     return false;
// //   }

// //   const auto & p1 = current_pose.position;
// //   const auto & p2 = trajectory->points.back().pose.position;

// //   const auto dist = std::hypot(p1.x - p2.x, p1.y - p2.y);

// //   return dist < th_dist;
// // }

// // bool isStopped(
// //   const std::deque<geometry_msgs::msg::TwistStamped::ConstSharedPtr> & twist_buffer,
// //   const double th_stopped_velocity_mps)
// // {
// //   for (const auto & twist : twist_buffer) {
// //     if (std::abs(twist->twist.linear.x) > th_stopped_velocity_mps) {
// //       return false;
// //     }
// //   }
// //   return true;
// // }

// // }  // namespace

// // autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr
// // ScenarioSelectorNode::getScenarioTrajectory(const std::string & scenario)
// // {
// //   if (scenario == tier4_planning_msgs::msg::Scenario::LANEDRIVING) {
// //     return lane_driving_trajectory_;
// //   }
// //   if (scenario == tier4_planning_msgs::msg::Scenario::PARKING) {
// //     return parking_trajectory_;
// //   }
// //   RCLCPP_ERROR_STREAM(this->get_logger(), "invalid scenario argument: " << scenario);
// //   return lane_driving_trajectory_;
// // }

// // std::string ScenarioSelectorNode::selectScenarioByPosition()
// // { 
// //   // return tier4_planning_msgs::msg::Scenario::PARKING;
// //   const auto is_in_lane = isInLane(lanelet_map_ptr_, current_pose_->pose.position);
// //   const auto is_goal_in_lane = isInLane(lanelet_map_ptr_, route_->goal_pose.position);
// //   const auto is_in_parking_lot = isInParkingLot(lanelet_map_ptr_, current_pose_->pose);

// //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::EMPTY) {
// //     if (is_in_lane && is_goal_in_lane) {
// //       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
// //     } else if (is_in_parking_lot) {
// //       return tier4_planning_msgs::msg::Scenario::PARKING;
// //     } else {
// //       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
// //     }
// //   }

// //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::LANEDRIVING) {
// //     if (is_in_parking_lot && !is_goal_in_lane) {
// //       return tier4_planning_msgs::msg::Scenario::PARKING;
// //     }
// //   }

// //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::PARKING) {
// //     if (is_parking_completed_ && is_in_lane) {
// //       is_parking_completed_ = false;
// //       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
// //     }
// //   }

// //   return current_scenario_;
// // }

// // void ScenarioSelectorNode::updateCurrentScenario()
// // {
// //   const auto prev_scenario = current_scenario_;

// //   const auto scenario_trajectory = getScenarioTrajectory(current_scenario_);
// //   const auto is_near_trajectory_end =
// //     isNearTrajectoryEnd(scenario_trajectory, current_pose_->pose, th_arrived_distance_m_);

// //   const auto is_stopped = isStopped(twist_buffer_, th_stopped_velocity_mps_);

// //   if (is_near_trajectory_end && is_stopped) {
// //     current_scenario_ = selectScenarioByPosition();
// //   }

// //   if (current_scenario_ != prev_scenario) {
// //     RCLCPP_INFO_STREAM(
// //       this->get_logger(), "scenario changed: " << prev_scenario << " -> " << current_scenario_);
// //   }
// // }

// // void ScenarioSelectorNode::onMap(
// //   const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg)
// // {
// //   lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
// //   lanelet::utils::conversion::fromBinMsg(
// //     *msg, lanelet_map_ptr_, &traffic_rules_ptr_, &routing_graph_ptr_);
// //   route_handler_ = std::make_shared<route_handler::RouteHandler>(*msg);
// // }

// // void ScenarioSelectorNode::onRoute(
// //   const autoware_planning_msgs::msg::LaneletRoute::ConstSharedPtr msg)
// // {
// //   route_ = msg;
// //   current_scenario_ = tier4_planning_msgs::msg::Scenario::EMPTY;
// // }

// // void ScenarioSelectorNode::onOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
// // {
// //   auto twist = std::make_shared<geometry_msgs::msg::TwistStamped>();
// //   twist->header = msg->header;
// //   twist->twist = msg->twist.twist;

// //   twist_ = twist;
// //   twist_buffer_.push_back(twist);

// //   // Delete old data in buffer
// //   while (true) {
// //     const auto time_diff =
// //       rclcpp::Time(msg->header.stamp) - rclcpp::Time(twist_buffer_.front()->header.stamp);

// //     if (time_diff.seconds() < th_stopped_time_sec_) {
// //       break;
// //     }

// //     twist_buffer_.pop_front();
// //   }
// // }

// // void ScenarioSelectorNode::onParkingState(const std_msgs::msg::Bool::ConstSharedPtr msg)
// // {
// //   is_parking_completed_ = msg->data;
// // }

// // bool ScenarioSelectorNode::isDataReady()
// // {
// //   if (!current_pose_) {
// //     RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for current pose.");
// //     return false;
// //   }

// //   if (!lanelet_map_ptr_) {
// //     RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for lanelet map.");
// //     return false;
// //   }

// //   if (!route_) {
// //     RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for route.");
// //     return false;
// //   }

// //   if (!twist_) {
// //     RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for twist.");
// //     return false;
// //   }

// //   // Check route handler is ready
// //   route_handler_->setRoute(*route_);
// //   if (!route_handler_->isHandlerReady()) {
// //     RCLCPP_WARN_THROTTLE(
// //       this->get_logger(), *this->get_clock(), 5000, "Waiting for route handler.");
// //     return false;
// //   }

// //   return true;
// // }

// // void ScenarioSelectorNode::onTimer()
// // {
// //   current_pose_ = getCurrentPose(tf_buffer_, this->get_logger());

// //   if (!isDataReady()) {
// //     return;
// //   }

// //   // Initialize Scenario
// //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::EMPTY) {
// //     current_scenario_ = selectScenarioByPosition();
// //   }

// //   updateCurrentScenario();
// //   tier4_planning_msgs::msg::Scenario scenario;
// //   scenario.current_scenario = current_scenario_;

// //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::PARKING) {
// //     scenario.activating_scenarios.push_back(current_scenario_);
// //   }

// //   pub_scenario_->publish(scenario);
// // }

// // void ScenarioSelectorNode::onLaneDrivingTrajectory(
// //   const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
// // {
// //   lane_driving_trajectory_ = msg;

// //   if (current_scenario_ != tier4_planning_msgs::msg::Scenario::LANEDRIVING) {
// //     return;
// //   }

// //   publishTrajectory(msg);
// // }

// // void ScenarioSelectorNode::onParkingTrajectory(
// //   const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
// // {
// //   parking_trajectory_ = msg;

// //   if (current_scenario_ != tier4_planning_msgs::msg::Scenario::PARKING) {
// //     return;
// //   }

// //   publishTrajectory(msg);
// // }

// // void ScenarioSelectorNode::publishTrajectory(
// //   const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
// // {
// //   const auto now = this->now();
// //   const auto delay_sec = (now - msg->header.stamp).seconds();
// //   if (delay_sec <= th_max_message_delay_sec_) {
// //     pub_trajectory_->publish(*msg);
// //   } else {
// //     RCLCPP_WARN_THROTTLE(
// //       this->get_logger(), *this->get_clock(), std::chrono::milliseconds(1000).count(),
// //       "trajectory is delayed: scenario = %s, delay = %f, th_max_message_delay = %f",
// //       current_scenario_.c_str(), delay_sec, th_max_message_delay_sec_);
// //   }
// // }

// // ScenarioSelectorNode::ScenarioSelectorNode(const rclcpp::NodeOptions & node_options)
// // : Node("scenario_selector", node_options),
// //   tf_buffer_(this->get_clock()),
// //   tf_listener_(tf_buffer_),
// //   current_scenario_(tier4_planning_msgs::msg::Scenario::EMPTY),
// //   update_rate_(this->declare_parameter<double>("update_rate", 10.0)),
// //   th_max_message_delay_sec_(this->declare_parameter<double>("th_max_message_delay_sec", 1.0)),
// //   th_arrived_distance_m_(this->declare_parameter<double>("th_arrived_distance_m", 1.0)),
// //   th_stopped_time_sec_(this->declare_parameter<double>("th_stopped_time_sec", 1.0)),
// //   th_stopped_velocity_mps_(this->declare_parameter<double>("th_stopped_velocity_mps", 0.01)),
// //   is_parking_completed_(false)
// // {
// //   // Input
// //   sub_lane_driving_trajectory_ =
// //     this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
// //       "input/lane_driving/trajectory", rclcpp::QoS{1},
// //       std::bind(&ScenarioSelectorNode::onLaneDrivingTrajectory, this, std::placeholders::_1));

// //   sub_parking_trajectory_ = this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
// //     "input/parking/trajectory", rclcpp::QoS{1},
// //     std::bind(&ScenarioSelectorNode::onParkingTrajectory, this, std::placeholders::_1));

// //   sub_lanelet_map_ = this->create_subscription<autoware_auto_mapping_msgs::msg::HADMapBin>(
// //     "input/lanelet_map", rclcpp::QoS{1}.transient_local(),
// //     std::bind(&ScenarioSelectorNode::onMap, this, std::placeholders::_1));
// //   sub_route_ = this->create_subscription<autoware_planning_msgs::msg::LaneletRoute>(
// //     "input/route", rclcpp::QoS{1}.transient_local(),
// //     std::bind(&ScenarioSelectorNode::onRoute, this, std::placeholders::_1));
// //   sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
// //     "input/odometry", rclcpp::QoS{100},
// //     std::bind(&ScenarioSelectorNode::onOdom, this, std::placeholders::_1));
// //   sub_parking_state_ = this->create_subscription<std_msgs::msg::Bool>(
// //     "is_parking_completed", rclcpp::QoS{100},
// //     std::bind(&ScenarioSelectorNode::onParkingState, this, std::placeholders::_1));

// //   // Output
// //   pub_scenario_ =
// //     this->create_publisher<tier4_planning_msgs::msg::Scenario>("output/scenario", rclcpp::QoS{1});
// //   pub_trajectory_ = this->create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
// //     "output/trajectory", rclcpp::QoS{1});

// //   // Timer Callback
// //   const auto period_ns = rclcpp::Rate(static_cast<double>(update_rate_)).period();

// //   timer_ = rclcpp::create_timer(
// //     this, get_clock(), period_ns, std::bind(&ScenarioSelectorNode::onTimer, this));

// //   // Wait for first tf
// //   while (rclcpp::ok()) {
// //     try {
// //       tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
// //       break;
// //     } catch (tf2::TransformException & ex) {
// //       RCLCPP_DEBUG(this->get_logger(), "waiting for initial pose...");
// //       rclcpp::sleep_for(std::chrono::milliseconds(100));
// //     }
// //   }
// // }

// // #include <rclcpp_components/register_node_macro.hpp>
// // RCLCPP_COMPONENTS_REGISTER_NODE(ScenarioSelectorNode)


























// // // Copyright 2020 Tier IV, Inc.
// // //
// // // Licensed under the Apache License, Version 2.0 (the "License");
// // // you may not use this file except in compliance with the License.
// // // You may obtain a copy of the License at
// // //
// // //     http://www.apache.org/licenses/LICENSE-2.0
// // //
// // // Unless required by applicable law or agreed to in writing, software
// // // distributed under the License is distributed on an "AS IS" BASIS,
// // // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // // See the License for the specific language governing permissions and
// // // limitations under the License.

// // #include "scenario_selector/scenario_selector_node.hpp"

// // #include <lanelet2_extension/utility/message_conversion.hpp>
// // #include <lanelet2_extension/utility/query.hpp>

// // #include <lanelet2_core/geometry/BoundingBox.h>
// // #include <lanelet2_core/geometry/Lanelet.h>
// // #include <lanelet2_core/geometry/LineString.h>
// // #include <lanelet2_core/geometry/Point.h>
// // #include <lanelet2_core/geometry/Polygon.h>

// // #include <deque>
// // #include <memory>
// // #include <string>
// // #include <utility>
// // #include <vector>

// // namespace
// // {
// // template <class T>
// // void onData(const T & data, T * buffer)
// // {
// //   *buffer = data;
// // }

// // using AutowareState = autoware_auto_system_msgs::msg::AutowareState;
// // using Engage = autoware_auto_vehicle_msgs::msg::Engage;
// // using PoseStamped = geometry_msgs::msg::PoseStamped;
// // Engage createEngageMessage()
// // {
// //   auto msg = Engage();
// //   msg.engage = true;
// //   return msg;
// // }
// // // std::shared_ptr<lanelet::ConstPolygon3d> findNearestParkinglot(
// // //   const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
// // //   const lanelet::BasicPoint2d & current_position)
// // // {
// // //   const auto all_parking_lots = lanelet::utils::query::getAllParkingLots(lanelet_map_ptr);

// // //   const auto linked_parking_lot = std::make_shared<lanelet::ConstPolygon3d>();
// // //   const auto result = lanelet::utils::query::getLinkedParkingLot(
// // //     current_position, all_parking_lots, linked_parking_lot.get());

// // //   if (result) {
// // //     return linked_parking_lot;
// // //   } else {
// // //     return {};
// // //   }
// // // }

// // geometry_msgs::msg::PoseStamped::ConstSharedPtr getCurrentPose(
// //   const tf2_ros::Buffer & tf_buffer, const rclcpp::Logger & logger)
// // {
// //   geometry_msgs::msg::TransformStamped tf_current_pose;

// //   try {
// //     tf_current_pose = tf_buffer.lookupTransform("map", "base_link", tf2::TimePointZero);
// //   } catch (tf2::TransformException & ex) {
// //     RCLCPP_ERROR(logger, "%s", ex.what());
// //     return nullptr;
// //   }

// //   geometry_msgs::msg::PoseStamped::SharedPtr p(new geometry_msgs::msg::PoseStamped());
// //   p->header = tf_current_pose.header;
// //   p->pose.orientation = tf_current_pose.transform.rotation;
// //   p->pose.position.x = tf_current_pose.transform.translation.x;
// //   p->pose.position.y = tf_current_pose.transform.translation.y;
// //   p->pose.position.z = tf_current_pose.transform.translation.z;

// //   return geometry_msgs::msg::PoseStamped::ConstSharedPtr(p);
// // }

// // // bool isInLane(
// // //   const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
// // //   const geometry_msgs::msg::Point & current_pos)
// // // {
// // //   const auto & p = current_pos;
// // //   const lanelet::Point3d search_point(lanelet::InvalId, p.x, p.y, p.z);

// // //   std::vector<std::pair<double, lanelet::Lanelet>> nearest_lanelets =
// // //     lanelet::geometry::findNearest(lanelet_map_ptr->laneletLayer, search_point.basicPoint2d(), 1);

// // //   if (nearest_lanelets.empty()) {
// // //     return false;
// // //   }

// // //   const auto nearest_lanelet = nearest_lanelets.front().second;

// // //   return lanelet::geometry::within(search_point, nearest_lanelet.polygon3d());
// // // }

// // // bool isInParkingLot(
// // //   const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
// // //   const geometry_msgs::msg::Pose & current_pose)
// // // {
// // //   const auto & p = current_pose.position;
// // //   const lanelet::Point3d search_point(lanelet::InvalId, p.x, p.y, p.z);

// // //   const auto nearest_parking_lot =
// // //     findNearestParkinglot(lanelet_map_ptr, search_point.basicPoint2d());

// // //   if (!nearest_parking_lot) {
// // //     return false;
// // //   }

// // //   return lanelet::geometry::within(search_point, nearest_parking_lot->basicPolygon());
// // // }



// // bool isNearTrajectoryEnd(
// //   const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr trajectory,
// //   const geometry_msgs::msg::Pose & current_pose, const double th_dist)
// // {
// //   if (!trajectory || trajectory->points.empty()) {
// //     return false;
// //   }

// //   const auto & p1 = current_pose.position;
// //   const auto & p2 = trajectory->points.back().pose.position;

// //   const auto dist = std::hypot(p1.x - p2.x, p1.y - p2.y);

// //   return dist < th_dist;
// // }

// // bool isStopped(
// //   const std::deque<geometry_msgs::msg::TwistStamped::ConstSharedPtr> & twist_buffer,
// //   const double th_stopped_velocity_mps)
// // {
// //   for (const auto & twist : twist_buffer) {
// //     if (std::abs(twist->twist.linear.x) > th_stopped_velocity_mps) {
// //       return false;
// //     }
// //   }
// //   return true;
// // }

// // }  // namespace

// // size_t ScenarioSelectorNode::arrivedWhichGoal(const geometry_msgs::msg::Pose & current_pose, const double th_dist)
// //   {
// //     double min_dis = 100.0;
// //     double min_ang = 100.0;
// //     size_t min_index = 0;
// //     for(size_t index = 0; index < this->pre_seted_pose_vector.size(); index++)
// //     {
// //       const auto & p1 = current_pose.position;
// //       const auto & p2 = this->pre_seted_pose_vector[index].pose.position;
// //       const auto dist = std::hypot(p1.x - p2.x, p1.y - p2.y);
// //       const auto & o1 = current_pose.orientation;
// //       const auto & o2 = this->pre_seted_pose_vector[index].pose.orientation;
// //       double dotProduct = o1.x * o2.x + o1.y * o2.y + o1.z * o2.z + o1.w * o2.w;
// //       double angleDifference = 2.0 * std::acos(std::abs(dotProduct));
// //       RCLCPP_INFO(this->get_logger(), "goal %ld dis is %f, ang is %f",index,dist,angleDifference);
// //       if(dist <= th_dist*1.5 && angleDifference <= 0.3)
// //       {
// //         if(min_dis > dist)
// //         {
// //           min_dis = dist;
// //           if(min_ang > angleDifference)
// //           {
// //             min_ang = angleDifference;
// //             min_index = index;
// //           }
          
// //         }
// //       }
// //     }
// //     if(std::fabs(min_dis - 100.00) < 0.0001)
// //     {
// //       return 60;
// //     }
// //     else{
// //       return min_index;
// //     }
    
// //   }


// // autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr
// // ScenarioSelectorNode::getScenarioTrajectory(const std::string & scenario)
// // {
// //   if (scenario == tier4_planning_msgs::msg::Scenario::LANEDRIVING) {
// //     return lane_driving_trajectory_;
// //   }
// //   if (scenario == tier4_planning_msgs::msg::Scenario::PARKING) {
// //     return parking_trajectory_;
// //   }
// //   RCLCPP_ERROR_STREAM(this->get_logger(), "invalid scenario argument: " << scenario);
// //   return lane_driving_trajectory_;
// // }

// // // std::string ScenarioSelectorNode::selectScenarioByPosition()
// // // {
// // //   return tier4_planning_msgs::msg::Scenario::PARKING;
// // //   const auto is_in_lane = isInLane(lanelet_map_ptr_, current_pose_->pose.position);
// // //   const auto is_goal_in_lane = isInLane(lanelet_map_ptr_, route_->goal_pose.position);
// // //   const auto is_in_parking_lot = isInParkingLot(lanelet_map_ptr_, current_pose_->pose);

// // //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::EMPTY) {
// // //     if (is_in_lane && is_goal_in_lane) {
// // //       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
// // //     } else if (is_in_parking_lot) {
// // //       return tier4_planning_msgs::msg::Scenario::PARKING;
// // //     } else {
// // //       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
// // //     }
// // //   }

// // //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::LANEDRIVING) {
// // //     if (is_in_parking_lot && !is_goal_in_lane) {
// // //       return tier4_planning_msgs::msg::Scenario::PARKING;
// // //     }
// // //   }

// // //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::PARKING) {
// // //     if (is_parking_completed_ && is_in_lane) {
// // //       is_parking_completed_ = false;
// // //       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
// // //     }
// // //   }

// // //   return current_scenario_;
// // // }

// // std::string ScenarioSelectorNode::selectScenarioByPosition()
// // {
// //   // return tier4_planning_msgs::msg::Scenario::PARKING;
// //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::EMPTY) {
// //       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
// //   }

// //   size_t goal_index = arrivedWhichGoal(current_pose_->pose, th_arrived_distance_m_);
// //   RCLCPP_INFO(this->get_logger(),"goal_index is %ld",goal_index);
// //   switch (goal_index)
// //   {
// //     case 0:
// //     this->drive_stage = goal_index;
// //     this->change_freespace_param(0);
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
    
// //     break;

// //     case 1:
// //     this->drive_stage = goal_index;
// //     // this->change_freespace_param(0);
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
    
// //     break;

// //     case 2:
// //     this->drive_stage = goal_index;
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
    
// //     break;

// //     case 3:
// //     this->drive_stage = goal_index;
// //     // this->change_freespace_param(3);
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //     break;
// //     case 4:
// //     this->drive_stage = goal_index;
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //     break;
// //     case 5:
// //     this->drive_stage = goal_index;
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //      break;
// //     case 6:
// //     this->drive_stage = goal_index;
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //     break;
// //     case 7:
// //     this->drive_stage = goal_index;
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //     break;
// //     case 8:
// //     this->drive_stage = goal_index;
// //     // this->change_pid_param();
    
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //     break;
// //     case 9:
// //     this->drive_stage = goal_index;
// //     this->change_freespace_param(0);
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //     case 10:
// //     this->drive_stage = goal_index;
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //     break;

// //     case 11:
// //     this->drive_stage = goal_index;
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //     break;

// //     case 12:
// //     this->drive_stage = goal_index;
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //     break;

// //     case 13:
// //     this->drive_stage = goal_index;
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //     break;
// //     case 14:
// //     this->drive_stage = goal_index;
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //     break;
    

// //     default:
// //     return tier4_planning_msgs::msg::Scenario::PARKING;
// //       break;
// //   }
// //   return current_scenario_;
// // }

// // void ScenarioSelectorNode::updateCurrentScenario()
// // {

// //   const auto prev_scenario = current_scenario_;
// //   // RCLCPP_INFO_STREAM(this->get_logger(), "updateCurrentScenario");
// //   const auto scenario_trajectory = getScenarioTrajectory(current_scenario_);
// //   const auto is_near_trajectory_end =
// //     isNearTrajectoryEnd(scenario_trajectory, current_pose_->pose, th_arrived_distance_m_);

// //   const auto is_stopped = isStopped(twist_buffer_, th_stopped_velocity_mps_);

// //   if (is_near_trajectory_end && is_stopped) {
// //     current_scenario_ = selectScenarioByPosition();
// //   }

// //   if (current_scenario_ != prev_scenario) {
   
// //     RCLCPP_INFO_STREAM(
// //       this->get_logger(), "scenario changed: " << prev_scenario << " -> " << current_scenario_);
// //     this->is_published_goal = false;
// //     this->is_engaged_goal = false;
// //   }

// //   //just for test 
// //   if (inSpecialPosition())
// //   {
// //     if(this->drive_stage == 2)
// //     {
// //       this->drive_stage = this->drive_stage + 1;
// //       RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing goal pose.");
// //       goal_pos_publisher->publish(pre_seted_pose_vector[this->drive_stage + 1]);
// //       engage_publisher->publish(createEngageMessage());
// //     }
    
// //   }
// //   engage_publisher->publish(createEngageMessage());
// // }

// // bool ScenarioSelectorNode::inSpecialPosition()
// // {
// //   for(size_t index = 0; index < this->pre_seted_special_pose_vector.size(); index++)
// //     {
// //       const auto & p1 = current_pose_->pose.position;
// //       const auto & p2 = this->pre_seted_special_pose_vector[index].pose.position;
// //       const auto dist = std::hypot(p1.x - p2.x, p1.y - p2.y);
// //       const auto & o1 = current_pose_->pose.orientation;
// //       const auto & o2 = this->pre_seted_special_pose_vector[index].pose.orientation;
// //       double dotProduct = o1.x * o2.x + o1.y * o2.y + o1.z * o2.z + o1.w * o2.w;
// //       double angleDifference = 2.0 * std::acos(std::abs(dotProduct));
// //       RCLCPP_INFO(this->get_logger(), "poseindex %ld dis is %f, ang is %f",index,dist,angleDifference);
// //       if(dist <= 1.5 && angleDifference <= 0.5)
// //       {
// //         return true;
// //       }
// //     }
// //   return false;
// // }

// // void ScenarioSelectorNode::onMap(
// //   const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg)
// // {
// //   lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
// //   lanelet::utils::conversion::fromBinMsg(
// //     *msg, lanelet_map_ptr_, &traffic_rules_ptr_, &routing_graph_ptr_);
// //   route_handler_ = std::make_shared<route_handler::RouteHandler>(*msg);
// // }

// // void ScenarioSelectorNode::onRoute(
// //   const autoware_planning_msgs::msg::LaneletRoute::ConstSharedPtr msg)
// // {
// //   route_ = msg;
// //   // current_scenario_ = tier4_planning_msgs::msg::Scenario::EMPTY;
// // }

// // void ScenarioSelectorNode::onOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
// // {
// //   auto twist = std::make_shared<geometry_msgs::msg::TwistStamped>();
// //   twist->header = msg->header;
// //   twist->twist = msg->twist.twist;

// //   twist_ = twist;
// //   twist_buffer_.push_back(twist);

// //   // Delete old data in buffer
// //   while (true) {
// //     const auto time_diff =
// //       rclcpp::Time(msg->header.stamp) - rclcpp::Time(twist_buffer_.front()->header.stamp);

// //     if (time_diff.seconds() < th_stopped_time_sec_) {
// //       break;
// //     }

// //     twist_buffer_.pop_front();
// //   }
// // }

// // void ScenarioSelectorNode::onParkingState(const std_msgs::msg::Bool::ConstSharedPtr msg)
// // {
// //   is_parking_completed_ = msg->data;
// // }

// // bool ScenarioSelectorNode::isDataReady()
// // {
// //   if (!current_pose_) {
// //     RCLCPP_INFO(this->get_logger(), "Waiting for current pose.");
// //     return false;
// //   }

// //   if (!lanelet_map_ptr_) {
// //     RCLCPP_INFO(this->get_logger(), "Waiting for lanelet map.");
// //     return false;
// //   }

// //   // if (!route_) {
// //   //   RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for route.");
// //   //   return false;
// //   // }

// //   // if (!twist_) {
// //   //   RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for twist.");
// //   //   return false;
// //   // }

// //   // Check route handler is ready
// //   route_handler_->setRoute(*route_);
// //   if (!route_handler_->isHandlerReady()) {
// //     RCLCPP_WARN_THROTTLE(
// //       this->get_logger(), *this->get_clock(), 5000, "Waiting for route handler.");
// //     return false;
// //   }

// //   return true;
// // }

// // void ScenarioSelectorNode::onTimer()
// // {
// //   // RCLCPP_INFO(this->get_logger(), "selectScenarioByPosition onTimer");
// //   current_pose_ = getCurrentPose(tf_buffer_, this->get_logger());

// //   // if (!isDataReady()) {
// //   //   RCLCPP_INFO(
// //   //     this->get_logger(), "DataNotReady");
// //   //   return;
// //   // }

// //   // Initialize Scenario
// //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::EMPTY) {
// //     current_scenario_ = selectScenarioByPosition();
// //   }
// //   // RCLCPP_INFO(this->get_logger(), "selectScenarioByPosition");
// //   updateCurrentScenario();
// //   tier4_planning_msgs::msg::Scenario scenario;
// //   scenario.current_scenario = current_scenario_;

// //   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::PARKING) {
// //     scenario.activating_scenarios.push_back(current_scenario_);
// //   }

// //   pub_scenario_->publish(scenario);
// // }




// // void ScenarioSelectorNode::onLaneDrivingTrajectory(
// //   const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
// // {
// //   lane_driving_trajectory_ = msg;

// //   if (current_scenario_ != tier4_planning_msgs::msg::Scenario::LANEDRIVING) {
// //     return;
// //   }

// //   publishTrajectory(msg);
// // }

// // void ScenarioSelectorNode::onParkingTrajectory(
// //   const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
// // {
// //   parking_trajectory_ = msg;

// //   if (current_scenario_ != tier4_planning_msgs::msg::Scenario::PARKING) {
// //     return;
// //   }

// //   publishTrajectory(msg);
// // }

// // void ScenarioSelectorNode::publishTrajectory(
// //   const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
// // {
// //   const auto now = this->now();
// //   const auto delay_sec = (now - msg->header.stamp).seconds();
// //   if (delay_sec <= th_max_message_delay_sec_) {
// //     pub_trajectory_->publish(*msg);
// //   } else {
// //     RCLCPP_WARN_THROTTLE(
// //       this->get_logger(), *this->get_clock(), std::chrono::milliseconds(1000).count(),
// //       "trajectory is delayed: scenario = %s, delay = %f, th_max_message_delay = %f",
// //       current_scenario_.c_str(), delay_sec, th_max_message_delay_sec_);
// //   }
// // }

// // void ScenarioSelectorNode::goalCallback(const AutowareState& msg)
// // {
// //   if(this->drive_stage >= 50)
// //     {
// //       // RCLCPP_INFO(this->get_logger(), "Cannot arrive any goal");
// //       return;
// //     }
// //   RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: current_scenario_ is %s. is_published_goal is %d. this->drive_stage is %d, is_engage_goal is %d",current_scenario_.c_str(),this->is_published_goal,this->drive_stage,this->is_engaged_goal);
 
// //   if (!current_pose_) {
// //     return;
// //   }
  
// //   switch (msg.state) {
// //     case AutowareState::WAITING_FOR_ROUTE:
// //       if(!this->is_published_goal)
// //       {
// //         RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing goal pose.");
// //         goal_pos_publisher->publish(pre_seted_pose_vector[this->drive_stage + 1]);
// //         this->is_published_goal = true;
// //       }
// //       break;
// //     case AutowareState::ARRIVED_GOAL:
// //       this->is_published_goal = false;
// //       this->is_engaged_goal = false;
// //       if(!this->is_published_goal)
// //       {
// //         RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing goal pose.");
// //         goal_pos_publisher->publish(pre_seted_pose_vector[this->drive_stage + 1]);
// //         this->is_published_goal = true;
// //       }
// //       break;
    
// //     case AutowareState::PLANNING:
// //       RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Planning...");
// //       if(!this->is_engaged_goal)
// //       {
// //         RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing Planning engage message.");
// //         engage_publisher->publish(createEngageMessage());
// //         this->is_engaged_goal = true;
// //       }
// //       break;
// //     case AutowareState::WAITING_FOR_ENGAGE:
// //       RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: WAITING_FOR_ENGAGE.");
// //       if(!this->is_engaged_goal)
// //       {
// //         RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing engage message.");
// //         engage_publisher->publish(createEngageMessage());
// //         this->is_engaged_goal = true;
// //       }
// //       break;
// //     default:
// //       break;
// //   }
// // }
// // void ScenarioSelectorNode::initPreSetedPose()
// // {
// //   auto msg = PoseStamped();

// //  //start--0
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3818.806640625;
// // //   msg.pose.position.y = 73772.3046875;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = -0.9712304294215052;
// // //   msg.pose.orientation.w = 0.23814166574902115;
// // //   pre_seted_pose_vector.push_back(msg);

// // // //box1--1
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3796.34033203125;
// // //   msg.pose.position.y = 73761.75;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = -0.9719930786142853;
// // //   msg.pose.orientation.w = 0.23500947880016188;
// // //   pre_seted_pose_vector.push_back(msg);

// // // //box2--2
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3789.184326171875;
// // //   msg.pose.position.y = 73755.6640625;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = -0.9719930786142853;
// // //   msg.pose.orientation.w = 0.23500947880016188;
// // //   pre_seted_pose_vector.push_back(msg);


// // // //box3--3
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3779.483642578125;
// // //   msg.pose.position.y = 73753.0390625;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = -0.962210886752998;
// // //   msg.pose.orientation.w = 0.27230536060461474;
// // //   pre_seted_pose_vector.push_back(msg);


// // //  //boxes--0
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3757.973388671875;
// // //   msg.pose.position.y = 73738.6484375;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = -0.9724300171876937;
// // //   msg.pose.orientation.w = 0.23319490061393244;
// // //   pre_seted_pose_vector.push_back(msg);

// // // // //boxes--0  ---------now
// // // //   msg.header.frame_id = "map";
// // // //   msg.pose.position.x = 3754.98876953125;
// // // //   msg.pose.position.y = 73737.453125;
// // // //   msg.pose.position.z = 0.0;
// // // //   msg.pose.orientation.x = 0.0;
// // // //   msg.pose.orientation.y = 0.0;
// // // //   msg.pose.orientation.z = -0.9724300171876937;
// // // //   msg.pose.orientation.w = 0.23319490061393244;
// // // //   pre_seted_pose_vector.push_back(msg);


// // // //honggang--5


// // //   // msg.header.frame_id = "map";
// // //   // msg.pose.position.x = 3745.2490234375;
// // //   // msg.pose.position.y = 73743.7109375;
// // //   // msg.pose.position.z = 0.0;
// // //   // msg.pose.orientation.x = 0.0;
// // //   // msg.pose.orientation.y = 0.0;
// // //   // msg.pose.orientation.z = 0.8693105746034401;
// // //   // msg.pose.orientation.w = 0.49426624898189925;
// // //   // pre_seted_pose_vector.push_back(msg);

// // //   // msg.header.frame_id = "map";
// // //   // msg.pose.position.x = 3756.67431640625;
// // //   // msg.pose.position.y = 73738.484375;
// // //   // msg.pose.position.z = 0.0;
// // //   // msg.pose.orientation.x = 0.0;
// // //   // msg.pose.orientation.y = 0.0;
// // //   // msg.pose.orientation.z = -0.9654916720794067;
// // //   // msg.pose.orientation.w = 0.2604339285602234;
// // //   // pre_seted_pose_vector.push_back(msg);
// // // //--1
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3745.2490234375;
// // //   msg.pose.position.y = 73743.7109375;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = 0.8693105746034401;
// // //   msg.pose.orientation.w = 0.49426624898189925;
// // //   pre_seted_pose_vector.push_back(msg);
  



// // //   //  L first 
// // //   // msg.header.frame_id = "map";
// // //   // msg.pose.position.x = 3749.715576171875;
// // //   // msg.pose.position.y = 73734.71875;
// // //   // msg.pose.position.z = 0.0;
// // //   // msg.pose.orientation.x = 0.0;
// // //   // msg.pose.orientation.y = 0.0;
// // //   // msg.pose.orientation.z = 0.7579653122371283;
// // //   // msg.pose.orientation.w = 0.6522948608147029;
// // //   // pre_seted_pose_vector.push_back(msg);
  
// // //   // //  L first replace
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3750.13525390625;
// // //   msg.pose.position.y = 73734.7890625;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = 0.7525766947068779;
// // //   msg.pose.orientation.w = 0.658504607868518;
// // //   pre_seted_pose_vector.push_back(msg);

  
  
// // //   // //  L second
// // //   // // msg.header.frame_id = "map";
// // //   // // msg.pose.position.x = 3749.709716796875;
// // //   // // msg.pose.position.y = 73735.859375;
// // //   // // msg.pose.position.z = 0.0;
// // //   // // msg.pose.orientation.x = 0.0;
// // //   // // msg.pose.orientation.y = 0.0;
// // //   // // msg.pose.orientation.z = 0.5708294720816296;
// // //   // // msg.pose.orientation.w = 0.8210686413467562;
// // //   // // pre_seted_pose_vector.push_back(msg);
  
// // //   //  //  L third
// // //   // // msg.header.frame_id = "map";
// // //   // // msg.pose.position.x = 3750.218505859375;
// // //   // // msg.pose.position.y = 73736.234375;
// // //   // // msg.pose.position.z = 0.0;
// // //   // // msg.pose.orientation.x = 0.0;
// // //   // // msg.pose.orientation.y = 0.0;
// // //   // // msg.pose.orientation.z = 0.43712374276198984;
// // //   // // msg.pose.orientation.w = 0.8994013750899815;
// // //   // // pre_seted_pose_vector.push_back(msg);

// // //   //  //  L third replace
// // //   // msg.header.frame_id = "map";
// // //   // msg.pose.position.x = 3750.20263671875;
// // //   // msg.pose.position.y = 73736.03125;
// // //   // msg.pose.position.z = 0.0;
// // //   // msg.pose.orientation.x = 0.0;
// // //   // msg.pose.orientation.y = 0.0;
// // //   // msg.pose.orientation.z = 0.5799025524453801;
// // //   // msg.pose.orientation.w = 0.8146858472241513;
// // //   // pre_seted_pose_vector.push_back(msg);

  
// // //   //  //  L third replace  ---
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3750.34814453125;
// // //   msg.pose.position.y = 73736.6875;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = 0.6017139825542301;
// // //   msg.pose.orientation.w = 0.7987116395788455;
// // //   pre_seted_pose_vector.push_back(msg);
  


// // //   // msg.header.frame_id = "map";
// // //   // msg.pose.position.x = 3751.2109375;
// // //   // msg.pose.position.y = 73737.1015625;
// // //   // msg.pose.position.z = 0.0;
// // //   // msg.pose.orientation.x = 0.0;
// // //   // msg.pose.orientation.y = 0.0;
// // //   // msg.pose.orientation.z = 0.3214301106708757;
// // //   // msg.pose.orientation.w = 0.9469333049133443;
// // //   // pre_seted_pose_vector.push_back(msg);

// // // //--2
// // //   // msg.header.frame_id = "map";
// // //   // msg.pose.position.x = 3749.689697265625;
// // //   // msg.pose.position.y = 73735.3203125;
// // //   // msg.pose.position.z = 0.0;
// // //   // msg.pose.orientation.x = 0.0;
// // //   // msg.pose.orientation.y = 0.0;
// // //   // msg.pose.orientation.z = 0.6678927619723282;
// // //   // msg.pose.orientation.w = 0.7442575216314411;
// // //   // pre_seted_pose_vector.push_back(msg);
// // // //--3
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3755.7021484375;
// // //   msg.pose.position.y = 73739.1640625;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = 0.24927759678599598;
// // //   msg.pose.orientation.w = 0.9684320728582869;
// // //   pre_seted_pose_vector.push_back(msg);

// // //  // -3 replace
// // //   // msg.header.frame_id = "map";
// // //   // msg.pose.position.x = 3755.517333984375;
// // //   // msg.pose.position.y = 73738.7421875;
// // //   // msg.pose.position.z = 0.0;
// // //   // msg.pose.orientation.x = 0.0;
// // //   // msg.pose.orientation.y = 0.0;
// // //   // msg.pose.orientation.z = 0.26645980910882694;
// // //   // msg.pose.orientation.w = 0.9638460303023961;
// // //   // pre_seted_pose_vector.push_back(msg);



// // //   //--4
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3745.0830078125;
// // //   msg.pose.position.y = 73743.4453125;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = -0.5178196572501046;
// // //   msg.pose.orientation.w = 0.8554898027243716;
// // //   pre_seted_pose_vector.push_back(msg);
// // // //--5
// // // //honggang

// // // //drive 1
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3739.350830078125;
// // //   msg.pose.position.y = 73754.3515625;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = -0.49164540017165514;
// // //   msg.pose.orientation.w = 0.8707954986620298;
// // //   pre_seted_pose_vector.push_back(msg);
// // // //--6
// // // //drive 1 middle
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3727.3359375;
// // //   msg.pose.position.y = 73757.21875;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = 0.2408351489047307;
// // //   msg.pose.orientation.w = 0.9705660364200038;
// // //   pre_seted_pose_vector.push_back(msg);

// // //   //--7
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3718.353271484375;
// // //   msg.pose.position.y = 73750.734375;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = 0.6679334714495515;
// // //   msg.pose.orientation.w = 0.7442209871519018;
// // //   pre_seted_pose_vector.push_back(msg);
// // // //--8
// // //--9 
// //   msg.header.frame_id = "map";
// //   msg.pose.position.x = 3720.607421875;
// //   msg.pose.position.y = 73744.21875;
// //   msg.pose.position.z = 0.0;
// //   msg.pose.orientation.x = 0.0;
// //   msg.pose.orientation.y = 0.0;
// //   msg.pose.orientation.z = -0.5521686665940426;
// //   msg.pose.orientation.w = 0.8337324292791765;
// //   pre_seted_pose_vector.push_back(msg);
// // //drive 2
// //   msg.header.frame_id = "map";
// //   msg.pose.position.x = 3713.041015625;
// //   msg.pose.position.y = 73733.625;
// //   msg.pose.position.z = 0.0;
// //   msg.pose.orientation.x = 0.0;
// //   msg.pose.orientation.y = 0.0;
// //   msg.pose.orientation.z = -0.9706179723799822;
// //   msg.pose.orientation.w = 0.24062575026994124;
// //   pre_seted_pose_vector.push_back(msg);



// // //--9 
// //   // msg.header.frame_id = "map";
// //   // msg.pose.position.x = 3708.572509765625;
// //   // msg.pose.position.y = 73731.375;
// //   // msg.pose.position.z = 0.0;
// //   // msg.pose.orientation.x = 0.0;
// //   // msg.pose.orientation.y = 0.0;
// //   // msg.pose.orientation.z = 0.24365928570205184;
// //   // msg.pose.orientation.w = 0.9698608933713978;
// //   // pre_seted_pose_vector.push_back(msg);
// // //--9

// // // //drive3
// // //   msg.header.frame_id = "map";
// // //   msg.pose.position.x = 3717.45068359375;
// // //   msg.pose.position.y = 73735.7109375;
// // //   msg.pose.position.z = 0.0;
// // //   msg.pose.orientation.x = 0.0;
// // //   msg.pose.orientation.y = 0.0;
// // //   msg.pose.orientation.z = 0.23903341739572623;
// // //   msg.pose.orientation.w = 0.9710113415239394;
// // //   pre_seted_pose_vector.push_back(msg);

// // //drive4
// // msg.header.frame_id = "map";
// // msg.pose.position.x = 3701.373779296875;
// // msg.pose.position.y = 73744.7890625;
// // msg.pose.position.z = 0.0;
// // msg.pose.orientation.x = 0.0;
// // msg.pose.orientation.y = 0.0;
// // msg.pose.orientation.z = -0.5372075731791744;
// // msg.pose.orientation.w = 0.8434500716218726;
// // pre_seted_pose_vector.push_back(msg);

// //   for(auto pose : pre_seted_pose_vector)
// //   {
// //     RCLCPP_INFO(this->get_logger(), "pre_seted_pose_vector x is %f",pose.pose.position.x);
// //   }



// // //special pose
// // msg.header.frame_id = "map";
// // msg.pose.position.x = 3754.385009765625;
// // msg.pose.position.y = 73739.8046875;
// // msg.pose.position.z = 0.0;
// // msg.pose.orientation.x = 0.0;
// // msg.pose.orientation.y = 0.0;
// // msg.pose.orientation.z = 0.1939932506522466;
// // msg.pose.orientation.w = 0.9810028637579885;
// // pre_seted_special_pose_vector.push_back(msg);

// // }

// // // std::string exec(const char* cmd) {
// // //     std::array<char, 128> buffer;
// // //     std::string result;
// // //     std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
// // //     if (!pipe) throw std::runtime_error("popen() failed!");
// // //     while (!feof(pipe.get())) {
// // //         if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
// // //             result += buffer.data();
// // //     }
// // //     return result;
// // // }


// // void ScenarioSelectorNode::change_freespace_param(const int & num)
// // {
// //     // std::string result = exec("ros2 param set /planning/scenario_planning/parking/freespace_planner jump_if_size_less 1");
// //     // RCLCPP_INFO(this->get_logger(), "Parameter set result is %s",result.c_str());
// //   // // 修改节点的参数值
// //     std::vector<rclcpp::Parameter> params;
// //     params.push_back(rclcpp::Parameter("jump_if_size_less", num));

// //     auto result = this->parameter_client_freespace->set_parameters_atomically(params);
// //     if(result.successful)
// //     {
// //         RCLCPP_INFO(this->get_logger(), "Parameter set successfully");
// //     }
// //     else
// //     {
// //         RCLCPP_ERROR(this->get_logger(), "Failed to set parameter");
// //     }
// //   }


// //   void ScenarioSelectorNode::change_pid_param()
// // {
// //   // // 修改节点的参数值
// //   // 修改节点的参数值
// //     std::vector<rclcpp::Parameter> params;
// //     params.push_back(rclcpp::Parameter("stopping_state_stop_dist", 0.1));

// //     auto result = this->parameter_client_pid->set_parameters_atomically(params);
// //     if(result.successful)
// //     {
// //         RCLCPP_INFO(this->get_logger(), "Parameter set successfully");
// //     }
// //     else
// //     {
// //         RCLCPP_ERROR(this->get_logger(), "Failed to set parameter");
// //     }
// //   }

  
// // ScenarioSelectorNode::ScenarioSelectorNode(const rclcpp::NodeOptions & node_options)
// // : Node("scenario_selector", node_options),
// //   tf_buffer_(this->get_clock()),
// //   tf_listener_(tf_buffer_),
// //   current_scenario_(tier4_planning_msgs::msg::Scenario::EMPTY),
// //   update_rate_(this->declare_parameter<double>("update_rate", 10.0)),
// //   th_max_message_delay_sec_(this->declare_parameter<double>("th_max_message_delay_sec", 1.0)),
// //   th_arrived_distance_m_(this->declare_parameter<double>("th_arrived_distance_m", 1.0)),
// //   th_stopped_time_sec_(this->declare_parameter<double>("th_stopped_time_sec", 1.0)),
// //   th_stopped_velocity_mps_(this->declare_parameter<double>("th_stopped_velocity_mps", 0.01)),
// //   is_parking_completed_(false),
// //   drive_stage(-1),
// //   is_published_goal(false),
// //   is_engaged_goal(false)
// // {
  

    
// //   // Input
// //   sub_lane_driving_trajectory_ =
// //     this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
// //       "input/lane_driving/trajectory", rclcpp::QoS{1},
// //       std::bind(&ScenarioSelectorNode::onLaneDrivingTrajectory, this, std::placeholders::_1));

// //   sub_parking_trajectory_ = this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
// //     "input/parking/trajectory", rclcpp::QoS{1},
// //     std::bind(&ScenarioSelectorNode::onParkingTrajectory, this, std::placeholders::_1));

// //   sub_lanelet_map_ = this->create_subscription<autoware_auto_mapping_msgs::msg::HADMapBin>(
// //     "input/lanelet_map", rclcpp::QoS{1}.transient_local(),
// //     std::bind(&ScenarioSelectorNode::onMap, this, std::placeholders::_1));
// //   sub_route_ = this->create_subscription<autoware_planning_msgs::msg::LaneletRoute>(
// //     "input/route", rclcpp::QoS{1}.transient_local(),
// //     std::bind(&ScenarioSelectorNode::onRoute, this, std::placeholders::_1));
// //   sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
// //     "input/odometry", rclcpp::QoS{100},
// //     std::bind(&ScenarioSelectorNode::onOdom, this, std::placeholders::_1));
// //   sub_parking_state_ = this->create_subscription<std_msgs::msg::Bool>(
// //     "is_parking_completed", rclcpp::QoS{100},
// //     std::bind(&ScenarioSelectorNode::onParkingState, this, std::placeholders::_1));

// //   // Output
// //   pub_scenario_ =
// //     this->create_publisher<tier4_planning_msgs::msg::Scenario>("output/scenario", rclcpp::QoS{1});
// //   pub_trajectory_ = this->create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
// //     "output/trajectory", rclcpp::QoS{1});


// //   //ai_challange_control
// //   // Publishers
// //   engage_publisher =
// //     this->create_publisher<Engage>("output/engage", 1);
// //   goal_pos_publisher =
// //     this->create_publisher<PoseStamped>("output/goal", 1);

// //   // Subscribers
// //   state_subscriber = this->create_subscription<AutowareState>(
// //     "input/state", 1, std::bind(&ScenarioSelectorNode::goalCallback, this, std::placeholders::_1));

// //   // Timer Callback
// //   const auto period_ns = rclcpp::Rate(static_cast<double>(update_rate_)).period();

// //   timer_ = rclcpp::create_timer(
// //     this, get_clock(), period_ns, std::bind(&ScenarioSelectorNode::onTimer, this));
// //   // 创建一个参数客户端来修改参数
// //   auto freespace_manager = rclcpp::Node::make_shared("freespace_manager");
// //   this->parameter_client_freespace = std::make_shared<rclcpp::SyncParametersClient>(freespace_manager,"/planning/scenario_planning/parking/freespace_planner");
// //   auto pid_manager = rclcpp::Node::make_shared("pid_manager");
// //   // 创建一个参数客户端来修改参数
// //   this->parameter_client_pid = std::make_shared<rclcpp::SyncParametersClient>(pid_manager,"/control/trajectory_follower/controller_node_exe");

// //   this->initPreSetedPose();
// //   // Wait for first tf
// //   while (rclcpp::ok()) {
// //     try {
// //       tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
// //       break;
// //     } catch (tf2::TransformException & ex) {
// //       RCLCPP_DEBUG(this->get_logger(), "waiting for initial pose...");
// //       rclcpp::sleep_for(std::chrono::milliseconds(100));
// //     }
// //   }
// // }

// // #include <rclcpp_components/register_node_macro.hpp>
// // RCLCPP_COMPONENTS_REGISTER_NODE(ScenarioSelectorNode)





















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

#include "scenario_selector/scenario_selector_node.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/query.hpp>

#include <lanelet2_core/geometry/BoundingBox.h>
#include <lanelet2_core/geometry/Lanelet.h>
#include <lanelet2_core/geometry/LineString.h>
#include <lanelet2_core/geometry/Point.h>
#include <lanelet2_core/geometry/Polygon.h>

#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{
template <class T>
void onData(const T & data, T * buffer)
{
  *buffer = data;
}

using AutowareState = autoware_auto_system_msgs::msg::AutowareState;
using Engage = autoware_auto_vehicle_msgs::msg::Engage;
using PoseStamped = geometry_msgs::msg::PoseStamped;
Engage createEngageMessage()
{
  auto msg = Engage();
  msg.engage = true;
  return msg;
}
// std::shared_ptr<lanelet::ConstPolygon3d> findNearestParkinglot(
//   const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
//   const lanelet::BasicPoint2d & current_position)
// {
//   const auto all_parking_lots = lanelet::utils::query::getAllParkingLots(lanelet_map_ptr);

//   const auto linked_parking_lot = std::make_shared<lanelet::ConstPolygon3d>();
//   const auto result = lanelet::utils::query::getLinkedParkingLot(
//     current_position, all_parking_lots, linked_parking_lot.get());

//   if (result) {
//     return linked_parking_lot;
//   } else {
//     return {};
//   }
// }

geometry_msgs::msg::PoseStamped::ConstSharedPtr getCurrentPose(
  const tf2_ros::Buffer & tf_buffer, const rclcpp::Logger & logger)
{
  geometry_msgs::msg::TransformStamped tf_current_pose;

  try {
    tf_current_pose = tf_buffer.lookupTransform("map", "base_link", tf2::TimePointZero);
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(logger, "%s", ex.what());
    return nullptr;
  }

  geometry_msgs::msg::PoseStamped::SharedPtr p(new geometry_msgs::msg::PoseStamped());
  p->header = tf_current_pose.header;
  p->pose.orientation = tf_current_pose.transform.rotation;
  p->pose.position.x = tf_current_pose.transform.translation.x;
  p->pose.position.y = tf_current_pose.transform.translation.y;
  p->pose.position.z = tf_current_pose.transform.translation.z;

  return geometry_msgs::msg::PoseStamped::ConstSharedPtr(p);
}

// bool isInLane(
//   const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
//   const geometry_msgs::msg::Point & current_pos)
// {
//   const auto & p = current_pos;
//   const lanelet::Point3d search_point(lanelet::InvalId, p.x, p.y, p.z);

//   std::vector<std::pair<double, lanelet::Lanelet>> nearest_lanelets =
//     lanelet::geometry::findNearest(lanelet_map_ptr->laneletLayer, search_point.basicPoint2d(), 1);

//   if (nearest_lanelets.empty()) {
//     return false;
//   }

//   const auto nearest_lanelet = nearest_lanelets.front().second;

//   return lanelet::geometry::within(search_point, nearest_lanelet.polygon3d());
// }

// bool isInParkingLot(
//   const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
//   const geometry_msgs::msg::Pose & current_pose)
// {
//   const auto & p = current_pose.position;
//   const lanelet::Point3d search_point(lanelet::InvalId, p.x, p.y, p.z);

//   const auto nearest_parking_lot =
//     findNearestParkinglot(lanelet_map_ptr, search_point.basicPoint2d());

//   if (!nearest_parking_lot) {
//     return false;
//   }

//   return lanelet::geometry::within(search_point, nearest_parking_lot->basicPolygon());
// }



bool isNearTrajectoryEnd(
  const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr trajectory,
  const geometry_msgs::msg::Pose & current_pose, const double th_dist)
{
  if (!trajectory || trajectory->points.empty()) {
    return false;
  }

  const auto & p1 = current_pose.position;
  const auto & p2 = trajectory->points.back().pose.position;

  const auto dist = std::hypot(p1.x - p2.x, p1.y - p2.y);

  return dist < th_dist;
}

bool isStopped(
  const std::deque<geometry_msgs::msg::TwistStamped::ConstSharedPtr> & twist_buffer,
  const double th_stopped_velocity_mps)
{
  for (const auto & twist : twist_buffer) {
    if (std::abs(twist->twist.linear.x) > th_stopped_velocity_mps) {
      return false;
    }
  }
  return true;
}

}  // namespace

size_t ScenarioSelectorNode::arrivedWhichGoal(const geometry_msgs::msg::Pose & current_pose, const double th_dist)
  {
    double min_dis = 100.0;
    double min_ang = 100.0;
    size_t min_index = 0;
    for(size_t index = 0; index < this->pre_seted_pose_vector.size(); index++)
    {
      const auto & p1 = current_pose.position;
      const auto & p2 = this->pre_seted_pose_vector[index].pose.position;
      const auto dist = std::hypot(p1.x - p2.x, p1.y - p2.y);
      const auto & o1 = current_pose.orientation;
      const auto & o2 = this->pre_seted_pose_vector[index].pose.orientation;
      double dotProduct = o1.x * o2.x + o1.y * o2.y + o1.z * o2.z + o1.w * o2.w;
      double angleDifference = 2.0 * std::acos(std::abs(dotProduct));
      RCLCPP_INFO(this->get_logger(), "goal %ld dis is %f, ang is %f",index,dist,angleDifference);

      if(dist <= 1.5 && angleDifference <= 0.5){
        if(min_dis > dist){
          min_dis = dist;
          min_index = index;

        }
      }

      // {
      //   if(min_dis > dist)
      //   {
      //     min_dis = dist;
      //     if(min_ang > angleDifference)
      //     {
      //       min_ang = angleDifference;
      //       min_index = index;
      //     }
          
      //   }
      // }
    }
    if(std::fabs(min_dis - 100.00) < 0.0001)
    {
      return 60;
    }
    else{
      return min_index;
    }
    
  }


autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr
ScenarioSelectorNode::getScenarioTrajectory(const std::string & scenario)
{
  if (scenario == tier4_planning_msgs::msg::Scenario::LANEDRIVING) {
    return lane_driving_trajectory_;
  }
  if (scenario == tier4_planning_msgs::msg::Scenario::PARKING) {
    return parking_trajectory_;
  }
  RCLCPP_ERROR_STREAM(this->get_logger(), "invalid scenario argument: " << scenario);
  return lane_driving_trajectory_;
}

// std::string ScenarioSelectorNode::selectScenarioByPosition()
// {
//   return tier4_planning_msgs::msg::Scenario::PARKING;
//   const auto is_in_lane = isInLane(lanelet_map_ptr_, current_pose_->pose.position);
//   const auto is_goal_in_lane = isInLane(lanelet_map_ptr_, route_->goal_pose.position);
//   const auto is_in_parking_lot = isInParkingLot(lanelet_map_ptr_, current_pose_->pose);

//   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::EMPTY) {
//     if (is_in_lane && is_goal_in_lane) {
//       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
//     } else if (is_in_parking_lot) {
//       return tier4_planning_msgs::msg::Scenario::PARKING;
//     } else {
//       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
//     }
//   }

//   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::LANEDRIVING) {
//     if (is_in_parking_lot && !is_goal_in_lane) {
//       return tier4_planning_msgs::msg::Scenario::PARKING;
//     }
//   }

//   if (current_scenario_ == tier4_planning_msgs::msg::Scenario::PARKING) {
//     if (is_parking_completed_ && is_in_lane) {
//       is_parking_completed_ = false;
//       return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
//     }
//   }

//   return current_scenario_;
// }

std::string ScenarioSelectorNode::selectScenarioByPosition()
{
  // return tier4_planning_msgs::msg::Scenario::PARKING;
  if (current_scenario_ == tier4_planning_msgs::msg::Scenario::EMPTY) {
      return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
  }

  size_t goal_index = arrivedWhichGoal(current_pose_->pose, th_arrived_distance_m_);
  RCLCPP_INFO(this->get_logger(),"goal_index is %ld",goal_index);
  switch (goal_index)
  {
    case 0:
    this->drive_stage = goal_index;
    return tier4_planning_msgs::msg::Scenario::LANEDRIVING;
    
    break;

    case 1:
    this->drive_stage = goal_index;
    // this->change_freespace_param(0);
    return tier4_planning_msgs::msg::Scenario::PARKING;
    
    break;

    case 2:
    this->drive_stage = goal_index;
    return tier4_planning_msgs::msg::Scenario::PARKING;
    
    break;

    case 3:
    this->drive_stage = goal_index;
    // this->change_freespace_param(3);
    return tier4_planning_msgs::msg::Scenario::PARKING;
    break;
    case 4:
    this->drive_stage = goal_index;
    return tier4_planning_msgs::msg::Scenario::PARKING;
    break;
    case 5:
    this->drive_stage = goal_index;
    return tier4_planning_msgs::msg::Scenario::PARKING;
     break;
    case 6:
    this->drive_stage = goal_index;
    return tier4_planning_msgs::msg::Scenario::PARKING;
    break;
    case 7:
    this->drive_stage = goal_index;
    return tier4_planning_msgs::msg::Scenario::PARKING;
    break;
    case 8:
    this->drive_stage = goal_index;
    // this->change_pid_param();
    
    return tier4_planning_msgs::msg::Scenario::PARKING;
    break;
    case 9:
    this->drive_stage = goal_index;
    
    return tier4_planning_msgs::msg::Scenario::PARKING;
    case 10:
    this->change_freespace_param(0);
    this->drive_stage = goal_index;
    return tier4_planning_msgs::msg::Scenario::PARKING;
    break;

    case 11:
    // this->change_freespace_param_width(0.1);
    this->drive_stage = goal_index;
    return tier4_planning_msgs::msg::Scenario::PARKING;
    break;

    case 12:
    this->drive_stage = goal_index;
    return tier4_planning_msgs::msg::Scenario::PARKING;
    break;

    case 13:
    this->drive_stage = goal_index;
    return tier4_planning_msgs::msg::Scenario::PARKING;
    break;
    case 14:
    this->drive_stage = goal_index;
    return tier4_planning_msgs::msg::Scenario::PARKING;
    break;
    

    default:
    return tier4_planning_msgs::msg::Scenario::PARKING;
      break;
  }
  return current_scenario_;
}

void ScenarioSelectorNode::updateCurrentScenario()
{

  const auto prev_scenario = current_scenario_;
  // RCLCPP_INFO_STREAM(this->get_logger(), "updateCurrentScenario");
  const auto scenario_trajectory = getScenarioTrajectory(current_scenario_);
  const auto is_near_trajectory_end =
    isNearTrajectoryEnd(scenario_trajectory, current_pose_->pose, th_arrived_distance_m_);

  const auto is_stopped = isStopped(twist_buffer_, th_stopped_velocity_mps_);

  if (is_near_trajectory_end && is_stopped) {
    current_scenario_ = selectScenarioByPosition();
  }

  if (current_scenario_ != prev_scenario) {
   
    RCLCPP_INFO_STREAM(
      this->get_logger(), "scenario changed: " << prev_scenario << " -> " << current_scenario_);
    this->is_published_goal = false;
    this->is_engaged_goal = false;
  }

  //just for test 
  if (is_special_position)
  {
      this->drive_stage = this->drive_stage + 1;
      RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing goal pose. stage add one. the current stage is %d", this->drive_stage);
      goal_pos_publisher->publish(pre_seted_pose_vector[this->drive_stage + 1]);
      engage_publisher->publish(createEngageMessage());
      is_special_position=false;
    
  }
  // engage_publisher->publish(createEngageMessage());
}

bool ScenarioSelectorNode::inSpecialPosition()
{
  for(size_t index = 0; index < this->pre_seted_special_pose_vector.size(); index++)
    {
      const auto & p1 = current_pose_->pose.position;
      const auto & p2 = this->pre_seted_special_pose_vector[index].pose.position;
      const auto dist = std::hypot(p1.x - p2.x, p1.y - p2.y);
      const auto & o1 = current_pose_->pose.orientation;
      const auto & o2 = this->pre_seted_special_pose_vector[index].pose.orientation;
      double dotProduct = o1.x * o2.x + o1.y * o2.y + o1.z * o2.z + o1.w * o2.w;
      double angleDifference = 2.0 * std::acos(std::abs(dotProduct));
      RCLCPP_INFO(this->get_logger(), "poseindex %ld dis is %f, ang is %f",index,dist,angleDifference);
      if(dist <= 1.5 && angleDifference <= 0.5)
      {
        return true;
      }
    }
  return false;
}

void ScenarioSelectorNode::onMap(
  const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg)
{
  lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(
    *msg, lanelet_map_ptr_, &traffic_rules_ptr_, &routing_graph_ptr_);
  route_handler_ = std::make_shared<route_handler::RouteHandler>(*msg);
}

void ScenarioSelectorNode::onRoute(
  const autoware_planning_msgs::msg::LaneletRoute::ConstSharedPtr msg)
{
  route_ = msg;
  // current_scenario_ = tier4_planning_msgs::msg::Scenario::EMPTY;
}

void ScenarioSelectorNode::onOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  auto twist = std::make_shared<geometry_msgs::msg::TwistStamped>();
  twist->header = msg->header;
  twist->twist = msg->twist.twist;

  twist_ = twist;
  twist_buffer_.push_back(twist);

  // Delete old data in buffer
  while (true) {
    const auto time_diff =
      rclcpp::Time(msg->header.stamp) - rclcpp::Time(twist_buffer_.front()->header.stamp);

    if (time_diff.seconds() < th_stopped_time_sec_) {
      break;
    }

    twist_buffer_.pop_front();
  }
}

void ScenarioSelectorNode::onParkingState(const std_msgs::msg::Bool::ConstSharedPtr msg)
{
  is_parking_completed_ = msg->data;
}

void ScenarioSelectorNode::onSpecialPosition(const std_msgs::msg::Bool::ConstSharedPtr msg)
{
  is_special_position = msg->data;
}

bool ScenarioSelectorNode::isDataReady()
{
  if (!current_pose_) {
    RCLCPP_INFO(this->get_logger(), "Waiting for current pose.");
    return false;
  }

  if (!lanelet_map_ptr_) {
    RCLCPP_INFO(this->get_logger(), "Waiting for lanelet map.");
    return false;
  }

  // if (!route_) {
  //   RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for route.");
  //   return false;
  // }

  // if (!twist_) {
  //   RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for twist.");
  //   return false;
  // }

  // Check route handler is ready
  route_handler_->setRoute(*route_);
  if (!route_handler_->isHandlerReady()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000, "Waiting for route handler.");
    return false;
  }

  return true;
}

void ScenarioSelectorNode::onTimer()
{
  // RCLCPP_INFO(this->get_logger(), "selectScenarioByPosition onTimer");
  current_pose_ = getCurrentPose(tf_buffer_, this->get_logger());

  // if (!isDataReady()) {
  //   RCLCPP_INFO(
  //     this->get_logger(), "DataNotReady");
  //   return;
  // }

  // Initialize Scenario
  if (current_scenario_ == tier4_planning_msgs::msg::Scenario::EMPTY) {
    current_scenario_ = selectScenarioByPosition();
  }
  // RCLCPP_INFO(this->get_logger(), "selectScenarioByPosition");
  updateCurrentScenario();
  tier4_planning_msgs::msg::Scenario scenario;
  scenario.current_scenario = current_scenario_;

  if (current_scenario_ == tier4_planning_msgs::msg::Scenario::PARKING) {
    scenario.activating_scenarios.push_back(current_scenario_);
  }

  pub_scenario_->publish(scenario);
}




void ScenarioSelectorNode::onLaneDrivingTrajectory(
  const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
{
  lane_driving_trajectory_ = msg;

  if (current_scenario_ != tier4_planning_msgs::msg::Scenario::LANEDRIVING) {
    return;
  }

  publishTrajectory(msg);
}

void ScenarioSelectorNode::onParkingTrajectory(
  const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
{
  parking_trajectory_ = msg;

  if (current_scenario_ != tier4_planning_msgs::msg::Scenario::PARKING) {
    return;
  }

  publishTrajectory(msg);
}

void ScenarioSelectorNode::publishTrajectory(
  const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
{
  const auto now = this->now();
  const auto delay_sec = (now - msg->header.stamp).seconds();
  if (delay_sec <= th_max_message_delay_sec_) {
    pub_trajectory_->publish(*msg);
  } else {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), std::chrono::milliseconds(1000).count(),
      "trajectory is delayed: scenario = %s, delay = %f, th_max_message_delay = %f",
      current_scenario_.c_str(), delay_sec, th_max_message_delay_sec_);
  }
}

void ScenarioSelectorNode::goalCallback(const AutowareState& msg)
{
  if(this->drive_stage >= 50)
    {
      // RCLCPP_INFO(this->get_logger(), "Cannot arrive any goal");
      return;
    }
  RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: current_scenario_ is %s. is_published_goal is %d. this->drive_stage is %d, is_engage_goal is %d",current_scenario_.c_str(),this->is_published_goal,this->drive_stage,this->is_engaged_goal);
 
  if (!current_pose_) {
    return;
  }
  
  RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: %d",msg.state);

  switch (msg.state) {
    case AutowareState::WAITING_FOR_ROUTE:
      if(this->drive_stage == -1){
        RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: exit WAITING_FOR_ROUTE due to drive_stage is -1");
        break;
      }
      if(!this->is_published_goal)
      {
        RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing goal pose.  WAITING_FOR_ROUTE");
        goal_pos_publisher->publish(pre_seted_pose_vector[this->drive_stage + 1]);
        this->is_published_goal = true;
      }
      break;
    case AutowareState::ARRIVED_GOAL:
      this->is_published_goal = false;
      this->is_engaged_goal = false;
      if(!this->is_published_goal)
      {
        RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing goal pose. ARRIVED_GOAL");
        goal_pos_publisher->publish(pre_seted_pose_vector[this->drive_stage + 1]);
        this->is_published_goal = true;
      }
      break;
    
    case AutowareState::PLANNING:
      RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Planning...");
      if(this->drive_stage == -1){
        RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: exit Planning due to drive_stage is -1");
        break;
      }
      if(!this->is_engaged_goal)
      {
        RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing Planning engage message.");
        engage_publisher->publish(createEngageMessage());
        this->is_engaged_goal = true;
      }
      break;
    case AutowareState::WAITING_FOR_ENGAGE:
      RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: WAITING_FOR_ENGAGE.");
      if(this->drive_stage == -1){
        RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: exit WAITING_FOR_ENGAGE due to drive_stage is -1");
        break;
      }
      if(!this->is_engaged_goal)
      {
        RCLCPP_INFO(this->get_logger(), "[AIChallengeSample]: Publishing engage message.");
        engage_publisher->publish(createEngageMessage());
        this->is_engaged_goal = true;
      }
      break;
    default:
      break;
  }
}
void ScenarioSelectorNode::initPreSetedPose()
{
  auto msg = PoseStamped();

 //start--0
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3818.806640625;
//   msg.pose.position.y = 73772.3046875;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = -0.9712304294215052;
//   msg.pose.orientation.w = 0.23814166574902115;
//   pre_seted_pose_vector.push_back(msg);

// //box1--1
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3796.34033203125;
//   msg.pose.position.y = 73761.75;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = -0.9719930786142853;
//   msg.pose.orientation.w = 0.23500947880016188;
//   pre_seted_pose_vector.push_back(msg);

// //box2--2
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3789.184326171875;
//   msg.pose.position.y = 73755.6640625;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = -0.9719930786142853;
//   msg.pose.orientation.w = 0.23500947880016188;
//   pre_seted_pose_vector.push_back(msg);


// //box3--3
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3779.483642578125;
//   msg.pose.position.y = 73753.0390625;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = -0.962210886752998;
//   msg.pose.orientation.w = 0.27230536060461474;
//   pre_seted_pose_vector.push_back(msg);


 //boxes--0
  msg.header.frame_id = "map";
  msg.pose.position.x = 3757.973388671875;
  msg.pose.position.y = 73738.6484375;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = -0.9724300171876937;
  msg.pose.orientation.w = 0.23319490061393244;
  pre_seted_pose_vector.push_back(msg);

// //boxes--0  ---------now
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3754.98876953125;
//   msg.pose.position.y = 73737.453125;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = -0.9724300171876937;
//   msg.pose.orientation.w = 0.23319490061393244;
//   pre_seted_pose_vector.push_back(msg);


//honggang--5


  // msg.header.frame_id = "map";
  // msg.pose.position.x = 3745.2490234375;
  // msg.pose.position.y = 73743.7109375;
  // msg.pose.position.z = 0.0;
  // msg.pose.orientation.x = 0.0;
  // msg.pose.orientation.y = 0.0;
  // msg.pose.orientation.z = 0.8693105746034401;
  // msg.pose.orientation.w = 0.49426624898189925;
  // pre_seted_pose_vector.push_back(msg);

  // msg.header.frame_id = "map";
  // msg.pose.position.x = 3756.67431640625;
  // msg.pose.position.y = 73738.484375;
  // msg.pose.position.z = 0.0;
  // msg.pose.orientation.x = 0.0;
  // msg.pose.orientation.y = 0.0;
  // msg.pose.orientation.z = -0.9654916720794067;
  // msg.pose.orientation.w = 0.2604339285602234;
  // pre_seted_pose_vector.push_back(msg);
//--1
  msg.header.frame_id = "map";
  msg.pose.position.x = 3745.2490234375;
  msg.pose.position.y = 73743.7109375;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = 0.8693105746034401;
  msg.pose.orientation.w = 0.49426624898189925;
  pre_seted_pose_vector.push_back(msg);
  



  //  L first 
  // msg.header.frame_id = "map";
  // msg.pose.position.x = 3749.715576171875;
  // msg.pose.position.y = 73734.71875;
  // msg.pose.position.z = 0.0;
  // msg.pose.orientation.x = 0.0;
  // msg.pose.orientation.y = 0.0;
  // msg.pose.orientation.z = 0.7579653122371283;
  // msg.pose.orientation.w = 0.6522948608147029;
  // pre_seted_pose_vector.push_back(msg);
  
  // //  L first replace
  msg.header.frame_id = "map";
  msg.pose.position.x = 3750.13525390625;
  msg.pose.position.y = 73734.7890625;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = 0.7525766947068779;
  msg.pose.orientation.w = 0.658504607868518;
  pre_seted_pose_vector.push_back(msg);

  
  
  // //  L second
  // // msg.header.frame_id = "map";
  // // msg.pose.position.x = 3749.709716796875;
  // // msg.pose.position.y = 73735.859375;
  // // msg.pose.position.z = 0.0;
  // // msg.pose.orientation.x = 0.0;
  // // msg.pose.orientation.y = 0.0;
  // // msg.pose.orientation.z = 0.5708294720816296;
  // // msg.pose.orientation.w = 0.8210686413467562;
  // // pre_seted_pose_vector.push_back(msg);
  
  //  //  L third
  // // msg.header.frame_id = "map";
  // // msg.pose.position.x = 3750.218505859375;
  // // msg.pose.position.y = 73736.234375;
  // // msg.pose.position.z = 0.0;
  // // msg.pose.orientation.x = 0.0;
  // // msg.pose.orientation.y = 0.0;
  // // msg.pose.orientation.z = 0.43712374276198984;
  // // msg.pose.orientation.w = 0.8994013750899815;
  // // pre_seted_pose_vector.push_back(msg);

  //  //  L third replace   2023-11-08
  msg.header.frame_id = "map";
  msg.pose.position.x = 3750.20263671875;
  msg.pose.position.y = 73736.03125;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = 0.5799025524453801;
  msg.pose.orientation.w = 0.8146858472241513;
  pre_seted_pose_vector.push_back(msg);

  
  //  //  L third replace  ---
  // msg.header.frame_id = "map";
  // msg.pose.position.x = 3750.34814453125;
  // msg.pose.position.y = 73736.6875;
  // msg.pose.position.z = 0.0;
  // msg.pose.orientation.x = 0.0;
  // msg.pose.orientation.y = 0.0;
  // msg.pose.orientation.z = 0.6017139825542301;
  // msg.pose.orientation.w = 0.7987116395788455;
  // pre_seted_pose_vector.push_back(msg);
  


  // msg.header.frame_id = "map";
  // msg.pose.position.x = 3751.2109375;
  // msg.pose.position.y = 73737.1015625;
  // msg.pose.position.z = 0.0;
  // msg.pose.orientation.x = 0.0;
  // msg.pose.orientation.y = 0.0;
  // msg.pose.orientation.z = 0.3214301106708757;
  // msg.pose.orientation.w = 0.9469333049133443;
  // pre_seted_pose_vector.push_back(msg);

//--2
  // msg.header.frame_id = "map";
  // msg.pose.position.x = 3749.689697265625;
  // msg.pose.position.y = 73735.3203125;
  // msg.pose.position.z = 0.0;
  // msg.pose.orientation.x = 0.0;
  // msg.pose.orientation.y = 0.0;
  // msg.pose.orientation.z = 0.6678927619723282;
  // msg.pose.orientation.w = 0.7442575216314411;
  // pre_seted_pose_vector.push_back(msg);
//--3
  // msg.header.frame_id = "map";
  // msg.pose.position.x = 3755.7021484375;
  // msg.pose.position.y = 73739.1640625;
  // msg.pose.position.z = 0.0;
  // msg.pose.orientation.x = 0.0;
  // msg.pose.orientation.y = 0.0;
  // msg.pose.orientation.z = 0.24927759678599598;
  // msg.pose.orientation.w = 0.9684320728582869;
  // pre_seted_pose_vector.push_back(msg);

// --3 box replace
  msg.header.frame_id = "map";
  msg.pose.position.x = 3754.760498046875;
  msg.pose.position.y = 73738.625;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = 0.24487996608642515;
  msg.pose.orientation.w = 0.9695534034850846;
  pre_seted_pose_vector.push_back(msg);

  //--3 + 
  msg.header.frame_id = "map";
  msg.pose.position.x = 3750.83154296875;
  msg.pose.position.y = 73737.640625;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = 0.024630872112737776;
  msg.pose.orientation.w = 0.9996966140479651;
  pre_seted_pose_vector.push_back(msg);

 // -3 replace
  // msg.header.frame_id = "map";
  // msg.pose.position.x = 3755.517333984375;
  // msg.pose.position.y = 73738.7421875;
  // msg.pose.position.z = 0.0;
  // msg.pose.orientation.x = 0.0;
  // msg.pose.orientation.y = 0.0;
  // msg.pose.orientation.z = 0.26645980910882694;
  // msg.pose.orientation.w = 0.9638460303023961;
  // pre_seted_pose_vector.push_back(msg);



  //--4
  msg.header.frame_id = "map";
  msg.pose.position.x = 3745.0830078125;
  msg.pose.position.y = 73743.4453125;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = -0.5178196572501046;
  msg.pose.orientation.w = 0.8554898027243716;
  pre_seted_pose_vector.push_back(msg);
//--5
//honggang

//drive 1
  msg.header.frame_id = "map";
  msg.pose.position.x = 3739.350830078125;
  msg.pose.position.y = 73754.3515625;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = -0.49164540017165514;
  msg.pose.orientation.w = 0.8707954986620298;
  pre_seted_pose_vector.push_back(msg);
//--6
//drive 1 middle
  msg.header.frame_id = "map";
  msg.pose.position.x = 3727.3359375;
  msg.pose.position.y = 73757.21875;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = 0.2408351489047307;
  msg.pose.orientation.w = 0.9705660364200038;
  pre_seted_pose_vector.push_back(msg);

  //--7
  msg.header.frame_id = "map";
  msg.pose.position.x = 3718.353271484375;
  msg.pose.position.y = 73750.734375;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = 0.6679334714495515;
  msg.pose.orientation.w = 0.7442209871519018;
  pre_seted_pose_vector.push_back(msg);
//--8
//drive 2
  msg.header.frame_id = "map";
  msg.pose.position.x = 3720.598388671875;
  msg.pose.position.y = 73744.0859375;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = 0.8632819539170411;
  msg.pose.orientation.w = 0.5047219710307603;
  pre_seted_pose_vector.push_back(msg);

// --9 
  msg.header.frame_id = "map";
  msg.pose.position.x = 3715.57568359375;
  msg.pose.position.y = 73734.90625;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = 0.26171122506788413;
  msg.pose.orientation.w = 0.9651462245035554;
  pre_seted_pose_vector.push_back(msg);

  //--9 
  // msg.header.frame_id = "map";
  // msg.pose.position.x = 3715.57568359375;
  // msg.pose.position.y = 73734.90625;
  // msg.pose.position.z = 0.0;
  // msg.pose.orientation.x = 0.0;
  // msg.pose.orientation.y = 0.0;
  // msg.pose.orientation.z = 0.23593056980882635;
  // msg.pose.orientation.w = 0.9717699142439441;
  // pre_seted_pose_vector.push_back(msg);
  
  
  //--10
  msg.header.frame_id = "map";
  msg.pose.position.x = 3706.668701171875;
  msg.pose.position.y = 73730.421875;
  msg.pose.position.z = 0.0;
  msg.pose.orientation.x = 0.0;
  msg.pose.orientation.y = 0.0;
  msg.pose.orientation.z = 0.2309412920875842;
  msg.pose.orientation.w = 0.9729676868267091;
  pre_seted_pose_vector.push_back(msg);


  //--10
  // msg.header.frame_id = "map";
  // msg.pose.position.x = 3704.99658203125;
  // msg.pose.position.y = 73737.609375;
  // msg.pose.position.z = 0.0;
  // msg.pose.orientation.x = 0.0;
  // msg.pose.orientation.y = 0.0;
  // msg.pose.orientation.z = -0.5021006243841415;
  // msg.pose.orientation.w = 0.8648092061218215;
  // pre_seted_pose_vector.push_back(msg);

//--9 
  // msg.header.frame_id = "map";
  // msg.pose.position.x = 3708.572509765625;
  // msg.pose.position.y = 73731.375;
  // msg.pose.position.z = 0.0;
  // msg.pose.orientation.x = 0.0;
  // msg.pose.orientation.y = 0.0;
  // msg.pose.orientation.z = 0.24365928570205184;
  // msg.pose.orientation.w = 0.9698608933713978;
  // pre_seted_pose_vector.push_back(msg);
//--9

// //drive3
//   msg.header.frame_id = "map";
//   msg.pose.position.x = 3717.45068359375;
//   msg.pose.position.y = 73735.7109375;
//   msg.pose.position.z = 0.0;
//   msg.pose.orientation.x = 0.0;
//   msg.pose.orientation.y = 0.0;
//   msg.pose.orientation.z = 0.23903341739572623;
//   msg.pose.orientation.w = 0.9710113415239394;
//   pre_seted_pose_vector.push_back(msg);

//drive4
msg.header.frame_id = "map";
msg.pose.position.x = 3701.373779296875;
msg.pose.position.y = 73744.7890625;
msg.pose.position.z = 0.0;
msg.pose.orientation.x = 0.0;
msg.pose.orientation.y = 0.0;
msg.pose.orientation.z = -0.5372075731791744;
msg.pose.orientation.w = 0.8434500716218726;
pre_seted_pose_vector.push_back(msg);

  for(auto pose : pre_seted_pose_vector)
  {
    RCLCPP_INFO(this->get_logger(), "pre_seted_pose_vector x is %f",pose.pose.position.x);
  }



//special pose
msg.header.frame_id = "map";
msg.pose.position.x = 3754.385009765625;
msg.pose.position.y = 73739.8046875;
msg.pose.position.z = 0.0;
msg.pose.orientation.x = 0.0;
msg.pose.orientation.y = 0.0;
msg.pose.orientation.z = 0.1939932506522466;
msg.pose.orientation.w = 0.9810028637579885;
pre_seted_special_pose_vector.push_back(msg);

}

// std::string exec(const char* cmd) {
//     std::array<char, 128> buffer;
//     std::string result;
//     std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
//     if (!pipe) throw std::runtime_error("popen() failed!");
//     while (!feof(pipe.get())) {
//         if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
//             result += buffer.data();
//     }
//     return result;
// }


void ScenarioSelectorNode::change_freespace_param(const int & num)
{
    // std::string result = exec("ros2 param set /planning/scenario_planning/parking/freespace_planner jump_if_size_less 1");
    // RCLCPP_INFO(this->get_logger(), "Parameter set result is %s",result.c_str());
  // // 修改节点的参数值
    std::vector<rclcpp::Parameter> params;
    params.push_back(rclcpp::Parameter("jump_if_size_less", num));

    auto result = this->parameter_client_freespace->set_parameters_atomically(params);
    if(result.successful)
    {
        RCLCPP_INFO(this->get_logger(), "Parameter set successfully");
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to set parameter");
    }
  }

void ScenarioSelectorNode::change_freespace_param_width(const double & num)
{
    // std::string result = exec("ros2 param set /planning/scenario_planning/parking/freespace_planner jump_if_size_less 1");
    // RCLCPP_INFO(this->get_logger(), "Parameter set result is %s",result.c_str());
  // // 修改节点的参数值
    std::vector<rclcpp::Parameter> params;
    params.push_back(rclcpp::Parameter("vehicle_shape_margin_width_m", num));

    auto result = this->parameter_client_freespace->set_parameters_atomically(params);
    if(result.successful)
    {
        RCLCPP_INFO(this->get_logger(), "vehicle_shape_margin_width_m Parameter set successfully");
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "vehicle_shape_margin_width_m Failed to set parameter");
    }
  }




  void ScenarioSelectorNode::change_pid_param()
{
  // // 修改节点的参数值
  // 修改节点的参数值
    std::vector<rclcpp::Parameter> params;
    params.push_back(rclcpp::Parameter("stopping_state_stop_dist", 0.1));

    auto result = this->parameter_client_pid->set_parameters_atomically(params);
    if(result.successful)
    {
        RCLCPP_INFO(this->get_logger(), "Parameter set successfully");
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to set parameter");
    }
  }

  
ScenarioSelectorNode::ScenarioSelectorNode(const rclcpp::NodeOptions & node_options)
: Node("scenario_selector", node_options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_),
  current_scenario_(tier4_planning_msgs::msg::Scenario::EMPTY),
  update_rate_(this->declare_parameter<double>("update_rate", 10.0)),
  th_max_message_delay_sec_(this->declare_parameter<double>("th_max_message_delay_sec", 1.0)),
  th_arrived_distance_m_(this->declare_parameter<double>("th_arrived_distance_m", 1.0)),
  th_stopped_time_sec_(this->declare_parameter<double>("th_stopped_time_sec", 1.0)),
  th_stopped_velocity_mps_(this->declare_parameter<double>("th_stopped_velocity_mps", 0.01)),
  is_parking_completed_(false),
  drive_stage(-1),
  is_published_goal(false),
  is_special_position(false),
  is_engaged_goal(false)
{
  

    
  // Input
  sub_lane_driving_trajectory_ =
    this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
      "input/lane_driving/trajectory", rclcpp::QoS{1},
      std::bind(&ScenarioSelectorNode::onLaneDrivingTrajectory, this, std::placeholders::_1));

  sub_parking_trajectory_ = this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
    "input/parking/trajectory", rclcpp::QoS{1},
    std::bind(&ScenarioSelectorNode::onParkingTrajectory, this, std::placeholders::_1));

  sub_lanelet_map_ = this->create_subscription<autoware_auto_mapping_msgs::msg::HADMapBin>(
    "input/lanelet_map", rclcpp::QoS{1}.transient_local(),
    std::bind(&ScenarioSelectorNode::onMap, this, std::placeholders::_1));
  sub_route_ = this->create_subscription<autoware_planning_msgs::msg::LaneletRoute>(
    "input/route", rclcpp::QoS{1}.transient_local(),
    std::bind(&ScenarioSelectorNode::onRoute, this, std::placeholders::_1));
  sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "input/odometry", rclcpp::QoS{100},
    std::bind(&ScenarioSelectorNode::onOdom, this, std::placeholders::_1));
  sub_parking_state_ = this->create_subscription<std_msgs::msg::Bool>(
    "is_parking_completed", rclcpp::QoS{100},
    std::bind(&ScenarioSelectorNode::onParkingState, this, std::placeholders::_1));

    sub_special_position = this->create_subscription<std_msgs::msg::Bool>(
    "is_special_position", rclcpp::QoS{100},
    std::bind(&ScenarioSelectorNode::onSpecialPosition, this, std::placeholders::_1));

  // Output
  pub_scenario_ =
    this->create_publisher<tier4_planning_msgs::msg::Scenario>("output/scenario", rclcpp::QoS{1});
  pub_trajectory_ = this->create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
    "output/trajectory", rclcpp::QoS{1});


  //ai_challange_control
  // Publishers
  engage_publisher =
    this->create_publisher<Engage>("output/engage", 1);
  goal_pos_publisher =
    this->create_publisher<PoseStamped>("output/goal", 1);

  // Subscribers
  state_subscriber = this->create_subscription<AutowareState>(
    "input/state", 1, std::bind(&ScenarioSelectorNode::goalCallback, this, std::placeholders::_1));

  // Timer Callback
  const auto period_ns = rclcpp::Rate(static_cast<double>(update_rate_)).period();

  timer_ = rclcpp::create_timer(
    this, get_clock(), period_ns, std::bind(&ScenarioSelectorNode::onTimer, this));
  // 创建一个参数客户端来修改参数
  auto freespace_manager = rclcpp::Node::make_shared("freespace_manager");
  this->parameter_client_freespace = std::make_shared<rclcpp::SyncParametersClient>(freespace_manager,"/planning/scenario_planning/parking/freespace_planner");
  auto pid_manager = rclcpp::Node::make_shared("pid_manager");
  // 创建一个参数客户端来修改参数
  this->parameter_client_pid = std::make_shared<rclcpp::SyncParametersClient>(pid_manager,"/control/trajectory_follower/controller_node_exe");

  this->initPreSetedPose();
  // Wait for first tf
  while (rclcpp::ok()) {
    try {
      tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
      break;
    } catch (tf2::TransformException & ex) {
      RCLCPP_DEBUG(this->get_logger(), "waiting for initial pose...");
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ScenarioSelectorNode)









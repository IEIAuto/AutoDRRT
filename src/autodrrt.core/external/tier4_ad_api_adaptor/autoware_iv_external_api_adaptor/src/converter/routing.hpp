// Copyright 2021 TIER IV, Inc.
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

#ifndef CONVERTER__ROUTING_HPP_
#define CONVERTER__ROUTING_HPP_

#include <autoware_adapi_v1_msgs/msg/route.hpp>
#include <autoware_adapi_v1_msgs/srv/set_route.hpp>
#include <tier4_external_api_msgs/msg/route.hpp>
#include <tier4_external_api_msgs/srv/set_route.hpp>

namespace external_api::converter
{

using AdSetRoute = autoware_adapi_v1_msgs::srv::SetRoute::Request;
using T4SetRoute = tier4_external_api_msgs::srv::SetRoute::Request;
using AdRoute = autoware_adapi_v1_msgs::msg::Route;
using T4Route = tier4_external_api_msgs::msg::Route;
using AdSegment = autoware_adapi_v1_msgs::msg::RouteSegment;
using T4Segment = tier4_external_api_msgs::msg::RouteSection;
using AdPrimitive = autoware_adapi_v1_msgs::msg::RoutePrimitive;

inline AdSegment convert(const T4Segment & t4)
{
  AdSegment ad;
  for (const auto & id : t4.lane_ids) {
    AdPrimitive primitive;
    primitive.id = id;
    primitive.type = "lane";
    if (id == t4.preferred_lane_id) {
      ad.preferred = primitive;
    } else {
      ad.alternatives.push_back(primitive);
    }
  }
  return ad;
}

inline T4Segment convert(const AdSegment & ad)
{
  T4Segment t4;
  t4.preferred_lane_id = ad.preferred.id;
  t4.lane_ids.push_back(ad.preferred.id);
  for (const auto & primitive : ad.alternatives) {
    t4.lane_ids.push_back(primitive.id);
  }
  return t4;
}

inline AdSetRoute convert(const T4SetRoute & t4)
{
  AdSetRoute ad;
  ad.header = t4.route.goal_pose.header;
  ad.goal = t4.route.goal_pose.pose;
  for (const auto & section : t4.route.route_sections) {
    ad.segments.push_back(convert(section));
  }
  return ad;
}

inline T4Route convert(const AdRoute & ad)
{
  T4Route t4;
  t4.goal_pose.header = ad.header;
  t4.goal_pose.pose = ad.data.front().goal;
  for (const auto & segment : ad.data.front().segments) {
    t4.route_sections.push_back(convert(segment));
  }
  return t4;
}

}  // namespace external_api::converter

#endif  // CONVERTER__ROUTING_HPP_

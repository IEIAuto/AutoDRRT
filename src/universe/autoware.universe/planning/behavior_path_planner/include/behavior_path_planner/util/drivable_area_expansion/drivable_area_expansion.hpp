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

#ifndef BEHAVIOR_PATH_PLANNER__UTIL__DRIVABLE_AREA_EXPANSION__DRIVABLE_AREA_EXPANSION_HPP_
#define BEHAVIOR_PATH_PLANNER__UTIL__DRIVABLE_AREA_EXPANSION__DRIVABLE_AREA_EXPANSION_HPP_

#include "behavior_path_planner/util/drivable_area_expansion/parameters.hpp"
#include "behavior_path_planner/util/drivable_area_expansion/types.hpp"

#include <route_handler/route_handler.hpp>

#include <lanelet2_core/Forward.h>

namespace drivable_area_expansion
{
/// @brief Expand the drivable area based on the projected ego footprint along the path
/// @param[in] path path whose drivable area will be expanded
/// @param[in] params expansion parameters
/// @param[in] dynamic_objects dynamic objects
/// @param[in] route_handler route handler
/// @param[in] path_lanes lanelets of the path
void expandDrivableArea(
  PathWithLaneId & path, const DrivableAreaExpansionParameters & params,
  const PredictedObjects & dynamic_objects, const route_handler::RouteHandler & route_handler,
  const lanelet::ConstLanelets & path_lanes);

/// @brief Create a polygon combining the drivable area of a path and some expansion polygons
/// @param[in] path path and its drivable area
/// @param[in] expansion_polygons polygons to add to the drivable area
/// @return expanded drivable area polygon
polygon_t createExpandedDrivableAreaPolygon(
  const PathWithLaneId & path, const multipolygon_t & expansion_polygons);

/// @brief Update the drivable area of the given path with the given polygon
/// @details this function splits the polygon into a left and right bound and sets it in the path
/// @param[in] path path whose drivable area will be expanded
/// @param[in] expanded_drivable_area polygon of the new drivable area
void updateDrivableAreaBounds(PathWithLaneId & path, const polygon_t & expanded_drivable_area);
}  // namespace drivable_area_expansion

#endif  // BEHAVIOR_PATH_PLANNER__UTIL__DRIVABLE_AREA_EXPANSION__DRIVABLE_AREA_EXPANSION_HPP_

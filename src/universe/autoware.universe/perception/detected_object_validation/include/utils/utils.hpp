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

#ifndef UTILS__UTILS_HPP_
#define UTILS__UTILS_HPP_

#include <cstdint>

namespace utils
{
struct FilterTargetLabel
{
  bool UNKNOWN;
  bool CAR;
  bool TRUCK;
  bool BUS;
  bool TRAILER;
  bool MOTORCYCLE;
  bool BICYCLE;
  bool PEDESTRIAN;
  bool isTarget(const uint8_t label) const;
};  // struct FilterTargetLabel
}  // namespace utils

#endif  // UTILS__UTILS_HPP_

// Copyright 2018-2021 The Autoware Foundation
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

#ifndef MPC_LATERAL_CONTROLLER__INTERPOLATE_HPP_
#define MPC_LATERAL_CONTROLLER__INTERPOLATE_HPP_

#include <cmath>
#include <iostream>
#include <vector>

namespace autoware::motion::control::mpc_lateral_controller
{

/**
 * @brief linearly interpolate the given values assuming a base indexing and a new desired indexing
 * @param [in] base_index indexes for each base value
 * @param [in] base_value values for each base index
 * @param [in] return_index desired interpolated indexes
 * @param [out] return_value resulting interpolated values
 */
bool linearInterpolate(
  const std::vector<double> & base_index, const std::vector<double> & base_value,
  const std::vector<double> & return_index, std::vector<double> & return_value);
/**
 * @brief linearly interpolate the given values assuming a base indexing and a new desired index
 * @param [in] base_index indexes for each base value
 * @param [in] base_value values for each base index
 * @param [in] return_index desired interpolated index
 * @param [out] return_value resulting interpolated value
 */
bool linearInterpolate(
  const std::vector<double> & base_index, const std::vector<double> & base_value,
  const double & return_index, double & return_value);
}  // namespace autoware::motion::control::mpc_lateral_controller
#endif  // MPC_LATERAL_CONTROLLER__INTERPOLATE_HPP_

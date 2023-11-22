// Copyright 2023 Autoware Foundation
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

#ifndef EKF_LOCALIZER__WARNING_MESSAGE_HPP_
#define EKF_LOCALIZER__WARNING_MESSAGE_HPP_

#include <string>

std::string poseDelayStepWarningMessage(
  const double delay_time, const int extend_state_step, const double ekf_dt);
std::string twistDelayStepWarningMessage(
  const double delay_time, const int extend_state_step, const double ekf_dt);
std::string poseDelayTimeWarningMessage(const double delay_time);
std::string twistDelayTimeWarningMessage(const double delay_time);
std::string mahalanobisWarningMessage(const double distance, const double max_distance);

#endif  // EKF_LOCALIZER__WARNING_MESSAGE_HPP_

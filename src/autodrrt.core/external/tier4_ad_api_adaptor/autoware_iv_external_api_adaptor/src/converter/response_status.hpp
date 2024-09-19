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

#ifndef CONVERTER__RESPONSE_STATUS_HPP_
#define CONVERTER__RESPONSE_STATUS_HPP_

#include <tier4_api_utils/tier4_api_utils.hpp>

#include <autoware_adapi_v1_msgs/msg/response_status.hpp>
#include <tier4_external_api_msgs/msg/response_status.hpp>

namespace external_api::converter
{

using AdResponseStatus = autoware_adapi_v1_msgs::msg::ResponseStatus;
using T4ResponseStatus = tier4_external_api_msgs::msg::ResponseStatus;

inline T4ResponseStatus convert(const AdResponseStatus & ad)
{
  if (ad.success) {
    return tier4_api_utils::response_success(ad.message);
  } else {
    return tier4_api_utils::response_error(ad.message);
  }
}

}  // namespace external_api::converter

#endif  // CONVERTER__RESPONSE_STATUS_HPP_

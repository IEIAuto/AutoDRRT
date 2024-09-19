// Copyright 2021 Apex.AI, Inc.
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

/// \copyright Copyright 2021 Apex.AI, Inc.
/// All rights reserved.

#ifndef POINT_CLOUD_MSG_WRAPPER__DEFAULT_FIELD_GENERATORS_HPP_
#define POINT_CLOUD_MSG_WRAPPER__DEFAULT_FIELD_GENERATORS_HPP_

#include <point_cloud_msg_wrapper/field_generators.hpp>

namespace point_cloud_msg_wrapper
{

LIDAR_UTILS__DEFINE_FIELD_GENERATOR_FOR_MEMBER(x);
LIDAR_UTILS__DEFINE_FIELD_GENERATOR_FOR_MEMBER(y);
LIDAR_UTILS__DEFINE_FIELD_GENERATOR_FOR_MEMBER(z);
LIDAR_UTILS__DEFINE_FIELD_GENERATOR_FOR_MEMBER(id);
LIDAR_UTILS__DEFINE_FIELD_GENERATOR_FOR_MEMBER(ring);
LIDAR_UTILS__DEFINE_FIELD_GENERATOR_FOR_MEMBER(intensity);
LIDAR_UTILS__DEFINE_FIELD_GENERATOR_FOR_MEMBER(timestamp);

}  // namespace point_cloud_msg_wrapper

#endif  // POINT_CLOUD_MSG_WRAPPER__DEFAULT_FIELD_GENERATORS_HPP_

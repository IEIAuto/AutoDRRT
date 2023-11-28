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

#ifndef NDT_SCAN_MATCHER__NDT_SCAN_MATCHER_OPT_HPP_
#define NDT_SCAN_MATCHER__NDT_SCAN_MATCHER_OPT_HPP_

#include <pcl/point_types.h>
#include <pclomp/ndt_omp.h>

namespace ndt_scan_matcher_opt
{
using PointSource = pcl::PointXYZ;
using PointTarget = pcl::PointXYZ;
using NormalDistributionsTransform =
pclomp::NormalDistributionsTransform<PointSource, PointTarget>;
void acc_ndt(
    const std::shared_ptr<NormalDistributionsTransform> & ndt_ptr_, pclomp::NdtParams & ndt_params);

}  // namespace pointcloud_preprocessor
#endif  // NDT_SCAN_MATCHER__NDT_SCAN_MATCHER_OPT_HPP_
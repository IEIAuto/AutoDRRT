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

#ifndef METADATA_PACKAGES_HPP_
#define METADATA_PACKAGES_HPP_

#include "rclcpp/rclcpp.hpp"
#include "tier4_api_utils/tier4_api_utils.hpp"

#include "tier4_external_api_msgs/srv/get_metadata_packages.hpp"

namespace external_api
{

class MetadataPackages : public rclcpp::Node
{
public:
  explicit MetadataPackages(const rclcpp::NodeOptions & options);

private:
  // ros interface
  rclcpp::CallbackGroup::SharedPtr group_;
  tier4_api_utils::Service<tier4_external_api_msgs::srv::GetMetadataPackages>::SharedPtr srv_;

  // ros callback
  void getVersions(
    const tier4_external_api_msgs::srv::GetMetadataPackages::Request::SharedPtr request,
    const tier4_external_api_msgs::srv::GetMetadataPackages::Response::SharedPtr response);

  // data
  tier4_external_api_msgs::msg::MetadataPackages metadata_;
};

}  // namespace external_api

#endif  // METADATA_PACKAGES_HPP_

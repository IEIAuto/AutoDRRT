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

#include "localization_score.hpp"

namespace external_api
{

LocalizationScore::LocalizationScore(const rclcpp::NodeOptions & options)
: Node("localization_score", options), clock_(this->get_clock())
{
  using std::placeholders::_1;

  status_pub_hz_ = this->declare_parameter("status_pub_hz", 10.0);

  score_tp_.name = "transform_probability";
  score_nvtl_.name = "nearest_voxel_transformation_likelihood";
  is_tp_received_ = false;
  is_nvtl_received_ = false;

  // Publisher
  pub_localization_scores_ =
    this->create_publisher<LocalizationScoreArray>("/api/external/get/localization_scores", 1);

  // Subscriber
  sub_transform_probability_ = this->create_subscription<Float32Stamped>(
    "/localization/pose_estimator/transform_probability", 1,
    std::bind(&LocalizationScore::callbackTpScore, this, _1));
  sub_nearest_voxel_transformation_likelihood_ = this->create_subscription<Float32Stamped>(
    "/localization/pose_estimator/nearest_voxel_transformation_likelihood", 1,
    std::bind(&LocalizationScore::callbackNvtlScore, this, _1));

  // Timer callback
  auto timer_callback = std::bind(&LocalizationScore::callbackTimer, this);
  auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(1.0 / status_pub_hz_));
  timer_ = rclcpp::create_timer(this, get_clock(), period, std::move(timer_callback));
}

void LocalizationScore::callbackTpScore(const Float32Stamped::ConstSharedPtr msg_ptr)
{
  is_tp_received_ = true;
  score_tp_.value = msg_ptr->data;
}
void LocalizationScore::callbackNvtlScore(const Float32Stamped::ConstSharedPtr msg_ptr)
{
  is_nvtl_received_ = true;
  score_nvtl_.value = msg_ptr->data;
}

void LocalizationScore::callbackTimer()
{
  LocalizationScoreArray localization_scores_msg;

  if (is_tp_received_) {
    localization_scores_msg.values.emplace_back(score_tp_);
    is_tp_received_ = false;
  }

  if (is_nvtl_received_) {
    localization_scores_msg.values.emplace_back(score_nvtl_);
    is_nvtl_received_ = false;
  }

  if (!localization_scores_msg.values.empty()) {
    pub_localization_scores_->publish(localization_scores_msg);
  }
}

}  // namespace external_api

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::LocalizationScore)

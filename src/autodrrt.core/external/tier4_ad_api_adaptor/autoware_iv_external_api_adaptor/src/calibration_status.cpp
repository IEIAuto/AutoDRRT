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

#include "calibration_status.hpp"

#include <memory>

namespace external_api
{

CalibrationStatus::CalibrationStatus(const rclcpp::NodeOptions & options)
: Node("calibration_status", options)
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  tier4_api_utils::ServiceProxyNodeInterface proxy(this);

  group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  srv_get_accel_brake_map_calibration_data_ =
    proxy.create_service<tier4_external_api_msgs::srv::GetAccelBrakeMapCalibrationData>(
      "/api/external/get/accel_brake_map_calibrator/data",
      std::bind(&CalibrationStatus::getAccelBrakeMapCalibrationData, this, _1, _2),
      rmw_qos_profile_services_default, group_);
  cli_get_accel_brake_map_calibration_data_ =
    proxy.create_client<tier4_external_api_msgs::srv::GetAccelBrakeMapCalibrationData>(
      "/accel_brake_map_calibrator/get_data_service", rmw_qos_profile_services_default);

  using namespace std::literals::chrono_literals;

  pub_calibration_status_ = create_publisher<tier4_external_api_msgs::msg::CalibrationStatusArray>(
    "/api/external/get/calibration_status", rclcpp::QoS(1));
  timer_ =
    rclcpp::create_timer(this, get_clock(), 200ms, std::bind(&CalibrationStatus::onTimer, this));

  sub_accel_brake_map_calibration_status_ =
    create_subscription<tier4_external_api_msgs::msg::CalibrationStatus>(
      "/accel_brake_map_calibrator/output/calibration_status", rclcpp::QoS(1),
      [this](const tier4_external_api_msgs::msg::CalibrationStatus::ConstSharedPtr msg) {
        accel_brake_map_status_ = msg;
      });
}

void CalibrationStatus::onTimer()
{
  tier4_external_api_msgs::msg::CalibrationStatusArray calibration_status;

  calibration_status.stamp = now();
  if (accel_brake_map_status_ != nullptr) {
    calibration_status.status_array.emplace_back(*accel_brake_map_status_);
    accel_brake_map_status_ = nullptr;
  }

  if (!calibration_status.status_array.empty()) {
    pub_calibration_status_->publish(calibration_status);
  }
}

void CalibrationStatus::getAccelBrakeMapCalibrationData(
  const tier4_external_api_msgs::srv::GetAccelBrakeMapCalibrationData::Request::SharedPtr request,
  const tier4_external_api_msgs::srv::GetAccelBrakeMapCalibrationData::Response::SharedPtr response)
{
  const auto [status, resp] =
    cli_get_accel_brake_map_calibration_data_->call(request, std::chrono::seconds(190));
  if (!tier4_api_utils::is_success(status)) {
    return;
  }
  response->graph_image = resp->graph_image;
  response->accel_map = resp->accel_map;
  response->brake_map = resp->brake_map;
}

}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::CalibrationStatus)

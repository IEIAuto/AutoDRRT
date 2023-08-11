// Copyright 2022 Tier IV, Inc.
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

#include "pacmod_steer_test/pacmod_steer_test.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

PacmodSteerTest::PacmodSteerTest()
: Node("pacmod_steer_test"),
  vehicle_info_(vehicle_info_util::VehicleInfoUtil(*this).getVehicleInfo())
{
  /* setup parameters */
  /* setup parameters */
  base_frame_id_ = declare_parameter("base_frame_id", "base_link");
  command_timeout_ms_ = declare_parameter("command_timeout_ms", 1000);
  loop_rate_ = declare_parameter("loop_rate", 30.0);

  /* parameters for vehicle specifications */
  tire_radius_ = vehicle_info_.wheel_radius_m;
  wheel_base_ = vehicle_info_.wheel_base_m;

  /* vehicle parameters */
  vgr_coef_a_ = declare_parameter("vgr_coef_a", 15.713);
  vgr_coef_b_ = declare_parameter("vgr_coef_b", 0.053);
  vgr_coef_c_ = declare_parameter("vgr_coef_c", 0.042);

  /* parameters for pacmod test */

  amplitude_ = declare_parameter("test_amplitude", 0.0);
  delta_amplitude_ = declare_parameter("test_delta_amplitude", 0.0);
  hz_ = declare_parameter("test_frequency", 0.0);
  offset_ = declare_parameter("test_steer_offset", 0.0);
  steer_rate_ = declare_parameter("test_steer_rate", 0.0);
  stopping_ = declare_parameter("test_stopping", true);
  accel_value_ = declare_parameter("accel_value", 0.0);
  brake_value_ = declare_parameter("brake_value", 0.0);
  int mode = declare_parameter("test_mode", 0);
  test_mode_ = static_cast<TestMode>(mode);

  /* subscribers */
  using std::placeholders::_1;

  // Engage
  engage_cmd_sub_ = create_subscription<autoware_auto_vehicle_msgs::msg::Engage>(
    "/vehicle/engage", rclcpp::QoS{1}, std::bind(&PacmodSteerTest::callbackEngage, this, _1));
  // From pacmod
  steer_wheel_rpt_sub_ =
    std::make_unique<message_filters::Subscriber<pacmod3_msgs::msg::SystemRptFloat>>(
      this, "/pacmod/steering_rpt");
  wheel_speed_rpt_sub_ =
    std::make_unique<message_filters::Subscriber<pacmod3_msgs::msg::WheelSpeedRpt>>(
      this, "/pacmod/wheel_speed_rpt");
  accel_rpt_sub_ = std::make_unique<message_filters::Subscriber<pacmod3_msgs::msg::SystemRptFloat>>(
    this, "/pacmod/accel_rpt");
  brake_rpt_sub_ = std::make_unique<message_filters::Subscriber<pacmod3_msgs::msg::SystemRptFloat>>(
    this, "/pacmod/brake_rpt");
  shift_rpt_sub_ = std::make_unique<message_filters::Subscriber<pacmod3_msgs::msg::SystemRptInt>>(
    this, "/pacmod/shift_rpt");
  turn_rpt_sub_ = std::make_unique<message_filters::Subscriber<pacmod3_msgs::msg::SystemRptInt>>(
    this, "/pacmod/turn_rpt");
  global_rpt_sub_ = std::make_unique<message_filters::Subscriber<pacmod3_msgs::msg::GlobalRpt>>(
    this, "/pacmod/global_rpt");

  pacmod_feedbacks_sync_ =
    std::make_unique<message_filters::Synchronizer<PacmodFeedbacksSyncPolicy>>(
      PacmodFeedbacksSyncPolicy(10), *steer_wheel_rpt_sub_, *wheel_speed_rpt_sub_, *accel_rpt_sub_,
      *brake_rpt_sub_, *shift_rpt_sub_, *turn_rpt_sub_, *global_rpt_sub_);

  pacmod_feedbacks_sync_->registerCallback(std::bind(
    &PacmodSteerTest::callbackPacmodRpt, this, std::placeholders::_1, std::placeholders::_2,
    std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
    std::placeholders::_7));

  /* publisher */
  // To pacmod
  accel_cmd_pub_ =
    create_publisher<pacmod3_msgs::msg::SystemCmdFloat>("/pacmod/accel_cmd", rclcpp::QoS{1});
  brake_cmd_pub_ =
    create_publisher<pacmod3_msgs::msg::SystemCmdFloat>("/pacmod/brake_cmd", rclcpp::QoS{1});
  steer_cmd_pub_ =
    create_publisher<pacmod3_msgs::msg::SteeringCmd>("/pacmod/steering_cmd", rclcpp::QoS{1});
  shift_cmd_pub_ =
    create_publisher<pacmod3_msgs::msg::SystemCmdInt>("/pacmod/shift_cmd", rclcpp::QoS{1});
  turn_cmd_pub_ =
    create_publisher<pacmod3_msgs::msg::SystemCmdInt>("/pacmod/turn_cmd", rclcpp::QoS{1});

  // To Autoware
  vehicle_twist_pub_ = create_publisher<autoware_auto_vehicle_msgs::msg::VelocityReport>(
    "/vehicle/status/velocity_status", rclcpp::QoS{1});
  steering_status_pub_ = create_publisher<autoware_auto_vehicle_msgs::msg::SteeringReport>(
    "/vehicle/status/steering_status", rclcpp::QoS{1});
  // Timer
  auto timer_callback = std::bind(&PacmodSteerTest::publishCommands, this);
  auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(1.0 / loop_rate_));
  timer_ = std::make_shared<rclcpp::GenericTimer<decltype(timer_callback)>>(
    this->get_clock(), period, std::move(timer_callback),
    this->get_node_base_interface()->get_context());
  this->get_node_timers_interface()->add_timer(timer_, nullptr);
}

void PacmodSteerTest::callbackEngage(
  const autoware_auto_vehicle_msgs::msg::Engage::ConstSharedPtr msg)
{
  engage_cmd_ = msg->engage;
  is_clear_override_needed_ = true;
}

void PacmodSteerTest::callbackPacmodRpt(
  const pacmod3_msgs::msg::SystemRptFloat::ConstSharedPtr steer_wheel_rpt,
  const pacmod3_msgs::msg::WheelSpeedRpt::ConstSharedPtr wheel_speed_rpt,
  const pacmod3_msgs::msg::SystemRptFloat::ConstSharedPtr accel_rpt,
  const pacmod3_msgs::msg::SystemRptFloat::ConstSharedPtr brake_rpt,
  const pacmod3_msgs::msg::SystemRptInt::ConstSharedPtr shift_rpt,
  const pacmod3_msgs::msg::SystemRptInt::ConstSharedPtr turn_rpt,
  const pacmod3_msgs::msg::GlobalRpt::ConstSharedPtr global_rpt)
{
  RCLCPP_ERROR_STREAM_THROTTLE(
    get_logger(), *get_clock(), 5000.0, "Pacmod Steer Test Now.");  // for safety

  is_pacmod_rpt_received_ = true;

  steer_wheel_rpt_ptr_ = steer_wheel_rpt;
  wheel_speed_rpt_ptr_ = wheel_speed_rpt;
  accel_rpt_ptr_ = accel_rpt;
  brake_rpt_ptr_ = brake_rpt;
  shift_rpt_ptr_ = shift_rpt;
  turn_rpt_ptr_ = turn_rpt;
  global_rpt_ptr_ = global_rpt;

  is_pacmod_enabled_ =
    steer_wheel_rpt_ptr_->enabled && accel_rpt_ptr_->enabled && brake_rpt_ptr_->enabled;
  RCLCPP_DEBUG(
    get_logger(),
    "[pacmod] enabled: is_pacmod_enabled_ %d, steer %d, accel %d, brake %d, shift %d, global %d",
    is_pacmod_enabled_, steer_wheel_rpt_ptr_->enabled, accel_rpt_ptr_->enabled,
    brake_rpt_ptr_->enabled, shift_rpt_ptr_->enabled, global_rpt_ptr_->enabled);

  const double current_velocity = calculateVehicleVelocity(
    *wheel_speed_rpt_ptr_, *shift_rpt_ptr_);  // current vehicle speed > 0 [m/s]
  const double current_steer_wheel =
    steer_wheel_rpt_ptr_->output;  // current vehicle steering wheel angle [rad]
  const double adaptive_gear_ratio =
    calculateVariableGearRatio(current_velocity, current_steer_wheel);
  const double current_steer = current_steer_wheel / adaptive_gear_ratio;

  std_msgs::msg::Header header;
  header.frame_id = base_frame_id_;
  header.stamp = now();

  /* publish vehicle status twist */
  {
    autoware_auto_vehicle_msgs::msg::VelocityReport velocity_report;
    velocity_report.header = header;
    velocity_report.longitudinal_velocity = current_velocity;  // [m/s]
    velocity_report.heading_rate =
      current_velocity * std::tan(current_steer) / wheel_base_;  // [rad/s]
    vehicle_twist_pub_->publish(velocity_report);
  }

  /* publish current steering angle */
  {
    autoware_auto_vehicle_msgs::msg::SteeringReport steer_msg;
    steer_msg.stamp = header.stamp;
    steer_msg.steering_tire_angle = current_steer;
    steering_status_pub_->publish(steer_msg);
  }
}

void PacmodSteerTest::publishCommands()
{
  /* guard */
  if (!is_pacmod_rpt_received_) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000.0, "[pacmod]  pacmod3_msgs = %d", is_pacmod_rpt_received_);
    return;
  }

  const rclcpp::Time current_time = now();

  /* check clear flag */
  bool clear_override = false;
  if (is_pacmod_enabled_ == true) {
    is_clear_override_needed_ = false;
  } else if (is_clear_override_needed_ == true) {
    clear_override = true;
  }

  /* make engage cmd false when a driver overrides vehicle control */
  if (!prev_override_ && global_rpt_ptr_->override_active) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000.0, "Pacmod is overridden, enable flag is back to false");
    engage_cmd_ = false;
  }
  prev_override_ = global_rpt_ptr_->override_active;

  /* make engage cmd false when vehicle report is timeouted, e.g. E-stop is depressed */
  const bool report_timeouted = ((now() - global_rpt_ptr_->header.stamp).seconds() > 1.0);
  if (report_timeouted) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000.0,
      "Pacmod report is timeouted, enable flag is back to false");
    engage_cmd_ = false;
  }

  /* make engage cmd false when vehicle fault is active */
  if (global_rpt_ptr_->pacmod_sys_fault_active) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000.0, "Pacmod fault is active, enable flag is back to false");
    engage_cmd_ = false;
  }

  RCLCPP_DEBUG(
    get_logger(),
    "[pacmod] is_pacmod_enabled_ = %d, is_clear_override_needed_ = %d, clear_override = %d",
    is_pacmod_enabled_, is_clear_override_needed_, clear_override);

  /* publish accel cmd */
  {
    pacmod3_msgs::msg::SystemCmdFloat accel_cmd;
    accel_cmd.header.frame_id = base_frame_id_;
    accel_cmd.header.stamp = current_time;
    accel_cmd.enable = engage_cmd_;
    accel_cmd.ignore_overrides = false;
    accel_cmd.clear_override = clear_override;
    accel_cmd.command = getAccel();
    accel_cmd_pub_->publish(accel_cmd);
  }

  /* publish brake cmd */
  {
    pacmod3_msgs::msg::SystemCmdFloat brake_cmd;
    brake_cmd.header.frame_id = base_frame_id_;
    brake_cmd.header.stamp = current_time;
    brake_cmd.enable = engage_cmd_;
    brake_cmd.ignore_overrides = false;
    brake_cmd.clear_override = clear_override;
    brake_cmd.command = getBrake();
    brake_cmd_pub_->publish(brake_cmd);
  }

  /* publish steering cmd */
  {
    pacmod3_msgs::msg::SteeringCmd steer_cmd;
    steer_cmd.header.frame_id = base_frame_id_;
    steer_cmd.header.stamp = current_time;
    steer_cmd.enable = engage_cmd_;
    steer_cmd.ignore_overrides = false;
    steer_cmd.clear_override = clear_override;
    steer_cmd.command = testSteerCommand();  // desired_steer_wheel;
    steer_cmd.rotation_rate = steer_rate_;
    steer_cmd_pub_->publish(steer_cmd);
  }

  /* publish shift cmd */
  {
    pacmod3_msgs::msg::SystemCmdInt shift_cmd;
    shift_cmd.header.frame_id = base_frame_id_;
    shift_cmd.header.stamp = current_time;
    shift_cmd.enable = engage_cmd_;
    shift_cmd.ignore_overrides = false;
    shift_cmd.clear_override = clear_override;
    shift_cmd.command = pacmod3_msgs::msg::SystemCmdInt::SHIFT_FORWARD;  // always drive shift
    shift_cmd_pub_->publish(shift_cmd);
  }

  /* publish shift cmd */
  pacmod3_msgs::msg::SystemCmdInt turn_cmd;
  turn_cmd.header.frame_id = base_frame_id_;
  turn_cmd.header.stamp = current_time;
  turn_cmd.enable = engage_cmd_;
  turn_cmd.ignore_overrides = false;
  turn_cmd.clear_override = clear_override;
  turn_cmd.command = pacmod3_msgs::msg::SystemCmdInt::TURN_HAZARDS;  // for safety
  turn_cmd_pub_->publish(turn_cmd);
}

double PacmodSteerTest::testSteerCommand()
{
  if (test_mode_ == TestMode::SineWave) {
    return sineWave();
  } else if (test_mode_ == TestMode::IncreasedSineWave) {
    return increasedSineWave();
  } else if (test_mode_ == TestMode::StepWithTwoValue) {
    return stepWithTwoValue();
  } else if (test_mode_ == TestMode::StepWithThreeValue) {
    return stepWithThreeValue();
  } else if (test_mode_ == TestMode::IncreasedStepWithTwoValue) {
    return increasedStepWithTwoValue();
  }

  return 0.0;
}

double PacmodSteerTest::sineWave()
{
  static rclcpp::Time ts = now();
  if (!engage_cmd_) {
    ts = now();
    return 0.0;
  }

  const double dt = (now() - ts).seconds();
  return std::sin(2 * M_PI * dt * hz_) + offset_;
}

double PacmodSteerTest::increasedSineWave()
{
  /*
  static double A = 0;      //Amplitude
  const double dA = 0.005;  //Delta Amplitude
  const double hz = 0.5;
  */
  static rclcpp::Time ts = now();
  if (!engage_cmd_ || !checkDriveShift()) {
    ts = now();
    amplitude_ = 0.0;
    return 0.0;
  }

  const double dt = (now() - ts).seconds();

  const double sine = std::sin(2 * M_PI * dt * hz_);
  double sine_sign = sine > 0.0 ? 1.0 : -1.0;
  static double sign = 0;
  if (sign != sine_sign && sine_sign > 0.5) {
    amplitude_ += delta_amplitude_;
  }
  sign = sine_sign;

  return amplitude_ * sine + offset_;
}

double PacmodSteerTest::stepWithTwoValue()
{
  static rclcpp::Time ts = now();
  if (!engage_cmd_ || !checkDriveShift()) {
    ts = now();
    return 0.0;
  }

  const double dt = (now() - ts).seconds();

  const double sine = std::sin(2 * M_PI * dt * hz_);
  double sine_sign = sine > 0.0 ? 1.0 : -1.0;
  return amplitude_ * sine_sign;
}

double PacmodSteerTest::stepWithThreeValue()
{
  static rclcpp::Time ts = now();
  if (!engage_cmd_ || !checkDriveShift()) {
    ts = now();
    return 0.0;
  }

  const double dt = (now() - ts).seconds();

  const double sine = std::sin(2 * M_PI * dt * hz_);
  const double thr = std::sin(M_PI * 1.0 / 4.0);
  double sine_sign_with_zero = sine > thr ? 1.0 : (sine < -thr ? -1.0 : 0.0);
  return amplitude_ * sine_sign_with_zero + offset_;
}

double PacmodSteerTest::increasedStepWithTwoValue()
{
  static rclcpp::Time ts = now();
  if (!engage_cmd_ || !checkDriveShift()) {
    ts = now();
    amplitude_ = 0.0;
    return 0.0;
  }

  const double dt = (now() - ts).seconds();

  const double sine = std::sin(2 * M_PI * dt * hz_);
  double sine_sign = sine > 0.0 ? 1.0 : -1.0;
  static double sign = 0;
  if (sign != sine_sign && sine_sign > 0.5) {
    amplitude_ += delta_amplitude_;
  }
  sign = sine_sign;

  return amplitude_ * sign + offset_;
}

bool PacmodSteerTest::checkDriveShift()
{
  /* check shift  */
  if (shift_rpt_ptr_->output != pacmod3_msgs::msg::SystemCmdInt::SHIFT_FORWARD) {  // need shift
                                                                                   // change.
    RCLCPP_WARN_STREAM_THROTTLE(get_logger(), *get_clock(), 5000.0, "current gear is not DRIVE.");
    return false;
  }
  return true;
}

double PacmodSteerTest::getAccel()
{
  if (!checkDriveShift()) {
    return 0.0;
  }

  if (stopping_) {
    // stopping
    return 0.0;
  } else {
    // return accel_value_(ros params)
    return accel_value_;
  }
}

double PacmodSteerTest::getBrake()
{
  if (!checkDriveShift()) {
    return brake_for_shift_trans;
  }

  if (stopping_) {
    // stopping
    return 1.0;
  } else {
    // return brake_value_(ros params)
    return brake_value_;
  }
}

double PacmodSteerTest::calculateVehicleVelocity(
  const pacmod3_msgs::msg::WheelSpeedRpt & wheel_speed_rpt,
  const pacmod3_msgs::msg::SystemRptInt & shift_rpt)
{
  const double sign = (shift_rpt.output == pacmod3_msgs::msg::SystemRptInt::SHIFT_REVERSE) ? -1 : 1;
  const double vel =
    (wheel_speed_rpt.rear_left_wheel_speed + wheel_speed_rpt.rear_right_wheel_speed) * 0.5 *
    tire_radius_;
  return sign * vel;
}

double PacmodSteerTest::calculateVariableGearRatio(const double vel, const double steer_wheel)
{
  return std::max(
    1e-5, vgr_coef_a_ + vgr_coef_b_ * vel * vel - vgr_coef_c_ * std::fabs(steer_wheel));
}

/*
 * Copyright 2020 Tier IV, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Copyright (c) 2019, Map IV, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the Map IV, Inc. nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/*
 * tag_serial_driver.cpp
 * Tamagawa IMU Driver
 * Author MapIV Sekino
 * Ver 1.00 2019/4/4
 */

#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/int32.hpp"

#include <sys/ioctl.h>

std::string device = "/dev/ttyUSB0";
std::string imu_type = "noGPS";
std::string rate = "50";

struct termios old_conf_tio;
struct termios conf_tio;

int fd;
int counter;
int raw_data;

sensor_msgs::msg::Imu imu_msg;

int serial_setup(const char * device)
{
  int fd = open(device, O_RDWR);

  speed_t BAUDRATE = B115200;

  conf_tio.c_cflag += CREAD;   // 受信有効
  conf_tio.c_cflag += CLOCAL;  // ローカルライン（モデム制御なし）
  conf_tio.c_cflag += CS8;     // データビット:8bit
  conf_tio.c_cflag += 0;       // ストップビット:1bit
  conf_tio.c_cflag += 0;

  cfsetispeed(&conf_tio, BAUDRATE);
  cfsetospeed(&conf_tio, BAUDRATE);

  tcsetattr(fd, TCSANOW, &conf_tio);
  ioctl(fd, TCSETS, &conf_tio);
  return fd;
}

void receive_ver_req([[maybe_unused]] const std_msgs::msg::Int32::ConstSharedPtr msg)
{
  char ver_req[] = "$TSC,VER*29\x0d\x0a";
  int ver_req_data = write(fd, ver_req, sizeof(ver_req));
  if (ver_req_data >= 0) {
    RCLCPP_INFO(rclcpp::get_logger("tag_serial_driver"), "Send Version Request: %s", ver_req);
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("tag_serial_driver"), "ERROR! Send Version Request: %s", ver_req);
  }
}

void receive_offset_cancel_req(const std_msgs::msg::Int32::ConstSharedPtr msg)
{
  char offset_cancel_req[32];
  sprintf(offset_cancel_req, "$TSC,OFC,%d\x0d\x0a", msg->data);
  int offset_cancel_req_data = write(fd, offset_cancel_req, sizeof(offset_cancel_req));
  if (offset_cancel_req_data >= 0) {
    RCLCPP_INFO(rclcpp::get_logger("tag_serial_driver"), "Send Offset Cancel Request: %s", offset_cancel_req);
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("tag_serial_driver"), "ERROR! Send Offset Cancel Request: %s", offset_cancel_req);
  }

}

void receive_heading_reset_req([[maybe_unused]] const std_msgs::msg::Int32::ConstSharedPtr msg)
{
  char heading_reset_req[] = "$TSC,HRST*29\x0d\x0a";
  int heading_reset_req_data = write(fd, heading_reset_req, sizeof(heading_reset_req));
  if (heading_reset_req_data >= 0) {
    RCLCPP_INFO(rclcpp::get_logger("tag_serial_driver"), "Send Heading reset Request: %s", heading_reset_req);
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("tag_serial_driver"), "ERROR! Send Heading reset Request: %s", heading_reset_req);
  }
}

void shutdown_cmd([[maybe_unused]] int sig)
{
  tcsetattr(fd, TCSANOW, &old_conf_tio);  // Revert to previous settings
  close(fd);
  RCLCPP_INFO(rclcpp::get_logger("tag_serial_driver"), "Port closed");
  rclcpp::shutdown();
}

#include <boost/asio.hpp>
using namespace boost::asio;

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("tag_serial_driver");
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub = node->create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", 1000);
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr sub1 = node->create_subscription<std_msgs::msg::Int32>("receive_ver_req", 10, receive_ver_req);
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr sub2 = node->create_subscription<std_msgs::msg::Int32>("receive_offset_cancel_req", 10, receive_offset_cancel_req);
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr sub3 = node->create_subscription<std_msgs::msg::Int32>("receive_heading_reset_req", 10, receive_heading_reset_req);

  std::string imu_frame_id = node->declare_parameter<std::string>("imu_frame_id", "imu");

  std::string port = node->declare_parameter<std::string>("port", "/dev/ttyUSB0");

  io_service io;
  serial_port serial_port(io, port.c_str());
  serial_port.set_option(serial_port_base::baud_rate(115200));
  serial_port.set_option(serial_port_base::character_size(8));
  serial_port.set_option(serial_port_base::flow_control(serial_port_base::flow_control::none));
  serial_port.set_option(serial_port_base::parity(serial_port_base::parity::none));
  serial_port.set_option(serial_port_base::stop_bits(serial_port_base::stop_bits::one));

  std::string wbuf = "$TSC,BIN,30\x0d\x0a";
  std::size_t length;
  serial_port.write_some(buffer(wbuf));

  rclcpp::Rate loop_rate(30.0);

  imu_msg.orientation.x = 0.0;
  imu_msg.orientation.y = 0.0;
  imu_msg.orientation.z = 0.0;
  imu_msg.orientation.w = 1.0;

  while (rclcpp::ok()) {
    rclcpp::spin_some(node);

    boost::asio::streambuf response;
    boost::asio::read_until(serial_port, response, "\n");
    std::string rbuf(
      boost::asio::buffers_begin(response.data()), boost::asio::buffers_end(response.data()));

    length = rbuf.size();

    if (length > 0) {
      if (rbuf[5] == 'B' && rbuf[6] == 'I' && rbuf[7] == 'N' && rbuf[8] == ',' && length == 58) {
        imu_msg.header.frame_id = imu_frame_id;
        imu_msg.header.stamp = node->now();

        counter = ((rbuf[11] << 8) & 0x0000FF00) | (rbuf[12] & 0x000000FF);
        raw_data = ((((rbuf[15] << 8) & 0xFFFFFF00) | (rbuf[16] & 0x000000FF)));
        imu_msg.angular_velocity.x =
          raw_data * (200 / pow(2, 15)) * M_PI / 180;  // LSB & unit [deg/s] => [rad/s]
        raw_data = ((((rbuf[17] << 8) & 0xFFFFFF00) | (rbuf[18] & 0x000000FF)));
        imu_msg.angular_velocity.y =
          raw_data * (200 / pow(2, 15)) * M_PI / 180;  // LSB & unit [deg/s] => [rad/s]
        raw_data = ((((rbuf[19] << 8) & 0xFFFFFF00) | (rbuf[20] & 0x000000FF)));
        imu_msg.angular_velocity.z =
          raw_data * (200 / pow(2, 15)) * M_PI / 180;  // LSB & unit [deg/s] => [rad/s]
        raw_data = ((((rbuf[21] << 8) & 0xFFFFFF00) | (rbuf[22] & 0x000000FF)));
        imu_msg.linear_acceleration.x = raw_data * (100 / pow(2, 15));  // LSB & unit [m/s^2]
        raw_data = ((((rbuf[23] << 8) & 0xFFFFFF00) | (rbuf[24] & 0x000000FF)));
        imu_msg.linear_acceleration.y = raw_data * (100 / pow(2, 15));  // LSB & unit [m/s^2]
        raw_data = ((((rbuf[25] << 8) & 0xFFFFFF00) | (rbuf[26] & 0x000000FF)));
        imu_msg.linear_acceleration.z = raw_data * (100 / pow(2, 15));  // LSB & unit [m/s^2]

        pub->publish(imu_msg);

      } else if (rbuf[5] == 'V' && rbuf[6] == 'E' && rbuf[7] == 'R' && rbuf[8] == ',') {
        RCLCPP_DEBUG(rclcpp::get_logger("tag_serial_driver"), "%s", rbuf.c_str());
      }
    }
  }

  return 0;
}

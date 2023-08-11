/*
 *  Copyright (C) 2007 Austin Robot Technology, Patrick Beeson
 *  Copyright (C) 2009-2012 Austin Robot Technology, Jack O'Quin
 *  Copyright (C) 2017, Velodyne LiDAR INC., Algorithms and Signal Processing Group
 *  Copyright (C) 2020, Tier IV, Inc., David Robert Wong
 *
 *  License: Modified BSD Software License Agreement
 *
 *  $Id$
 */

/** \file
 *
 *  ROS driver implementation for the Velodyne 3D LIDARs
 */

#include <string>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>

#include "driver.h"

namespace velodyne_driver
{
  inline   std::string toBinary(int n)
  {
        std::string r;
        while(n!=0) {r=(n%2==0 ?"0":"1")+r; n/=2;}
        while (r.length() != 8){
          r = '0' + r;
        }
        return r;
  }

  inline   double convertBinaryToDecimal(std::string binaryString)
  {
      double value = 0;
      int indexCounter = 0;
      for(int i=binaryString.length()-1;i>=0;i--){

          if(binaryString[i]=='1'){
              value += pow(2, indexCounter);
          }
          indexCounter++;
      }
      return value;
  }


/** Utility function for Velodyne Driver
 *  gets the number of laser beams fired concurrently
 *  for different sensor models
*/

inline int get_concurrent_beams(uint8_t sensor_model)
{
/*
Strongest 0x37 (55)   HDL-32E 0x21 (33)
Last Return 0x38 (56) VLP-16 0x22 (34)
Dual Return 0x39 (57) Puck LITE 0x22 (34)
         -- --        Puck Hi-Res 0x24 (36)
         -- --        VLP-32C 0x28 (40)
         -- --        Velarray 0x31 (49)
         -- --        VLS-128 0xA1 (161)
*/

  switch(sensor_model)
  {
    case 33:
        return(2); // hdl32e
    case 34:
        return(1); // vlp16 puck lite
    case 36:
        return(1); // puck hires  (same as vlp16 ?? need to check)
    case 40:
        return(2); // vlp32c
    case 49:
        return(2); // velarray
    case 161:
        return(8); // vls128
    case 99:
        return(8); // vls128
    default:
        RCLCPP_WARN_STREAM(rclcpp::get_logger("get_concurrent_beams"), "[Velodyne Ros driver]Default assumption of device id .. Defaulting to HDL64E with 2 simultaneous firings");
        return(2); // hdl-64e

  }
}

/** Utility function for Velodyne Driver
 *  gets the number of packet multiplier for dual return mode vs
 *  single return mode
*/

inline int get_rmode_multiplier(uint8_t sensor_model, uint8_t packet_rmode)
{
 /*
    HDL64E 2
    VLP32C 2
    HDL32E 2
    VLS128 3
    VLSP16 2
*/
  if(packet_rmode  == 57)
  {
    switch(sensor_model)
    {
      case 33:
          return(2); // hdl32e
      case 34:
          return(2); // vlp16 puck lite
      case 36:
          return(2); // puck hires
      case 40:
          return(2); // vlp32c
      case 49:
          return(2); // velarray
      case 161:
          return(3); // vls128
      case 99:
          return(3); // vls128
      default:
          RCLCPP_WARN_STREAM(rclcpp::get_logger("get_rmode_multiplier"), "[Velodyne Ros driver]Default assumption of device id .. Defaulting to HDL64E with 2x number of packekts for Dual return");
          return(2); // hdl-64e
    }
   }
   else
   {
     return(1);
   }
}

/** \brief For parameter service callback */
template <typename T>
bool get_param(const std::vector<rclcpp::Parameter> & p, const std::string & name, T & value)
{
  auto it = std::find_if(p.cbegin(), p.cend(), [&name](const rclcpp::Parameter & parameter) {
    return parameter.get_name() == name;
  });
  if (it != p.cend()) {
    value = it->template get_value<T>();
    return true;
  }
  return false;
}

/** Constructor for the Velodyne driver
 *
 *  provides a binding to ROS node for processing and
 *  configuration
 *  @returns handle to driver object
 */

VelodyneDriverCore::VelodyneDriverCore(rclcpp::Node * node_ptr)
: node_ptr_(node_ptr),
  diagnostics_(node_ptr_, 0.2)
{
  // use private node handle to get parameters
  config_.frame_id = node_ptr_->declare_parameter("frame_id", std::string("velodyne"));

  // TODO (mitsudome-r) : port this when getPrefix becomes available in ROS2
  // std::string tf_prefix = tf::getPrefixParam(private_nh);
  //  RCLCPP_DEBUG_STREAM(node_ptr_->get_logger(), "tf_prefix: " << tf_prefix);
  // config_.frame_id = tf::resolve(tf_prefix, config_.frame_id);

  // get model name, validate string, determine packet rate
  config_.model = node_ptr_->declare_parameter("model", std::string("64E"));
  std::string model_full_name;
  if ((config_.model == "64E_S2") ||
      (config_.model == "64E_S2.1"))
    {
      model_full_name = std::string("HDL-") + config_.model;
    }
  else if (config_.model == "64E")
    {
      model_full_name = std::string("HDL-") + config_.model;
    }
  else if (config_.model == "64E_S3")
    {
      model_full_name = std::string("HDL-") + config_.model;
    }
  else if (config_.model == "32E")
    {
      model_full_name = std::string("HDL-") + config_.model;
    }
    else if (config_.model == "32C")
    {
      model_full_name = std::string("VLP-") + config_.model;
    }
  else if (config_.model == "VLP16")
    {
      model_full_name = "VLP-16";
    }
  else if (config_.model == "VLS128")
    {
      model_full_name = "VLS-128";
    }
  else
    {
      RCLCPP_ERROR_STREAM(node_ptr_->get_logger(), "Unknown Velodyne LIDAR model: " << config_.model);
    }
  std::string deviceName(std::string("Velodyne ") + model_full_name);

  config_.rpm = node_ptr_->declare_parameter("rpm", 600.0);
  config_.rpm = node_ptr_->get_parameter("rpm").as_double();
  RCLCPP_INFO_STREAM(node_ptr_->get_logger(), deviceName << " rotating at " << config_.rpm << " RPM");
  double frequency = (config_.rpm / 60.0);     // expected Hz rate

  config_.scan_phase = node_ptr_->declare_parameter("scan_phase", 0.0);
  config_.scan_phase = node_ptr_->get_parameter("scan_phase").as_double();
  RCLCPP_INFO_STREAM(node_ptr_->get_logger(), "Scan start/end will be at a phase of " << config_.scan_phase  << " degrees");

  config_.time_offset = node_ptr_->declare_parameter("time_offset", 0.0);
  config_.time_offset = node_ptr_->get_parameter("time_offset").as_double();
  RCLCPP_INFO_STREAM(
    node_ptr_->get_logger(),
    "time in seconds added to each velodyne time stamp " << config_.time_offset  << " s");

  dump_file = node_ptr_->declare_parameter("pcap", std::string(""));

  int udp_port;
  udp_port = node_ptr_->declare_parameter("port", (int) DATA_PORT_NUMBER);

  // Initialize dynamic reconfigure
  using std::placeholders::_1;
  set_param_res_ = node_ptr_->add_on_set_parameters_callback(
    std::bind(&VelodyneDriverCore::paramCallback, this, _1));

  // Initialize diagnostics
  diagnostics_.setHardwareID(deviceName);
  const double diag_freq = frequency;
  diag_max_freq_ = diag_freq;
  diag_min_freq_ = diag_freq;
  RCLCPP_INFO(node_ptr_->get_logger(), "Expected frequency: %.3f (Hz)", diag_freq);

  using namespace diagnostic_updater;
  diag_topic_.reset(new TopicDiagnostic("velodyne_packets", diagnostics_,
                                        FrequencyStatusParam(&diag_min_freq_,
                                                             &diag_max_freq_,
                                                             0.1, 10),
                                        TimeStampStatusParam()));

  // open Velodyne input device or file
  if (dump_file != "")                  // have PCAP file?
    {
      // read data from packet capture file
      input_.reset(new velodyne_driver::InputPCAP(node_ptr_, udp_port, dump_file));
    }
  else
    {
      // read data from live socket
      input_.reset(new velodyne_driver::InputSocket(node_ptr_, udp_port));
    }

  // raw packet output topic
  output_ =
    node_ptr_->create_publisher<velodyne_msgs::msg::VelodyneScan>(
      "velodyne_packets", rclcpp::SensorDataQoS());
}

/** poll the device
 *
 * poll is used by nodelet to bind to the ROS thread.
 *  @returns true unless end of file reached
 */
bool VelodyneDriverCore::poll(void)
{
  // Allocate a new shared pointer for zero-copy sharing with other nodelets.
  auto scan = std::make_shared<velodyne_msgs::msg::VelodyneScan>();

  // Since the velodyne delivers data at a very high rate, keep
  // reading and publishing scans as fast as possible.
  uint16_t packet_first_azm = 0;
  uint16_t packet_first_azm_phased = 0;
  uint16_t packet_last_azm = 0;
  uint16_t packet_last_azm_phased = 0;
  uint16_t prev_packet_first_azm_phased = 0;

  uint16_t phase = (uint16_t)round(config_.scan_phase*100);
  bool use_next_packet = true;
  uint processed_packets = 0;
  while (use_next_packet && rclcpp::ok())
  {
    while (rclcpp::ok())
    {
        // keep reading until full packet received
        velodyne_msgs::msg::VelodynePacket new_packet;
        scan->packets.push_back(new_packet);
        int rc = input_->getPacket(&scan->packets.back(), config_.time_offset);
        if (rc == 1) break;       // got a full packet?
        if (rc < 0) return false; // end of file reached?
	if (rc == 0) continue; // timeout?
    }
    processed_packets++;

    // uint8_t  curr_packet_rmode;
    packet_first_azm  = scan->packets.back().data[2]; // lower word of azimuth block 0
    packet_first_azm |= scan->packets.back().data[3] << 8; // higher word of azimuth block 0

    packet_last_azm = scan->packets.back().data[1102];
    packet_last_azm |= scan->packets.back().data[1103] << 8;

    // curr_packet_rmode = scan->packets.back().data[1204];
    // curr_packet_sensor_model = scan->packets.back().data[1205];

    // For correct pointcloud assembly, always stop the scan after passing the
    // zero phase point. The pointcloud assembler will remedy this after unpacking
    // the packets, by buffering the overshot azimuths for the next cloud.
    // NOTE: this also works for dual echo mode because the last blank data block
    // still contains azimuth data (for VLS128). This should be modified in future
    // to concretely handle blank data blocks.
    packet_first_azm_phased = (36000 + packet_first_azm - phase) % 36000;
    packet_last_azm_phased = (36000 + packet_last_azm - phase) % 36000;
    if (processed_packets > 1)
    {
      if (packet_last_azm_phased < packet_first_azm_phased || packet_first_azm_phased < prev_packet_first_azm_phased)
      {
        use_next_packet = false;
      }
    }
    prev_packet_first_azm_phased = packet_first_azm_phased;
  }

  // average the time stamp from first package and last package
  rclcpp::Time firstTimeStamp = scan->packets.front().stamp;
  rclcpp::Time lastTimeStamp = scan->packets.back().stamp;
  rclcpp::Time  meanTimeStamp = firstTimeStamp + (lastTimeStamp - firstTimeStamp) * 0.5;

  // publish message using time of first packet read
  RCLCPP_DEBUG(node_ptr_->get_logger(), "Publishing a full Velodyne scan.");
  scan->header.stamp = scan->packets.front().stamp;
  // scan->scan->header.stamp = scan->packets[scan->packets.size()/2].stamp;
  scan->header.frame_id = config_.frame_id;
  output_->publish(*scan);
  // notify diagnostics that a message has been published, updating
  // its status
  diag_topic_->tick(scan->header.stamp);

  return true;
}

rcl_interfaces::msg::SetParametersResult VelodyneDriverCore::paramCallback(const std::vector<rclcpp::Parameter> & p)
{
  RCLCPP_INFO(node_ptr_->get_logger(), "Reconfigure Request");

  if (get_param(p, "time_offset", config_.time_offset)) {
    RCLCPP_DEBUG(node_ptr_->get_logger(), "Setting new time_offset to: %f.", config_.time_offset);
  }
  if (get_param(p, "scan_phase", config_.scan_phase)) {
    RCLCPP_DEBUG(node_ptr_->get_logger(), "Setting scan_phase to: %f.", config_.scan_phase);
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  return result;

}

} // namespace velodyne_driver

/* -*- mode: C++ -*- */
/*
 *  Copyright (C) 2012 Austin Robot Technology, Jack O'Quin
 *  Copyright (C) 2020, Tier IV, Inc., David Robert Wong
 *
 *  License: Modified BSD Software License Agreement
 *
 *  $Id$
 */

/** \file
 *
 *  ROS driver interface for the Velodyne 3D LIDARs
 */

#ifndef _VELODYNE_DRIVER_H_
#define _VELODYNE_DRIVER_H_ 1

#include <string>
#include <rclcpp/rclcpp.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <diagnostic_updater/publisher.hpp>
#include <velodyne_msgs/msg/velodyne_scan.hpp>

#include <velodyne_driver/input.h>

namespace velodyne_driver
{

class VelodyneDriverCore
{
public:

  VelodyneDriverCore(rclcpp::Node * node_ptr);
  ~VelodyneDriverCore() {}

  bool poll(void);

private:

  // opinter to node for loggers and clocks
  rclcpp::Node * node_ptr_;

  ///Callback for parameter service
  rcl_interfaces::msg::SetParametersResult paramCallback(const std::vector<rclcpp::Parameter> & p);

  ///Pointer to parameter update service
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr set_param_res_;

  // configuration parameters
  struct
  {
    std::string frame_id;            ///< tf frame ID
    std::string model;               ///< device model name
    double rpm;                      ///< device rotation rate (RPMs)
    double scan_phase;               ///< scan phase (degrees)
    double time_offset;              ///< time in seconds added to each velodyne time stamp
  } config_;

  std::shared_ptr<Input> input_;
  rclcpp::Publisher<velodyne_msgs::msg::VelodyneScan>::SharedPtr output_;

  /** diagnostics updater */
  diagnostic_updater::Updater diagnostics_;
  double diag_min_freq_;
  double diag_max_freq_;
  std::shared_ptr<diagnostic_updater::TopicDiagnostic> diag_topic_;

  // uint8_t  curr_packet_rmode; //    [strongest return or farthest mode => Singular Retruns per firing]
                              // or [Both  => Dual Retruns per fire]
  // uint8_t  curr_packet_sensor_model; // extract the sensor id from packet
  std::string dump_file; // string to hold pcap file name
};

} // namespace velodyne_driver

#endif // _VELODYNE_DRIVER_H_

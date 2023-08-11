/*
 *  Copyright (C) 2012 Austin Robot Technology, Jack O'Quin
 * 
 *  License: Modified BSD Software License Agreement
 *
 *  $Id$
 */

/** \file
 *
 *  ROS driver nodelet for the Velodyne 3D LIDARs
 */

#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
// #include <pluginlib/class_list_macros.h>
// #include <nodelet/nodelet.h>

#include "driver.h"

namespace velodyne_driver
{

class VelodyneDriver: public rclcpp::Node
{
public:

  VelodyneDriver(const rclcpp::NodeOptions & options)
  : Node("velodyne_driver_node", options),
    running_(false)
  {
    onInit();
  }

  ~VelodyneDriver()
  {
    if (running_)
      {
        RCLCPP_INFO(this->get_logger(), "shutting down driver thread");
        running_ = false;
        deviceThread_->join();
        RCLCPP_INFO(this->get_logger(), "driver thread stopped");
      }
  }

private:

  virtual void onInit(void);
  virtual void devicePoll(void);

  volatile bool running_;               ///< device thread is running
  std::shared_ptr<std::thread> deviceThread_;

  std::shared_ptr<VelodyneDriverCore> dvr_; ///< driver implementation class
};

void VelodyneDriver::onInit()
{
  // start the driver
  dvr_.reset(new VelodyneDriverCore(this));

  // spawn device poll thread
  running_ = true;
  deviceThread_ = std::shared_ptr< std::thread >
    (new std::thread(std::bind(&VelodyneDriver::devicePoll, this)));
}

/** @brief Device poll thread main loop. */
void VelodyneDriver::devicePoll()
{
  while(rclcpp::ok())
  {
    // poll device until end of file
    running_ = dvr_->poll();
    if (!running_)
      break;
  }
  running_ = false;
}

} // namespace velodyne_driver

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(velodyne_driver::VelodyneDriver)

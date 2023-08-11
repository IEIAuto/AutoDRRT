/* -*- mode: C++ -*- */
/*
 *  Copyright (C) 2009, 2010 Austin Robot Technology, Jack O'Quin
 *  Copyright (C) 2011 Jesse Vera
 *  Copyright (C) 2012 Austin Robot Technology, Jack O'Quin
 *  License: Modified BSD Software License Agreement
 *
 *  $Id$
 */

#ifndef _VELODYNE_POINTCLOUD_INTERPOLATE_H_
#define _VELODYNE_POINTCLOUD_INTERPOLATE_H_ 1

#include <deque>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <tf2/convert.h>
#include <tf2/transform_datatypes.h>

#ifdef USE_TF2_GEOMETRY_MSGS_DEPRECATED_HEADER
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <tf2_ros/transform_listener.h>

#include <autoware_auto_vehicle_msgs/msg/velocity_report.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <velodyne_pointcloud/pointcloudXYZIRADT.h>

namespace velodyne_pointcloud
{
class Interpolate : public rclcpp::Node
{
public:
  Interpolate(const rclcpp::NodeOptions & options);
  ~Interpolate() {}

private:
  /** \brief Parameter service callback */
  rcl_interfaces::msg::SetParametersResult paramCallback(const std::vector<rclcpp::Parameter> & p);

  void processPoints(
    const sensor_msgs::msg::PointCloud2::SharedPtr points_xyziradt);
  void processVelocityReport(const autoware_auto_vehicle_msgs::msg::VelocityReport::SharedPtr velocity_report_msg);
  bool getTransform(
    const std::string & target_frame, const std::string & source_frame,
    tf2::Transform * tf2_transform_ptr);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr velodyne_points_ex_sub_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::VelocityReport>::SharedPtr velocity_report_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr velodyne_points_interpolate_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr velodyne_points_interpolate_ex_pub_;

  tf2::BufferCore tf2_buffer_;
  tf2_ros::TransformListener tf2_listener_;

  std::deque<autoware_auto_vehicle_msgs::msg::VelocityReport> velocity_report_queue_;

  std::string base_link_frame_;
};

}  // namespace velodyne_pointcloud

#endif  // _VELODYNE_POINTCLOUD_INTERPOLATE_H_

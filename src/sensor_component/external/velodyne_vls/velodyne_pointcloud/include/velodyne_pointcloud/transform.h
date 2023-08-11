/* -*- mode: C++ -*- */
/*
 *  Copyright (C) 2009, 2010 Austin Robot Technology, Jack O'Quin
 *  Copyright (C) 2011 Jesse Vera
 *  Copyright (C) 2012 Austin Robot Technology, Jack O'Quin
 *  License: Modified BSD Software License Agreement
 *
 *  $Id$
 */

/** @file

    This class transforms raw Velodyne 3D LIDAR packets to PointCloud2
    in the /odom frame of reference.

*/

#ifndef _VELODYNE_POINTCLOUD_TRANSFORM_H_
#define _VELODYNE_POINTCLOUD_TRANSFORM_H_ 1

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <velodyne_msgs/msg/velodyne_scan.hpp>
#include <message_filters/subscriber.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include <velodyne_pointcloud/pointcloudXYZIR.h>
#include <velodyne_pointcloud/rawdata.h>

// include template implementations to transform a custom point cloud
#include <pcl_ros/impl/transforms.hpp>

// instantiate template for transforming a VPointCloud
template bool pcl_ros::transformPointCloud<velodyne_rawdata::VPoint>(
  const std::string &, const velodyne_rawdata::VPointCloud &, velodyne_rawdata::VPointCloud &,
  const tf2_ros::Buffer & tf_buffer);

namespace velodyne_pointcloud
{
class Transform : public rclcpp::Node
{
public:
  Transform(const rclcpp::NodeOptions & options);
  ~Transform() {}

private:
  void processScan(const velodyne_msgs::msg::VelodyneScan::ConstSharedPtr & scanMsg);

  /// Pointer to dynamic reconfigure service srv_
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr set_param_res_;
  rcl_interfaces::msg::SetParametersResult paramCallback(const std::vector<rclcpp::Parameter> & p);

  const std::string transaction_safe;
  std::shared_ptr<velodyne_rawdata::RawData> data_;
  message_filters::Subscriber<velodyne_msgs::msg::VelodyneScan> velodyne_scan_;
  std::shared_ptr<tf2_ros::MessageFilter<velodyne_msgs::msg::VelodyneScan>> tf_filter_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr output_;
  tf2_ros::Buffer tf_buffer_;

  /// configuration parameters
  typedef struct
  {
    double min_range;
    double max_range;
    double view_direction;
    double view_width;
    std::string frame_id;  ///< target frame ID
  } Config;
  Config config_;

  // Point cloud buffers for collecting points within a packet.  The
  // inPc_ and tfPc_ are class members only to avoid reallocation on
  // every message.
  PointcloudXYZIR inPc_;                ///< input packet point cloud
  velodyne_rawdata::VPointCloud tfPc_;  ///< transformed packet point cloud
};

}  // namespace velodyne_pointcloud

#endif  // _VELODYNE_POINTCLOUD_TRANSFORM_H_

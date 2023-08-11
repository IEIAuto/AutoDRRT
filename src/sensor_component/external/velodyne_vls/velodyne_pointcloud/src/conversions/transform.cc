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

    @author Jack O'Quin
    @author Jesse Vera

*/

#include "velodyne_pointcloud/transform.h"

#include <pcl_conversions/pcl_conversions.h>

namespace velodyne_pointcloud
{

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

/** @brief Constructor. */
Transform::Transform(const rclcpp::NodeOptions & options)
: Node("velodyne_tranform_node", options),
  velodyne_scan_(this, "velodyne_packets"),
  tf_buffer_(this->get_clock())
{
  data_ = std::make_shared<velodyne_rawdata::RawData>(this);

  rcl_interfaces::msg::ParameterDescriptor min_range_desc;
  min_range_desc.name = "min_range";
  min_range_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE;
  min_range_desc.description = "minimum range to publish";
  rcl_interfaces::msg::FloatingPointRange min_range_range;
  min_range_range.from_value = 0.1;
  min_range_range.to_value = 10.0;
  min_range_desc.floating_point_range.push_back(min_range_range);
  config_.min_range = this->declare_parameter("min_range", 0.9, min_range_desc);

  rcl_interfaces::msg::ParameterDescriptor max_range_desc;
  max_range_desc.name = "max_range";
  max_range_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE;
  max_range_desc.description = "maximum range to publish";
  rcl_interfaces::msg::FloatingPointRange max_range_range;
  max_range_range.from_value = 0.1;
  max_range_range.to_value = 250.0;
  max_range_desc.floating_point_range.push_back(max_range_range);
  config_.max_range = this->declare_parameter("max_range", 130.0, max_range_desc);

  rcl_interfaces::msg::ParameterDescriptor view_direction_desc;
  view_direction_desc.name = "view_direction";
  view_direction_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE;
  view_direction_desc.description = "angle defining the center of view";
  rcl_interfaces::msg::FloatingPointRange view_direction_range;
  view_direction_range.from_value = -M_PI;
  view_direction_range.to_value = M_PI;
  view_direction_desc.floating_point_range.push_back(view_direction_range);
  config_.view_direction = this->declare_parameter("view_direction", 0.0, view_direction_desc);

  rcl_interfaces::msg::ParameterDescriptor view_width_desc;
  view_width_desc.name = "view_width";
  view_width_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE;
  view_width_desc.description = "angle defining the view width";
  rcl_interfaces::msg::FloatingPointRange view_width_range;
  view_width_range.from_value = 0.0;
  view_width_range.to_value = 2.0 * M_PI;
  view_width_desc.floating_point_range.push_back(view_width_range);
  config_.view_width = this->declare_parameter("view_width", 2.0 * M_PI, view_width_desc);

  config_.frame_id = this->declare_parameter("frame_id", "odom");

  this->declare_parameter("calibration", "");

  // Read calibration.
  data_->setup();
  data_->setParameters(
    config_.min_range, config_.max_range, config_.view_direction, config_.view_width);

  // advertise output point cloud (before subscribing to input data)
  output_ =
    this->create_publisher<sensor_msgs::msg::PointCloud2>("velodyne_points", rclcpp::SensorDataQoS());

  using std::placeholders::_1;
  set_param_res_ = this->add_on_set_parameters_callback(
    std::bind(&Transform::paramCallback, this, _1));

  // subscribe to VelodyneScan packets using transform filter
  tf_filter_ = std::make_shared<tf2_ros::MessageFilter<velodyne_msgs::msg::VelodyneScan>>(
    velodyne_scan_, tf_buffer_, config_.frame_id, 10,
    this->get_node_logging_interface(), this->get_node_clock_interface());
  tf_filter_->registerCallback(std::bind(&Transform::processScan, this, _1));
}

rcl_interfaces::msg::SetParametersResult Transform::paramCallback(const std::vector<rclcpp::Parameter> & p)
{
  RCLCPP_INFO_STREAM(this->get_logger(), "Reconfigure request.");

  if(get_param(p, "min_range", config_.min_range) ||
     get_param(p, "max_range", config_.max_range) ||
     get_param(p, "view_direction", config_.view_direction) ||
     get_param(p, "view_width", config_.view_width))
  {
    data_->setParameters(
      config_.min_range, config_.max_range, config_.view_direction, config_.view_width);
  }
  if(get_param(p, "frame_id", config_.frame_id))
  {
    RCLCPP_INFO_STREAM(this->get_logger(), "Target frame ID: " << config_.frame_id);
  }
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  return result;
}

/** @brief Callback for raw scan messages.
   *
   *  @pre TF message filter has already waited until the transform to
   *       the configured @c frame_id can succeed.
   */
void Transform::processScan(const velodyne_msgs::msg::VelodyneScan::ConstSharedPtr & scanMsg)
{
  if (output_->get_subscription_count() == 0 &&
    output_->get_intra_process_subscription_count() == 0)    // no one listening?
  {
    return;
  }

  // allocate an output point cloud with same time as raw data
  velodyne_rawdata::VPointCloud::Ptr outMsg(new velodyne_rawdata::VPointCloud());
  outMsg->header.stamp = pcl_conversions::toPCL(scanMsg->header).stamp;
  outMsg->header.frame_id = config_.frame_id;
  outMsg->height = 1;

  // process each packet provided by the driver
  for (size_t next = 0; next < scanMsg->packets.size(); ++next) {
    // clear input point cloud to handle this packet
    inPc_.pc->points.clear();
    inPc_.pc->width = 0;
    inPc_.pc->height = 1;
    std_msgs::msg::Header header;
    header.stamp = scanMsg->packets[next].stamp;
    header.frame_id = scanMsg->header.frame_id;
    pcl_conversions::toPCL(header, inPc_.pc->header);

    // unpack the raw data
    data_->unpack(scanMsg->packets[next], inPc_);

    // clear transform point cloud for this packet
    tfPc_.points.clear();  // is this needed?
    tfPc_.width = 0;
    tfPc_.height = 1;
    header.stamp = scanMsg->packets[next].stamp;
    pcl_conversions::toPCL(header, tfPc_.header);
    tfPc_.header.frame_id = config_.frame_id;

    // transform the packet point cloud into the target frame
    try {
      RCLCPP_DEBUG_STREAM(this->get_logger(), 
        "transforming from " << inPc_.pc->header.frame_id << " to " << config_.frame_id);
      pcl_ros::transformPointCloud(config_.frame_id, *(inPc_.pc), tfPc_, tf_buffer_);
#if 0  // use the latest transform available, should usually work fine
            pcl_ros::transformPointCloud(inPc_.pc->header.frame_id,
                                         ros::Time(0), *(inPc_.pc),
                                         config_.frame_id,
                                         tfPc_, tf_buffer_);
#endif
    } catch (tf2::TransformException & ex) {
      // only log tf error once every 100 times
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000 /* ms */, "%s", ex.what());
      continue;  // skip this packet
    }

    // append transformed packet data to end of output message
    outMsg->points.insert(outMsg->points.end(), tfPc_.points.begin(), tfPc_.points.end());
    outMsg->width += tfPc_.points.size();
  }

  // publish the accumulated cloud message
  RCLCPP_DEBUG_STREAM(this->get_logger(), 
    "Publishing " << outMsg->height * outMsg->width
                  << " Velodyne points, time: " << outMsg->header.stamp);
  sensor_msgs::msg::PointCloud2 ros_pc_msg;
  pcl::toROSMsg(*outMsg, ros_pc_msg);
  output_->publish(ros_pc_msg);
}

}  // namespace velodyne_pointcloud

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(velodyne_pointcloud::Transform)

#include <boost/predef/other/endian.h>
#include <pcl_conversions/pcl_conversions.h>

#include <velodyne_pointcloud/output_builder.h>

namespace velodyne_pointcloud {

OutputBuilder::OutputBuilder(size_t output_max_points_num, const VelodyneScan & scan_msg,
    bool activate_xyziradt, bool activate_xyzir) : xyziradt_activated_(activate_xyziradt), xyzir_activated_(activate_xyzir) {
  pcl::for_each_type<typename pcl::traits::fieldList<PointXYZIRADT>::type>(
      pcl::detail::FieldAdder<PointXYZIRADT>(xyziradt_fields_));

  pcl::for_each_type<typename pcl::traits::fieldList<PointXYZIR>::type>(
      pcl::detail::FieldAdder<PointXYZIR>(xyzir_fields_));

  output_xyziradt_ = std::make_unique<sensor_msgs::msg::PointCloud2>();
  output_xyzir_ = std::make_unique<sensor_msgs::msg::PointCloud2>();

  if (xyziradt_activated_) {
    init_output_msg<PointXYZIRADT>(*output_xyziradt_, output_max_points_num, scan_msg);

    offsets_xyziradt_.x_offset = output_xyziradt_->fields[pcl::getFieldIndex(*output_xyziradt_, "x")].offset;
    offsets_xyziradt_.y_offset = output_xyziradt_->fields[pcl::getFieldIndex(*output_xyziradt_, "y")].offset;
    offsets_xyziradt_.z_offset = output_xyziradt_->fields[pcl::getFieldIndex(*output_xyziradt_, "z")].offset;
    offsets_xyziradt_.intensity_offset = output_xyziradt_->fields[pcl::getFieldIndex(*output_xyziradt_, "intensity")].offset;
    offsets_xyziradt_.ring_offset = output_xyziradt_->fields[pcl::getFieldIndex(*output_xyziradt_, "ring")].offset;
    offsets_xyziradt_.azimuth_offset = output_xyziradt_->fields[pcl::getFieldIndex(*output_xyziradt_, "azimuth")].offset;
    offsets_xyziradt_.distance_offset = output_xyziradt_->fields[pcl::getFieldIndex(*output_xyziradt_, "distance")].offset;
    offsets_xyziradt_.return_type_offset = output_xyziradt_->fields[pcl::getFieldIndex(*output_xyziradt_, "return_type")].offset;
    offsets_xyziradt_.time_stamp_offset = output_xyziradt_->fields[pcl::getFieldIndex(*output_xyziradt_, "time_stamp")].offset;
  }

  if (xyzir_activated_) {
    init_output_msg<PointXYZIR>(*output_xyzir_, output_max_points_num, scan_msg);

    offsets_xyzir_.x_offset = output_xyzir_->fields[pcl::getFieldIndex(*output_xyzir_, "x")].offset;
    offsets_xyzir_.y_offset = output_xyzir_->fields[pcl::getFieldIndex(*output_xyzir_, "y")].offset;
    offsets_xyzir_.z_offset = output_xyzir_->fields[pcl::getFieldIndex(*output_xyzir_, "z")].offset;
    offsets_xyzir_.intensity_offset = output_xyzir_->fields[pcl::getFieldIndex(*output_xyzir_, "intensity")].offset;
    offsets_xyzir_.ring_offset = output_xyzir_->fields[pcl::getFieldIndex(*output_xyzir_, "ring")].offset;
  }
}

void OutputBuilder::set_extract_range(double min_range, double max_range) {
  min_range_ = min_range;
  max_range_ = max_range;
}

bool OutputBuilder::xyziradt_is_activated() {
  return xyziradt_activated_;
}

bool OutputBuilder::xyzir_is_activated() {
  return xyzir_activated_;
}

std::unique_ptr<sensor_msgs::msg::PointCloud2> OutputBuilder::move_xyziradt_output() {
  if (!xyziradt_activated_ || output_xyziradt_moved_) {
    return std::unique_ptr<sensor_msgs::msg::PointCloud2>(nullptr);
  }

  output_xyziradt_->data.resize(output_xyziradt_data_size_);

  double time_stamp = *reinterpret_cast<double *>(&output_xyziradt_->data[offsets_xyziradt_.time_stamp_offset]);
  auto stamp = rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(time_stamp)).count());
  output_xyziradt_->header.stamp = stamp;

  output_xyziradt_moved_ = true;
  return std::move(output_xyziradt_);
}

std::unique_ptr<sensor_msgs::msg::PointCloud2> OutputBuilder::move_xyzir_output() {
  if (!xyzir_activated_ || output_xyzir_moved_) {
    return std::unique_ptr<sensor_msgs::msg::PointCloud2>(nullptr);
  }

  output_xyzir_->data.resize(output_xyzir_data_size_);
  auto stamp = rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(first_timestamp_)).count());
  output_xyzir_->header.stamp = stamp;

  output_xyzir_moved_ = true;
  return std::move(output_xyzir_);
}

void OutputBuilder::addPoint(
    const float & x, const float & y, const float & z,
    const uint8_t & return_type, const uint16_t & ring, const uint16_t & azimuth,
    const float & distance, const float & intensity, const double & time_stamp) {
  // Needed for velodyne_convert_node logic.
  last_azimuth = azimuth;

  if (xyziradt_activated_ && !output_xyziradt_moved_ &&
      min_range_ <= distance && distance <= max_range_) {

    /* Slightly slower
    auto &msg = *output_xyziradt_;
    auto &sz = output_xyziradt_data_size_;
    *reinterpret_cast<float *>(&msg.data[sz + offsets_xyziradt_.x_offset]) = x;
    *reinterpret_cast<float *>(&msg.data[sz + offsets_xyziradt_.y_offset]) = y;
    *reinterpret_cast<float *>(&msg.data[sz + offsets_xyziradt_.z_offset]) = z;
    *reinterpret_cast<float *>(&msg.data[sz + offsets_xyziradt_.intensity_offset]) = intensity;
    *reinterpret_cast<uint16_t *>(&msg.data[sz + offsets_xyziradt_.ring_offset]) = ring;
    *reinterpret_cast<float *>(&msg.data[sz + offsets_xyziradt_.azimuth_offset]) = azimuth;
    *reinterpret_cast<float *>(&msg.data[sz + offsets_xyziradt_.distance_offset]) = distance;
    *reinterpret_cast<uint8_t *>(&msg.data[sz + offsets_xyziradt_.return_type_offset]) = return_type;
    *reinterpret_cast<double *>(&msg.data[sz + offsets_xyziradt_.time_stamp_offset]) = time_stamp;
    */

    PointXYZIRADT *point = (PointXYZIRADT *) &output_xyziradt_->data[output_xyziradt_data_size_];
    point->x = x;
    point->y = y;
    point->z = z;
    point->intensity = intensity;
    point->ring = ring;
    point->azimuth = azimuth;
    point->distance = distance;
    point->return_type = return_type;
    point->time_stamp = time_stamp;

    output_xyziradt_data_size_ += output_xyziradt_->point_step;
    output_xyziradt_->width++;
    output_xyziradt_->row_step += output_xyziradt_->point_step;
  }

  if (xyzir_activated_ && !output_xyzir_moved_ &&
      min_range_ <= distance && distance <= max_range_) {
    if (first_timestamp_ == 0) first_timestamp_ = time_stamp;

    /* Slightly slower
    auto &msg = *output_xyzir_;
    auto &sz = output_xyzir_data_size_;
    *reinterpret_cast<float *>(&msg.data[sz + offsets_xyzir_.x_offset]) = x;
    *reinterpret_cast<float *>(&msg.data[sz + offsets_xyzir_.y_offset]) = y;
    *reinterpret_cast<float *>(&msg.data[sz + offsets_xyzir_.z_offset]) = z;
    *reinterpret_cast<float *>(&msg.data[sz + offsets_xyzir_.intensity_offset]) = intensity;
    *reinterpret_cast<uint16_t *>(&msg.data[sz + offsets_xyzir_.ring_offset]) = ring;
    */

    PointXYZIR *point = (PointXYZIR *) &output_xyzir_->data[output_xyzir_data_size_];
    point->x = x;
    point->y = y;
    point->z = z;
    point->intensity = intensity;
    point->ring = ring;

    output_xyzir_data_size_ += output_xyzir_->point_step;
    output_xyzir_->width++;
    output_xyzir_->row_step += output_xyzir_->point_step;
  }
}

} // namespace velodyne_pointcloud

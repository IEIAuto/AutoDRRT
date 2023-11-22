#pragma once

#include <cfloat>
#include <memory>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <velodyne_pointcloud/datacontainerbase.h>
#include <velodyne_pointcloud/pointcloudXYZIRADT.h>
#include <velodyne_pointcloud/pointcloudXYZIR.h>
#include <velodyne_msgs/msg/velodyne_scan.hpp>

namespace velodyne_pointcloud {

class OutputBuilder : public velodyne_rawdata::DataContainerBase {
  using PointXYZIRADT = velodyne_pointcloud::PointXYZIRADT;
  using PointXYZIR = velodyne_pointcloud::PointXYZIR;
  using VelodyneScan = velodyne_msgs::msg::VelodyneScan;

  std::unique_ptr<sensor_msgs::msg::PointCloud2> output_xyziradt_;
  size_t output_xyziradt_data_size_ = 0;
  bool output_xyziradt_moved_ = false;
  bool xyziradt_activated_ = false;

  std::unique_ptr<sensor_msgs::msg::PointCloud2> output_xyzir_;
  size_t output_xyzir_data_size_ = 0;
  bool output_xyzir_moved_ = false;
  bool xyzir_activated_ = false;

  double min_range_ = 0;
  double max_range_ = DBL_MAX;

  struct OffsetsXYZIRADT {
    uint32_t x_offset;
    uint32_t y_offset;
    uint32_t z_offset;
    uint32_t intensity_offset;
    uint32_t ring_offset;
    uint32_t azimuth_offset;
    uint32_t distance_offset;
    uint32_t return_type_offset;
    uint32_t time_stamp_offset;
  } offsets_xyziradt_;

  struct OffsetsXYZIR {
    uint32_t x_offset;
    uint32_t y_offset;
    uint32_t z_offset;
    uint32_t intensity_offset;
    uint32_t ring_offset;
  } offsets_xyzir_;

  std::vector<pcl::PCLPointField> xyziradt_fields_;
  std::vector<pcl::PCLPointField> xyzir_fields_;

  // Only used for PointXYZIR
  double first_timestamp_ = 0;

  template <class PointT>
  void init_output_msg(sensor_msgs::msg::PointCloud2 & msg, size_t output_max_points_num,
      const velodyne_msgs::msg::VelodyneScan & scan_msg) {
    msg.data.resize(output_max_points_num * sizeof(PointT));

    msg.header = scan_msg.header;
    // To be updated when a first point is added
    msg.header.stamp = scan_msg.packets[0].stamp;

    msg.height = 1;
    msg.width = 0;

    if (std::is_same<PointT, PointXYZIRADT>::value) pcl_conversions::fromPCL(xyziradt_fields_, msg.fields);
    if (std::is_same<PointT, PointXYZIR>::value) pcl_conversions::fromPCL(xyzir_fields_, msg.fields);

    // https://github.com/PointCloudLibrary/pcl/blob/b551ee47b24e90cfb430ee4474f569e1411cd7bc/common/include/pcl/PCLPointCloud2.h#L25-L26
    static_assert(BOOST_ENDIAN_BIG_BYTE || BOOST_ENDIAN_LITTLE_BYTE, "unable to determine system endianness");
    msg.is_bigendian = BOOST_ENDIAN_BIG_BYTE;

    msg.point_step = sizeof(PointT);
    msg.row_step = sizeof(PointT) * msg.width;
    msg.is_dense = true;
  }

public:
  // Needed for velodyne_convert_node logic
  uint16_t last_azimuth;

  OutputBuilder(size_t output_max_points_num, const VelodyneScan & scan_msg, bool activate_xyziradt, bool activate_xyzir);

  void set_extract_range(double min_range, double max_range);

  bool xyziradt_is_activated();
  bool xyzir_is_activated();

  std::unique_ptr<sensor_msgs::msg::PointCloud2> move_xyziradt_output();
  std::unique_ptr<sensor_msgs::msg::PointCloud2> move_xyzir_output();

  virtual void addPoint(
    const float & x, const float & y, const float & z,
    const uint8_t & return_type, const uint16_t & ring, const uint16_t & azimuth,
    const float & distance, const float & intensity,
    const double & time_stamp) override;
};

} // namespace velodyne_pointcloud


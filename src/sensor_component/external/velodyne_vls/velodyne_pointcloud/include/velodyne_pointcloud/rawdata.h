/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
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

/* -*- mode: C++ -*-
 *
 *  Copyright (C) 2007 Austin Robot Technology, Yaxin Liu, Patrick Beeson
 *  Copyright (C) 2009, 2010, 2012 Austin Robot Technology, Jack O'Quin
 *
 *  License: Modified BSD Software License Agreement
 *
 *  $Id$
 */

/** @file
 *
 *  @brief Interfaces for interpreting raw packets from the Velodyne 3D LIDAR.
 *
 *  @author Yaxin Liu
 *  @author Patrick Beeson
 *  @author Jack O'Quin
 */

#ifndef __VELODYNE_RAWDATA_H
#define __VELODYNE_RAWDATA_H

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <boost/format.hpp>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>

#include <rclcpp/rclcpp.hpp>
#include <velodyne_msgs/msg/velodyne_packet.hpp>

#include <velodyne_pointcloud/calibration.h>
#include <velodyne_pointcloud/point_types.h>

#include <velodyne_pointcloud/datacontainerbase.h>

namespace velodyne_rawdata
{
// Shorthand typedefs for point cloud representations
typedef velodyne_pointcloud::PointXYZIR VPoint;
typedef pcl::PointCloud<VPoint> VPointCloud;

/**
 * Raw Velodyne packet constants and structures.
 */
static const int SIZE_BLOCK = 100;
static const int RAW_SCAN_SIZE = 3;
static const int SCANS_PER_BLOCK = 32;
static const int BLOCK_DATA_SIZE = (SCANS_PER_BLOCK * RAW_SCAN_SIZE);

static const float ROTATION_RESOLUTION = 0.01f;     // [deg]
static const uint16_t ROTATION_MAX_UNITS = 36000u;  // [deg/100]

/** @todo make this work for both big and little-endian machines */
static const uint16_t UPPER_BANK = 0xeeff;
static const uint16_t LOWER_BANK = 0xddff;

/** Return Modes **/
static const uint16_t RETURN_MODE_STRONGEST = 55;
static const uint16_t RETURN_MODE_LAST = 56;
static const uint16_t RETURN_MODE_DUAL = 57;

/** Special Defines for VLP16 support **/
static const int VLP16_FIRINGS_PER_BLOCK = 2;
static const int VLP16_SCANS_PER_FIRING = 16;
static const float VLP16_BLOCK_TDURATION = 110.592f;  // [µs]
static const float VLP16_DSR_TOFFSET = 2.304f;        // [µs]
static const float VLP16_FIRING_TOFFSET = 55.296f;    // [µs]

/** Special Definitions for VLS128 support **/
static const float VLP128_DISTANCE_RESOLUTION   =    0.004f;  // [m]

/** Special Definitions for VLS128 support **/
// These are used to detect which bank of 32 lasers is in this block
static const uint16_t VLS128_BANK_1 = 0xeeff;
static const uint16_t VLS128_BANK_2 = 0xddff;
static const uint16_t VLS128_BANK_3 = 0xccff;
static const uint16_t VLS128_BANK_4 = 0xbbff;

static const float  VLS128_CHANNEL_TDURATION  =  2.665f;  // [µs] Channels corresponds to one laser firing
static const float  VLS128_SEQ_TDURATION      =  53.3f;   // [µs] Sequence is a set of laser firings including recharging

/** \brief Raw Velodyne data block.
 *
 *  Each block contains data from either the upper or lower laser
 *  bank.  The device returns three times as many upper bank blocks.
 *
 *  use stdint.h types, so things work with both 64 and 32-bit machines
 */
typedef struct raw_block
{
  uint16_t header;    ///< UPPER_BANK or LOWER_BANK
  uint16_t rotation;  ///< 0-35999, divide by 100 to get degrees
  uint8_t data[BLOCK_DATA_SIZE];
} raw_block_t;

/** used for unpacking the first two data bytes in a block
 *
 *  They are packed into the actual data stream misaligned.  I doubt
 *  this works on big endian machines.
 */
union two_bytes {
  uint16_t uint;
  uint8_t bytes[2];
};

static const int PACKET_SIZE = 1206;
static const int BLOCKS_PER_PACKET = 12;
static const int PACKET_STATUS_SIZE = 4;
static const int SCANS_PER_PACKET = (SCANS_PER_BLOCK * BLOCKS_PER_PACKET);

/** \brief Raw Velodyne packet.
 *
 *  revolution is described in the device manual as incrementing
 *    (mod 65536) for each physical turn of the device.  Our device
 *    seems to alternate between two different values every third
 *    packet.  One value increases, the other decreases.
 *
 *  \todo figure out if revolution is only present for one of the
 *  two types of status fields
 *
 *  status has either a temperature encoding or the microcode level
 */
typedef struct raw_packet
{
  raw_block_t blocks[BLOCKS_PER_PACKET];
  uint16_t revolution;
  uint8_t status[PACKET_STATUS_SIZE];
} raw_packet_t;

/** \brief Velodyne echo types */
enum RETURN_TYPE
{
  INVALID = 0,
  SINGLE_STRONGEST = 1,
  SINGLE_LAST = 2,
  DUAL_STRONGEST_FIRST = 3,
  DUAL_STRONGEST_LAST = 4,
  DUAL_WEAK_FIRST = 5,
  DUAL_WEAK_LAST = 6,
  DUAL_ONLY = 7
};

/** \brief Velodyne data conversion class */
class RawData
{
public:
  RawData(rclcpp::Node * node_ptr);
  ~RawData() {}

  /** \brief Set up for data processing.
   *
   *  Perform initializations needed before data processing can
   *  begin:
   *
   *    - read device-specific angles calibration
   *
   *  @returns 0 if successful;
   *           errno value for failure
   */
  int setup();

  /** \brief Set up for data processing offline.
   * Performs the same initialization as in setup, in the abscence of a ros::NodeHandle.
   * this method is useful if unpacking data directly from bag files, without passing
   * through a communication overhead.
   *
   * @param calibration_file path to the calibration file
   * @param max_range_ cutoff for maximum range
   * @param min_range_ cutoff for minimum range
   * @returns 0 if successful;
   *           errno value for failure
   */
  int setupOffline(std::string calibration_file, double max_range_, double min_range_);

  void unpack(const velodyne_msgs::msg::VelodynePacket & pkt, DataContainerBase & data);

  void setParameters(double min_range, double max_range, double view_direction, double view_width);

  int scansPerPacket() const;
  int getNumLasers() const;
  double getMaxRange() const;
  double getMinRange() const;

private:
  /** configuration parameters */
  typedef struct
  {
    std::string calibrationFile;  ///< calibration file name
    double max_range;             ///< maximum range to publish
    double min_range;             ///< minimum range to publish
    int min_angle;                ///< minimum angle to publish
    int max_angle;                ///< maximum angle to publish

    double tmp_min_angle;
    double tmp_max_angle;
  } Config;
  Config config_;

  // ROS node hanlder
  rclcpp::Node * node_ptr_;

  /**
   * Calibration file
   */
  velodyne_pointcloud::Calibration calibration_;
  float sin_rot_table_[ROTATION_MAX_UNITS];
  float cos_rot_table_[ROTATION_MAX_UNITS];

  // Caches the azimuth percent offset for the VLS-128 laser firings
  float vls_128_laser_azimuth_cache[16];

  /** add private function to handle the VLP16 **/
  void unpack_vlp16(const velodyne_msgs::msg::VelodynePacket & pkt, DataContainerBase & data);

  /** add private function to handle the VLS128 **/
  void unpack_vls128(const velodyne_msgs::msg::VelodynePacket &pkt, DataContainerBase &data);

  /** in-line test whether a point is in range */
  bool pointInRange(float range)
  {
    return (range >= config_.min_range && range <= config_.max_range);
  }
};

}  // namespace velodyne_rawdata

#endif  // __VELODYNE_RAWDATA_H

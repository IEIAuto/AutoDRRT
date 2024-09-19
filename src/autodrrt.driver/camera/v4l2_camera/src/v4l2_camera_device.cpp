// Copyright 2019 Bold Hearts
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "v4l2_camera/v4l2_camera_device.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <utility>
#include <memory>
#include <fstream>

#include "v4l2_camera/fourcc.hpp"

using v4l2_camera::V4l2CameraDevice;
using sensor_msgs::msg::Image;

V4l2CameraDevice::V4l2CameraDevice(std::string device, bool use_v4l2_buffer_timestamps, rclcpp::Duration timestamp_offset_duration)
: device_{std::move(device)}, use_v4l2_buffer_timestamps_{use_v4l2_buffer_timestamps}, timestamp_offset_{timestamp_offset_duration}
{
}

bool V4l2CameraDevice::open()
{
  // Check if TSC offset applies
  setTSCOffset();
  
  fd_ = ::open(device_.c_str(), O_RDWR);

  if (fd_ < 0) {
    auto msg = std::ostringstream{};
    msg << "Failed opening device " << device_ << ": " << strerror(errno) << " (" << errno << ")";
    RCLCPP_ERROR(rclcpp::get_logger("v4l2_camera"), "%s", msg.str().c_str());
    return false;
  }

  // List capabilities
  ioctl(fd_, VIDIOC_QUERYCAP, &capabilities_);

  auto canRead = capabilities_.capabilities & V4L2_CAP_READWRITE;
  auto canStream = capabilities_.capabilities & V4L2_CAP_STREAMING;

  RCLCPP_INFO(
    rclcpp::get_logger("v4l2_camera"),
    "Driver: %s", capabilities_.driver);
  RCLCPP_INFO(
    rclcpp::get_logger("v4l2_camera"),
    "Version: %s", std::to_string(capabilities_.version).c_str());
  RCLCPP_INFO(
    rclcpp::get_logger("v4l2_camera"),
    "Device: %s", capabilities_.card);
  RCLCPP_INFO(
    rclcpp::get_logger("v4l2_camera"),
    "Location: %s", capabilities_.bus_info);

  RCLCPP_INFO(
    rclcpp::get_logger("v4l2_camera"),
    "Capabilities:");
  RCLCPP_INFO(
    rclcpp::get_logger("v4l2_camera"),
    "  Read/write: %s", (canRead ? "YES" : "NO"));
  RCLCPP_INFO(
    rclcpp::get_logger("v4l2_camera"),
    "  Streaming: %s", (canStream ? "YES" : "NO"));

  // Get current data (pixel) format
  auto formatReq = v4l2_format{};
  formatReq.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  ioctl(fd_, VIDIOC_G_FMT, &formatReq);
  cur_data_format_ = PixelFormat{formatReq.fmt.pix};

  RCLCPP_INFO(
    rclcpp::get_logger("v4l2_camera"),
    "Current pixel format: %s @ %sx%s", FourCC::toString(cur_data_format_.pixelFormat).c_str(),
    std::to_string(cur_data_format_.width).c_str(),
    std::to_string(cur_data_format_.height).c_str());

  // List all available image formats, sizes, intervals, controls and capture parameters
  getCaptureParameters();
  listImageFormats();
  listImageSizes();
  listFrameIntervals();
  listControls();

  // Log results
  RCLCPP_INFO(rclcpp::get_logger("v4l2_camera"), "Available pixel formats: ");
  for (auto const & format : image_formats_) {
    RCLCPP_INFO(
      rclcpp::get_logger("v4l2_camera"),
      "  %s - %s", FourCC::toString(format.pixelFormat).c_str(), format.description.c_str());
  }

  RCLCPP_INFO(rclcpp::get_logger("v4l2_camera"), "Available controls: ");
  for (auto const & control : controls_) {
    RCLCPP_INFO(
      rclcpp::get_logger("v4l2_camera"),
      "  %s (%s) = %s", control.name.c_str(),
      std::to_string(static_cast<unsigned>(control.type)).c_str(),
      std::to_string(getControlValue(control.id)).c_str());
  }

  if (timePerFrameSupported()) {
    RCLCPP_INFO(rclcpp::get_logger("v4l2_camera"), "Time-per-frame support: YES");
    RCLCPP_INFO(
      rclcpp::get_logger("v4l2_camera"),
      "  Current time per frame: %d/%d s",
      capture_parm_.timeperframe.numerator,
      capture_parm_.timeperframe.denominator);

    RCLCPP_INFO(rclcpp::get_logger("v4l2_camera"), "  Available intervals:");
    for (auto const & kv : frame_intervals_) {
      auto oss = std::ostringstream{};
      oss << FourCC::toString(std::get<0>(kv.first)) << " " << std::get<1>(kv.first) << "x" <<
        std::get<2>(kv.first) << ":";
      for (auto const & i : kv.second) {
        oss << " " << i.first << "/" << i.second;
      }
      RCLCPP_INFO(rclcpp::get_logger("v4l2_camera"), "    %s", oss.str().c_str());
    }
  } else {
    RCLCPP_INFO(
      rclcpp::get_logger("v4l2_camera"), "Time-per-frame support: NO");
  }

  return true;
}

bool V4l2CameraDevice::start()
{
  RCLCPP_INFO(rclcpp::get_logger("v4l2_camera"), "Starting camera");
  if (!initMemoryMapping()) {
    return false;
  }

  // Queue the buffers
  for (auto const & buffer : buffers_) {
    auto buf = v4l2_buffer{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = buffer.index;

    if (-1 == ioctl(fd_, VIDIOC_QBUF, &buf)) {
      RCLCPP_ERROR(
        rclcpp::get_logger("v4l2_camera"),
        "Buffer failure on capture start: %s (%s)", strerror(errno),
        std::to_string(errno).c_str());
      return false;
    }
  }

  // Start stream
  unsigned type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == ioctl(fd_, VIDIOC_STREAMON, &type)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("v4l2_camera"),
      "Failed stream start: %s (%s)", strerror(errno),
      std::to_string(errno).c_str());
    return false;
  }
  return true;
}

bool V4l2CameraDevice::stop()
{
  RCLCPP_INFO(rclcpp::get_logger("v4l2_camera"), "Stopping camera");
  // Stop stream
  unsigned type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == ioctl(fd_, VIDIOC_STREAMOFF, &type)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("v4l2_camera"),
      "Failed stream stop");
    return false;
  }

  // De-initialize buffers
  for (auto const & buffer : buffers_) {
    munmap(buffer.start, buffer.length);
  }

  buffers_.clear();

  auto req = v4l2_requestbuffers{};

  // Free all buffers
  req.count = 0;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  ioctl(fd_, VIDIOC_REQBUFS, &req);

  return true;
}

std::string V4l2CameraDevice::getCameraName()
{
  auto name = std::string{reinterpret_cast<char *>(capabilities_.card)};
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);
  std::replace(name.begin(), name.end(), ' ', '_');
  return name;
}

int64_t V4l2CameraDevice::getTimeOffset()
{
  timespec system_sample, monotonic_sample;
  clock_gettime(CLOCK_REALTIME, &system_sample);
  clock_gettime(CLOCK_MONOTONIC_RAW, &monotonic_sample);
  return (static_cast<int64_t>(system_sample.tv_sec * 1e9) - static_cast<int64_t>(monotonic_sample.tv_sec * 1e9)
          + static_cast<int64_t>(system_sample.tv_nsec) - static_cast<int64_t>(monotonic_sample.tv_nsec));
}

void V4l2CameraDevice::setTSCOffset()
{
  std::ifstream offset_ns_file("/sys/devices/system/clocksource/clocksource0/offset_ns");
  if (offset_ns_file.good()) {
    std::string offset;
    offset_ns_file >> offset;
    offset_ns_file.close();
    tsc_offset_ = std::stoull(offset);
  }
  else {
    tsc_offset_ = 0;
  }
}

Image::UniquePtr V4l2CameraDevice::capture()
{
  auto buf = v4l2_buffer{};
  rclcpp::Time buf_stamp;

  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;

  // Dequeue buffer with new image
  if (-1 == ioctl(fd_, VIDIOC_DQBUF, &buf)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("v4l2_camera"),
      "Error dequeueing buffer: %s (%s)", strerror(errno),
      std::to_string(errno).c_str());
    return nullptr;
  }

  if (use_v4l2_buffer_timestamps_) {
    buf_stamp = rclcpp::Time(static_cast<int64_t>(buf.timestamp.tv_sec) * 1e9
                             + static_cast<int64_t>(buf.timestamp.tv_usec) * 1e3 
                             + getTimeOffset() - tsc_offset_);
  
  }
  else {
    buf_stamp = rclcpp::Clock{RCL_SYSTEM_TIME}.now();
  }
  buf_stamp = buf_stamp + timestamp_offset_;

  // Create image object
  auto img = std::make_unique<Image>();

  // Copy over buffer data
  auto const & buffer = buffers_[buf.index];
  img->data.resize(cur_data_format_.imageByteSize);
  std::copy(buffer.start, buffer.start + img->data.size(), img->data.begin());

  // Requeue buffer to be reused for new captures
  if (-1 == ioctl(fd_, VIDIOC_QBUF, &buf)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("v4l2_camera"),
      "Error re-queueing buffer: %s (%s)", strerror(errno),
      std::to_string(errno).c_str());
    return nullptr;
  }

  // Fill in remaining image information
  img->header.stamp = buf_stamp;
  img->width = cur_data_format_.width;
  img->height = cur_data_format_.height;
  img->step = cur_data_format_.bytesPerLine;
  if (cur_data_format_.pixelFormat == V4L2_PIX_FMT_YUYV) {
    img->encoding = sensor_msgs::image_encodings::YUV422_YUY2;
  } else if (cur_data_format_.pixelFormat == V4L2_PIX_FMT_UYVY) {
    img->encoding = sensor_msgs::image_encodings::YUV422;
  } else if (cur_data_format_.pixelFormat == V4L2_PIX_FMT_GREY) {
    img->encoding = sensor_msgs::image_encodings::MONO8;
  } else {
    RCLCPP_WARN(rclcpp::get_logger("v4l2_camera"), "Current pixel format is not supported yet");
  }

  return img;
}

int32_t V4l2CameraDevice::getControlValue(uint32_t id)
{
  auto ctrl = v4l2_control{};
  ctrl.id = id;
  if (-1 == ioctl(fd_, VIDIOC_G_CTRL, &ctrl)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("v4l2_camera"),
      "Failed getting value for control %s: %s (%s); returning 0!", std::to_string(id).c_str(),
      strerror(errno), std::to_string(errno).c_str());
    return 0;
  }
  return ctrl.value;
}

bool V4l2CameraDevice::setControlValue(uint32_t id, int32_t value)
{
  auto ctrl = v4l2_control{};
  ctrl.id = id;
  ctrl.value = value;
  if (-1 == ioctl(fd_, VIDIOC_S_CTRL, &ctrl)) {
    auto control = std::find_if(
      controls_.begin(), controls_.end(),
      [id](Control const & c) {return c.id == id;});
    RCLCPP_ERROR(
      rclcpp::get_logger("v4l2_camera"),
      "Failed setting value for control %s to %s: %s (%s)", control->name.c_str(),
      std::to_string(value).c_str(), strerror(errno), std::to_string(errno).c_str());
    return false;
  }
  return true;
}

bool V4l2CameraDevice::requestDataFormat(const PixelFormat & format)
{
  auto formatReq = v4l2_format{};
  formatReq.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  formatReq.fmt.pix.pixelformat = format.pixelFormat;
  formatReq.fmt.pix.width = format.width;
  formatReq.fmt.pix.height = format.height;

  RCLCPP_INFO(
    rclcpp::get_logger("v4l2_camera"),
    "Requesting format: %sx%s", std::to_string(format.width).c_str(),
    std::to_string(format.height).c_str());

  // Perform request
  if (-1 == ioctl(fd_, VIDIOC_S_FMT, &formatReq)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("v4l2_camera"),
      "Failed requesting pixel format: %s (%s)", strerror(errno),
      std::to_string(errno).c_str());
    return false;
  }

  RCLCPP_INFO(rclcpp::get_logger("v4l2_camera"), "Success");
  cur_data_format_ = PixelFormat{formatReq.fmt.pix};
  return true;
}

bool V4l2CameraDevice::requestTimePerFrame(std::pair<uint32_t, uint32_t> tpf)
{
  if (!timePerFrameSupported()) {
    RCLCPP_ERROR(
      rclcpp::get_logger("v4l2_camera"),
      "Device does not support setting time per frame");
    return false;
  }

  auto parmReq = v4l2_streamparm{};
  parmReq.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  parmReq.parm.capture = capture_parm_;
  parmReq.parm.capture.timeperframe.numerator = tpf.first;
  parmReq.parm.capture.timeperframe.denominator = tpf.second;

  // Perform request
  if (-1 == ioctl(fd_, VIDIOC_S_PARM, &parmReq)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("v4l2_camera"),
      "Failed requesting time per frame: %s (%s)", strerror(errno),
      std::to_string(errno).c_str());
    return false;
  }

  if (parmReq.parm.capture.timeperframe.numerator != tpf.first ||
    parmReq.parm.capture.timeperframe.denominator != tpf.second)
  {
    RCLCPP_WARN(
      rclcpp::get_logger("v4l2_camera"),
      "Requesting time per frame succeeded, but driver overwrote value: %d/%d",
      parmReq.parm.capture.timeperframe.numerator,
      parmReq.parm.capture.timeperframe.denominator);
    return false;
  }

  return true;
}

void V4l2CameraDevice::getCaptureParameters()
{
  struct v4l2_streamparm streamParm;
  streamParm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == ioctl(fd_, VIDIOC_G_PARM, &streamParm)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("v4l2_camera"),
      "Failed requesting streaming parameters: %s (%s)", strerror(errno),
      std::to_string(errno).c_str());
    return;
  }

  capture_parm_ = streamParm.parm.capture;
}

void V4l2CameraDevice::listImageFormats()
{
  image_formats_.clear();

  struct v4l2_fmtdesc fmtDesc;
  fmtDesc.index = 0;
  fmtDesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  while (ioctl(fd_, VIDIOC_ENUM_FMT, &fmtDesc) == 0) {
    image_formats_.emplace_back(fmtDesc);
    fmtDesc.index++;
  }
}

void V4l2CameraDevice::listImageSizes()
{
  image_sizes_.clear();
  struct v4l2_frmsizeenum frmSizeEnum;
  // Supported sizes can be different per format
  for (auto const & f : image_formats_) {
    frmSizeEnum.index = 0;
    frmSizeEnum.pixel_format = f.pixelFormat;

    if (-1 == ioctl(fd_, VIDIOC_ENUM_FRAMESIZES, &frmSizeEnum)) {
      RCLCPP_ERROR_STREAM(
        rclcpp::get_logger("v4l2_camera"),
        "Failed listing frame size " << strerror(errno) << " (" << errno << ")");
      continue;
    }

    switch (frmSizeEnum.type) {
      case V4L2_FRMSIZE_TYPE_DISCRETE:
        image_sizes_[f.pixelFormat] = listDiscreteImageSizes(frmSizeEnum);
        break;
      case V4L2_FRMSIZE_TYPE_STEPWISE:
        image_sizes_[f.pixelFormat] = listStepwiseImageSizes(frmSizeEnum);
        break;
      case V4L2_FRMSIZE_TYPE_CONTINUOUS:
        image_sizes_[f.pixelFormat] = listContinuousImageSizes(frmSizeEnum);
        break;
      default:
        RCLCPP_WARN_STREAM(
          rclcpp::get_logger("v4l2_camera"),
          "Frame size type not supported: " << frmSizeEnum.type);
        continue;
    }
  }
}

V4l2CameraDevice::ImageSizesDescription V4l2CameraDevice::listDiscreteImageSizes(
  v4l2_frmsizeenum frm_size_enum)
{
  auto sizes = ImageSizesVector{};

  do {
    sizes.emplace_back(std::make_pair(frm_size_enum.discrete.width, frm_size_enum.discrete.height));
    frm_size_enum.index++;
  } while (ioctl(fd_, VIDIOC_ENUM_FRAMESIZES, &frm_size_enum) == 0);

  return std::make_pair(ImageSizeType::DISCRETE, std::move(sizes));
}

V4l2CameraDevice::ImageSizesDescription V4l2CameraDevice::listStepwiseImageSizes(
  v4l2_frmsizeenum frm_size_enum)
{
  // Three entries: min size, max size and stepsize
  auto sizes = ImageSizesVector(3);
  sizes[0] = std::make_pair(frm_size_enum.stepwise.min_width, frm_size_enum.stepwise.min_height);
  sizes[1] = std::make_pair(frm_size_enum.stepwise.max_width, frm_size_enum.stepwise.max_height);
  sizes[2] = std::make_pair(frm_size_enum.stepwise.step_width, frm_size_enum.stepwise.step_height);

  return std::make_pair(ImageSizeType::STEPWISE, std::move(sizes));
}

V4l2CameraDevice::ImageSizesDescription V4l2CameraDevice::listContinuousImageSizes(
  v4l2_frmsizeenum frm_size_enum)
{
  // Two entries: min size and max size, stepsize is implicitly 1
  auto sizes = ImageSizesVector(2);
  sizes[0] = std::make_pair(frm_size_enum.stepwise.min_width, frm_size_enum.stepwise.min_height);
  sizes[1] = std::make_pair(frm_size_enum.stepwise.max_width, frm_size_enum.stepwise.max_height);

  return std::make_pair(ImageSizeType::CONTINUOUS, std::move(sizes));
}

void V4l2CameraDevice::listFrameIntervals()
{
  if (!timePerFrameSupported()) {
    RCLCPP_WARN(
      rclcpp::get_logger(
        "v4l2_camera"), "Time per frame not supported, cannot list intervals");
    return;
  }

  frame_intervals_.clear();
  struct v4l2_frmivalenum frmIvalEnum;
  // Supported intervals can be different per format and image size
  for (auto const & f : image_formats_) {
    auto sd = image_sizes_[f.pixelFormat];
    switch (sd.first) {
      case ImageSizeType::DISCRETE:

        for (auto const & s : sd.second) {
          frmIvalEnum.index = 0;
          frmIvalEnum.pixel_format = f.pixelFormat;
          frmIvalEnum.width = s.first;
          frmIvalEnum.height = s.second;

          if (-1 == ioctl(fd_, VIDIOC_ENUM_FRAMEINTERVALS, &frmIvalEnum)) {
            RCLCPP_ERROR_STREAM(
              rclcpp::get_logger("v4l2_camera"),
              "Failed listing frame interval " << strerror(errno) << " (" << errno << ")");
            continue;
          }

          if (frmIvalEnum.type != V4L2_FRMIVAL_TYPE_DISCRETE) {
            RCLCPP_WARN(
              rclcpp::get_logger("v4l2_camera"),
              "Listing of non-discrete frame intervals is not currently supported");
            continue;
          }

          auto intervals = FrameIntervalsVector{};
          do {
            intervals.emplace_back(
              std::make_pair(
                frmIvalEnum.discrete.numerator,
                frmIvalEnum.discrete.denominator));
            frmIvalEnum.index++;
          } while (ioctl(fd_, VIDIOC_ENUM_FRAMEINTERVALS, &frmIvalEnum) == 0);

          frame_intervals_[std::make_tuple(
              f.pixelFormat, s.first,
              s.second)] = std::move(intervals);
        }
        break;

      default:
        RCLCPP_WARN(
          rclcpp::get_logger("v4l2_camera"),
          "Listing of frame intervals for non-discrete image sizes is not currently supported");
    }
  }
}

void V4l2CameraDevice::listControls()
{
  controls_.clear();

  auto queryctrl = v4l2_query_ext_ctrl{};
  queryctrl.id = V4L2_CID_USER_CLASS | V4L2_CTRL_FLAG_NEXT_CTRL;

  while (ioctl(fd_, VIDIOC_QUERY_EXT_CTRL, &queryctrl) == 0) {
    // Ignore disabled controls
    if (queryctrl.flags & V4L2_CTRL_FLAG_DISABLED) {
      continue;
    }

    auto menuItems = std::map<int, std::string>{};
    if (queryctrl.type == (unsigned)ControlType::MENU) {
      auto querymenu = v4l2_querymenu{};
      querymenu.id = queryctrl.id;

      // Query all enum values
      for (auto i = queryctrl.minimum; i <= queryctrl.maximum; i++) {
        querymenu.index = i;
        if (ioctl(fd_, VIDIOC_QUERYMENU, &querymenu) == 0) {
          menuItems[i] = (const char *)querymenu.name;
        }
      }
    }

    auto control = Control{};
    control.id = queryctrl.id;
    control.name = std::string{reinterpret_cast<char *>(queryctrl.name)};
    control.type = static_cast<ControlType>(queryctrl.type);
    control.minimum = queryctrl.minimum;
    control.maximum = queryctrl.maximum;
    control.defaultValue = queryctrl.default_value;
    control.menuItems = std::move(menuItems);

    controls_.push_back(control);

    // Get ready to query next item
    queryctrl.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
  }
}

bool V4l2CameraDevice::initMemoryMapping()
{
  auto req = v4l2_requestbuffers{};

  // Request 4 buffers
  req.count = 4;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  ioctl(fd_, VIDIOC_REQBUFS, &req);

  // Didn't get more than 1 buffer
  if (req.count < 2) {
    RCLCPP_ERROR(rclcpp::get_logger("v4l2_camera"), "Insufficient buffer memory");
    return false;
  }

  buffers_ = std::vector<Buffer>(req.count);

  for (auto i = 0u; i < req.count; ++i) {
    auto buf = v4l2_buffer{};

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;

    ioctl(fd_, VIDIOC_QUERYBUF, &buf);

    buffers_[i].index = buf.index;
    buffers_[i].length = buf.length;
    buffers_[i].start =
      static_cast<unsigned char *>(
      mmap(
        NULL /* start anywhere */,
        buf.length,
        PROT_READ | PROT_WRITE /* required */,
        MAP_SHARED /* recommended */,
        fd_, buf.m.offset));

    if (MAP_FAILED == buffers_[i].start) {
      RCLCPP_ERROR(rclcpp::get_logger("v4l2_camera"), "Failed mapping device memory");
      return false;
    }
  }

  return true;
}

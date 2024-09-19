#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cassert>
#include <map>

#include <yaml-cpp/yaml.h>
#include "bevdet.h"
#include "cpu_jpegdecoder.h"

// #include <pcl/io/pcd_io.h>
// #include <pcl/point_types.h>
// #include <pcl/visualization/cloud_viewer.h>
// #include <pcl/common/transforms.h>


// #include <ros/ros.h>
// #include <ros/package.h>
// #include <opencv2/opencv.hpp>
// #include <cv_bridge/cv_bridge.h>
// #include <sensor_msgs/Image.h>

// #include <sensor_msgs/PointCloud2.h>
// #include <pcl_conversions/pcl_conversions.h>  // msg2pcl

// #include <jsk_recognition_msgs/BoundingBoxArray.h>
// #include <jsk_recognition_msgs/BoundingBox.h>

// 消息同步
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h> // 时间接近
#include <ament_index_cpp/get_package_share_directory.hpp>


#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
/***********************************swpld added**********************************/
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

#include "sensor_msgs/msg/camera_info.hpp"
#include "bevdet.h"

#include <autoware_auto_perception_msgs/msg/detected_object_kinematics.hpp>
#include <autoware_auto_perception_msgs/msg/detected_objects.hpp>
#include <autoware_auto_perception_msgs/msg/object_classification.hpp>
#include <autoware_auto_perception_msgs/msg/shape.hpp>
#include <tier4_autoware_utils/geometry/geometry.hpp>
#include <tier4_autoware_utils/math/constants.hpp>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

typedef pcl::PointXYZI PointT;

// 打印网络信息
void Getinfo(void);
// # box转txt文件
void Boxes2Txt(const std::vector<Box> &boxes, std::string file_name, bool with_vel);
// box的坐标从ego系变到雷达系
void Egobox2Lidarbox(const std::vector<Box>& ego_boxes, 
                                        std::vector<Box> &lidar_boxes,
                                        const Eigen::Quaternion<float> &lidar2ego_rot,
                                        const Eigen::Translation3f &lidar2ego_trans);

// void Egobox2Lidarbox(const std::vector<Box>& ego_boxes, 
//                      jsk_recognition_msgs::BoundingBoxArrayPtr lidar_boxes,
//                      const Eigen::Quaternion<float> &lidar2ego_rot,
//                      const Eigen::Translation3f &lidar2ego_trans,  float score_thre);


// 添加从opencv的Mat转换到std::vector<char>的函数 读取图像 cv2data 
int cvToArr(cv::Mat img, std::vector<char> &raw_data)
{
    if (img.empty())
    {
        std::cerr << "image is empty. " << std::endl;
        return EXIT_FAILURE;
    }
    
    std::vector<u_char> raw_data_;
    cv::imencode(".jpg", img, raw_data_);
    raw_data = std::vector<char>(raw_data_.begin(), raw_data_.end());
    return EXIT_SUCCESS;
}

int cvImgToArr(std::vector<cv::Mat> &imgs, std::vector<std::vector<char>> &imgs_data)
{
    imgs_data.resize(imgs.size());

    for(size_t i = 0; i < imgs_data.size(); i++)
    {   
        if(cvToArr(imgs[i], imgs_data[i]))
        {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
 

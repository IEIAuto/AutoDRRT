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
//                                         jsk_recognition_msgs::BoundingBoxArrayPtr lidar_boxes,
//                                         const Eigen::Quaternion<float> &lidar2ego_rot,
//                                         const Eigen::Translation3f &lidar2ego_trans,  float score_thre);


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
 
// ros2函数,对TestSample做了封装,改成回调传参
class BevDetNode : public rclcpp::Node
{
private:

    std::string pkg_path_;
    YAML::Node config_;
    size_t img_N_;
    int img_w_; 
    int img_h_;
    unsigned char** uchar_images;
    // 模型配置文件路径 
    std::string model_config_;
    
    // 权重文件路径 图像部分 bev部分
    std::string imgstage_file_;
    std::string bevstage_file_;
   
    // 相机的内外配置参数
    YAML::Node camconfig_; 
    
    // 结果保存文件
    std::string output_lidarbox_;

    YAML::Node sample_;

    std::vector<std::string> imgs_file_;
    std::vector<std::string> imgs_name_;

    camsData sampleData_;
    std::shared_ptr<BEVDet> bevdet_;

    uchar* imgs_dev_ = nullptr; 

    // ros::NodeHandle n_;
    // // 发布点云和盒子
    // ros::Publisher pub_cloud_;
    // ros::Publisher pub_boxes_;
    
    // 订阅点云和图像
    float score_thre_;
    // std::string topic_cloud_;
    // std::string topic_img_f_, topic_img_fl_, topic_img_fr_;
    // std::string topic_img_b_, topic_img_bl_, topic_img_br_;

    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_sync[6];
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pcl_sub_sync;

    using approximate_policy_stream = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
    typedef message_filters::Synchronizer<approximate_policy_stream> Synchron_stream;
    std::unique_ptr<Synchron_stream> sync_stream;
    rclcpp::Publisher<autoware_auto_perception_msgs::msg::DetectedObjects>::SharedPtr objects_pub_;
    // message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_; 
    // message_filters::Subscriber<sensor_msgs::Image> sub_f_img_; 
    // message_filters::Subscriber<sensor_msgs::Image> sub_fl_img_; 
    // message_filters::Subscriber<sensor_msgs::Image> sub_fr_img_; 
    // message_filters::Subscriber<sensor_msgs::Image> sub_b_img_; 
    // message_filters::Subscriber<sensor_msgs::Image> sub_bl_img_; 
    // message_filters::Subscriber<sensor_msgs::Image> sub_br_img_; 

    // typedef message_filters::sync_policies::ApproximateTime<
    //     sensor_msgs::PointCloud2, 
    //     sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image,
    //     sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    
    // typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    //     std::shared_ptr<Sync> sync_;
    std::vector<std::string> input_topics_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> filters_;
public:
    BevDetNode();
    ~BevDetNode();
    // void callback(const sensor_msgs::ImageConstPtr& msg);
    void sync_stream_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg0, const sensor_msgs::msg::Image::ConstSharedPtr &msg1, const sensor_msgs::msg::Image::ConstSharedPtr &msg2, const sensor_msgs::msg::Image::ConstSharedPtr &msg3, const sensor_msgs::msg::Image::ConstSharedPtr &msg4, const sensor_msgs::msg::Image::ConstSharedPtr &msg5);
    void image_trans_to_uncharptr_to_gpu( const std::vector<sensor_msgs::msg::Image>  & msg_total, uchar* out_imgs);
    void async_stream_callback(const sensor_msgs::msg::Image::ConstSharedPtr & input_ptr, const std::string & topic_name);
};
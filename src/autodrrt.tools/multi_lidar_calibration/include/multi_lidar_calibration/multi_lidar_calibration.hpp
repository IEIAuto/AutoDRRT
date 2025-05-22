#ifndef MULTI_LIDAR_CALIBRATION__MULTI_LIDAR_CALIRATION_HPP_
#define MULTI_LIDAR_CALIBRATION__MULTI_LIDAR_CALIRATION_HPP_

#include <rclcpp/rclcpp.hpp>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_cloud.hpp"
#include "sensor_msgs/point_cloud_conversion.hpp"
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
// #include <pcl_ros/point_cloud.hpp>
#include <pcl_ros/transforms.hpp>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "tf2/transform_datatypes.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "tf2_ros/transform_broadcaster.h"
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Transform.h"
// #include "tf2_eigen/tf2_eigen.hpp"

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <iostream>
#include <fstream>

#define __APP_NAME__ "multi_lidar_calibration"

class MULTI_LIDAR_CALIBRATION : public rclcpp::Node
{
public:
    MULTI_LIDAR_CALIBRATION(const std::string & node_name, const rclcpp::NodeOptions & node_options);
    ~MULTI_LIDAR_CALIBRATION() = default;

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr calibrated_cloud_publisher_;
    
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

private:
    int child_topic_num_;
    std::map<std::string, std::vector<double>> transfer_map_;
    std::string points_parent_topic_str, points_child_topic_str;

    std::string initial_pose_topic_str = "/initialpose";
    std::string calibrated_points_topic_str = "/points_calibrated";

    double                              voxel_size_;
	double                              ndt_epsilon_;
	double                              ndt_step_size_;
	double                              ndt_resolution_;

	double                              initial_x_;
	double                              initial_y_;
	double                              initial_z_;
	double                              initial_roll_;
	double                              initial_pitch_;
	double                              initial_yaw_;

    int                                 ndt_iterations_;

    std::string                         parent_frame_;
	std::string                         child_frame_;

    Eigen::Matrix4f                     current_guess_;

    typedef pcl::PointXYZ PointT;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> *cloud_parent_subscriber_, *cloud_child_subscriber_;
    using SyncPolicyT = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2,sensor_msgs::msg::PointCloud2>;
    using Sync = message_filters::Synchronizer<SyncPolicyT>;
    std::shared_ptr<Sync> sync_ptr_;

    pcl::PointCloud<PointT>::Ptr in_parent_cloud_, in_child_cloud_, in_child_filtered_cloud_;

private:
    void syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &in_parent_cloud_msg,
                      const sensor_msgs::msg::PointCloud2::ConstSharedPtr &in_child_cloud_msg);
    
    void InitializeROSIo();

    void DownsampleCloud(pcl::PointCloud<PointT>::ConstPtr in_cloud_ptr, pcl::PointCloud<PointT>::Ptr out_cloud_ptr, double in_leaf_size);

    void PublishCloud(pcl::PointCloud<PointT>::ConstPtr in_cloud_to_publish_ptr);

    void MatrixToTranform(Eigen::Matrix4f & matrix, tf2::Transform & trans);

    void PerformNdtOptimize();

public: 
    void Run();

};

#endif
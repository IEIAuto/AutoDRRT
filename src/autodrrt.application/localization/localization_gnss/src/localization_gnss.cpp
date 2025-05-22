#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include <iostream>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>
#include <cstdlib>

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>

#include <autoware_sensing_msgs/msg/gnss_ins_orientation_stamped.hpp>
#include <autoware_auto_vehicle_msgs/msg/velocity_report.hpp>
#include <tf2/transform_datatypes.h>
#include <autoware_adapi_v1_msgs/msg/localization_initialization_state.hpp>

struct vehicle_state
{
  double x;
  double y;
  double z;

  double roll;
  double pitch;
  double yaw;

  double r_w, r_x, r_y, r_z;

  // linear velocity
  double lv_x, lv_y, lv_z;

  // linear acceleration
  double la_x, la_y, la_z;

  double av_x, av_y, av_z; // angular velocity
  double aa_x, aa_y, aa_z; // angular acceleration
};


class LocalizationGnss : public rclcpp::Node
{
private:
  tf2_ros::TransformBroadcaster tf2_broadcaster_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher;
  rclcpp::Publisher<geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr accel_publisher;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr kinematic_publisher;

  rclcpp::Publisher<autoware_adapi_v1_msgs::msg::LocalizationInitializationState>::SharedPtr pub_state_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_3d_publisher;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr gnss_pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<autoware_sensing_msgs::msg::GnssInsOrientationStamped>::SharedPtr orientation_sub_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::VelocityReport>::SharedPtr velocity_status_sub_;

  std::string base_link_frame_id;

  vehicle_state state;

  bool state_published = false;

public:
  LocalizationGnss() : Node("carla_dbg_svr"), tf2_broadcaster_(*this)
  {

    memset(&state, 0, sizeof state);

    base_link_frame_id = declare_parameter("base_link_frame_id", "base_link");

    pose_publisher = this->create_publisher<geometry_msgs::msg::PoseStamped>("output/pose", 10);
    accel_publisher = this->create_publisher<geometry_msgs::msg::AccelWithCovarianceStamped>("output/accel", 10);
    kinematic_publisher = this->create_publisher<nav_msgs::msg::Odometry>("output/kinematic", 10);
    pub_state_ = this->create_publisher<autoware_adapi_v1_msgs::msg::LocalizationInitializationState>(
      "output/localization_init_state", 
      rclcpp::QoS{10}.transient_local());
    
    initial_pose_3d_publisher = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "output/initial_pose_3d", 
      rclcpp::QoS{10}.transient_local());
    
    gnss_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "input/gnss/pose", 10, std::bind(&LocalizationGnss::gnss_pose_callback, this, std::placeholders::_1));
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "input/imu", 10, std::bind(&LocalizationGnss::imu_callback, this, std::placeholders::_1));
    velocity_status_sub_ = this->create_subscription<autoware_auto_vehicle_msgs::msg::VelocityReport>(
        "input/velocity_status", 10, std::bind(&LocalizationGnss::velocity_callback, this, std::placeholders::_1));
  }

  ~LocalizationGnss()
  {
   
  }

  void gnss_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    state.y = msg->pose.position.y;
    state.x = msg->pose.position.x;
    state.z = msg->pose.position.z;

    state.r_w = msg->pose.orientation.w;
    state.r_x = msg->pose.orientation.x;
    state.r_y = msg->pose.orientation.y;
    state.r_z = msg->pose.orientation.z;

    if (!state_published) {
      send_initial_pose_3d();
      change_state();
    }

    send_base_link(base_link_frame_id);
    
    send_acceleration();
    send_kinematic();
    send_pose();
  }


  void velocity_callback(const autoware_auto_vehicle_msgs::msg::VelocityReport::SharedPtr msg)
  {
    state.lv_x = msg->longitudinal_velocity;
    state.lv_y = msg->lateral_velocity;
    state.lv_z = 0;
  }


  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    state.av_x = msg->angular_velocity.x;
    state.av_y = msg->angular_velocity.y;
    state.av_z = msg->angular_velocity.z;

    state.la_x = msg->linear_acceleration.x;
    state.la_y = msg->linear_acceleration.y;
    state.la_z = msg->linear_acceleration.z;
  }


  void send_base_link(std::string frameid)
  {
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped.header.frame_id = "map";
    transform_stamped.child_frame_id = frameid;
    transform_stamped.header.stamp = this->now();

    transform_stamped.transform.translation.x = state.x;
    transform_stamped.transform.translation.y = state.y;
    transform_stamped.transform.translation.z = state.z;

    transform_stamped.transform.rotation.x = state.r_x;
    transform_stamped.transform.rotation.y = state.r_y;
    transform_stamped.transform.rotation.z = state.r_z;
    transform_stamped.transform.rotation.w = state.r_w;

    tf2_broadcaster_.sendTransform(transform_stamped);

   
  }

  void send_pose()
  {
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.frame_id = "map";
    pose_stamped.header.stamp = this->now();
    pose_stamped.pose.position.x = state.x;
    pose_stamped.pose.position.y = state.y;
    pose_stamped.pose.position.z = state.z;

    pose_stamped.pose.orientation.x = state.r_x;
    pose_stamped.pose.orientation.y = state.r_y;
    pose_stamped.pose.orientation.z = state.r_z;
    pose_stamped.pose.orientation.w = state.r_w;

    pose_publisher->publish(pose_stamped);
  }

  void send_acceleration()
  {
    geometry_msgs::msg::AccelWithCovarianceStamped accel_stamped;
    accel_stamped.header.frame_id = "base_link";
    accel_stamped.header.stamp = this->now();
    accel_stamped.accel.accel.linear.x = state.la_x;
    accel_stamped.accel.accel.linear.y = state.la_y;
    accel_stamped.accel.accel.linear.z = state.la_z;

    accel_stamped.accel.accel.angular.x = state.aa_x;
    accel_stamped.accel.accel.angular.y = state.aa_y;
    accel_stamped.accel.accel.angular.z = state.aa_z;

    accel_publisher->publish(accel_stamped);
  }

  void send_kinematic()
  {
    nav_msgs::msg::Odometry kinematic;
    kinematic.header.frame_id = "map";
    kinematic.header.stamp = this->now();
    kinematic.pose.pose.position.x = state.x;
    kinematic.pose.pose.position.y = state.y;
    kinematic.pose.pose.position.z = state.z;

    kinematic.pose.pose.orientation.x = state.r_x;
    kinematic.pose.pose.orientation.y = state.r_y;
    kinematic.pose.pose.orientation.z = state.r_z;
    kinematic.pose.pose.orientation.w = state.r_w;

    kinematic.twist.twist.linear.x = state.lv_x;
    kinematic.twist.twist.linear.y = state.lv_y;
    kinematic.twist.twist.linear.z = state.lv_z;

    kinematic.twist.twist.angular.x = state.av_x;
    kinematic.twist.twist.angular.y = state.av_y;
    kinematic.twist.twist.angular.z = state.av_z;

    kinematic_publisher->publish(kinematic);
  }

  void send_initial_pose_3d() {
    geometry_msgs::msg::PoseWithCovarianceStamped initial_pose_3d;
    initial_pose_3d.header.frame_id = "map";
    initial_pose_3d.header.stamp = this->now();
    initial_pose_3d.pose.pose.position.x = state.x;
    initial_pose_3d.pose.pose.position.y = state.y;
    initial_pose_3d.pose.pose.position.z = state.z;

    initial_pose_3d.pose.pose.orientation.x = state.r_x;
    initial_pose_3d.pose.pose.orientation.y = state.r_y;
    initial_pose_3d.pose.pose.orientation.z = state.r_z;
    initial_pose_3d.pose.pose.orientation.w = state.r_w;

    initial_pose_3d_publisher->publish(initial_pose_3d);

  }
  void change_state()
  {

    if (state_published) {
      return;
    }

    autoware_adapi_v1_msgs::msg::LocalizationInitializationState state_;

    state_.stamp = now();
    state_.state = autoware_adapi_v1_msgs::msg::LocalizationInitializationState::INITIALIZING;
    pub_state_->publish(state_);

    state_.state = autoware_adapi_v1_msgs::msg::LocalizationInitializationState::INITIALIZED;
    pub_state_->publish(state_);


    state_published = true; 


  }
};


int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalizationGnss>());
  rclcpp::shutdown();
  return 0;
}
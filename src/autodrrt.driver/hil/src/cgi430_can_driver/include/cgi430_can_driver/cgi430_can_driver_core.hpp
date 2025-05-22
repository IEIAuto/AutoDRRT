#ifndef CGI430_CAN_DRIVER__CGI430_CAN_DRIVER_CORE_HPP_
#define CGI430_CAN_DRIVER__CGI430_CAN_DRIVER_CORE_HPP_

#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <chrono>
#include <unistd.h>
#include "cgi430_can_driver/controlcan.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>

#include <GeographicLib/LocalCartesian.hpp>
#include <GeographicLib/Geocentric.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std::chrono_literals;
// using namespace GeographicLib;

class Cgi430Can : public rclcpp::Node
{
public:
    Cgi430Can(const std::string & node_name, const rclcpp::NodeOptions & node_options);
    ~Cgi430Can();

struct ImuData
{
    float accel_x;
    float accel_y;
    float accel_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
};

private:
    /*publisher*/
    // To Autoware
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr Imu_pub_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr Gnss_pub_;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr Gnss_pose_pub_;


    /* input values msg:pub*/
    // sensor_msgs::msg::Imu::ConstSharedPtr imu_msg_ptr;

    rclcpp::TimerBase::SharedPtr pub_imu_timer_;
    void pubImuOnTimer();

    rclcpp::TimerBase::SharedPtr pub_gnss_timer_;
    void pubGnssOnTimer();

private:
    /*ros param*/
    float pub_imu_rate_;
    std::string imu_frame_id_;
    std::string imu_topic_name_;
    std::string gnss_pose_topic_name_;
    float pub_imu_dt_;

    std::string gnss_topic_name_;
    float pub_gnss_dt_;
    GeographicLib::LocalCartesian local_cartesian_;

private:

    using ImuDataPtr = std::shared_ptr<ImuData>;
    ImuDataPtr imu_data;

    double latest_latitude_ = 0.0;
    double latest_longitude_ = 0.0;
    double latest_altitude_ = 0.0;

    double latest_heading_ = 0.0;
    double latest_pitch_ = 0.0;
    double latest_roll_ = 0.0;

    double x_,y_,z_;

    double parseLatitude(const uint8_t* can_data);
    double parseAltitude(const uint8_t* can_data);
    double parseHeading(const uint8_t* can_data);
    int64_t extractSignedSignal(const uint8_t* data, int startBit, int length);
    double decodeLatitudeOrLongitude(const uint8_t* data, int startBit, int length, double scale, double offset);

private:
    /*CAN CARD */
    VCI_BOARD_INFO pInfo[5];

    int receive_length_;
    VCI_CAN_OBJ rec[300];    //CAN帧结构体
    int len_=299;           //设定用来接收的帧结构体数组长度

    /*Thread 相关*/
    std::thread cgi430_rec_thread_;
    std::mutex imu_data_mutex_;

    std::mutex gnss_data_mutex_;

    /*Can Card Fuction*/
    void handleCanFunc();
    void cgi430_rec_func();
    void checkCanCardStatus();
};

#endif
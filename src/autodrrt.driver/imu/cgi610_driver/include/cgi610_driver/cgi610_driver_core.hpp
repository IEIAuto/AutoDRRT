#ifndef CGI610_DRIVER__CGI610_DRIVER_CORE_HPP_
#define CGI610_DRIVER__CGI610_DRIVER_CORE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <serial/serial.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unistd.h>

#define GPSWeek     2
#define GPSTime     3
#define Heading     4
#define Pitch       5
#define Roll        6
#define Gyro_x      7
#define Gyro_y      8
#define Gyro_z      9
#define Acc_x       10
#define Acc_y       11
#define Acc_z       12
#define Latitude    13
#define Longitude   14
#define Altitude    15
#define VE          16
#define VN          17
#define VU          18
#define Velocity    19
#define NSV1        20
#define NSV2        21
#define Status      22
#define Age         23
#define Space       24

using namespace std::chrono_literals;

class CGI610_Analyze : public rclcpp::Node
{
public:
    CGI610_Analyze(const std::string & node_name, const rclcpp::NodeOptions & node_options);
    ~CGI610_Analyze();
private:
    void SerialPortInit();
    void SerialPortAnalyze();

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr pub_navsatfix_;

    sensor_msgs::msg::Imu imu_msg;
    sensor_msgs::msg::NavSatFix navsatfix_msg;

    serial::Serial ser;

    std::string port_name;
    int baudrate_;
    int output_hz_;

    std::string read;
    std::string str;
    std::string foundCHC = "$GPCHC";

    std::chrono::milliseconds timer_ms{10};

    std::thread Monitor_thread_;
    void monitor_func();
    std::map<std::string, std::string> key_value_stdmap_;
};


#endif
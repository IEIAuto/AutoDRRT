#ifndef CGI430_CAN_DRIVER__CGI430_CAN_DRIVER_CORE_HPP_
#define CGI430_CAN_DRIVER__CGI430_CAN_DRIVER_CORE_HPP_

#include <rclcpp/rclcpp.hpp>
#include "sensor_msgs/msg/imu.hpp"

#include <iostream>
#include <thread>
#include <string>
#include <ctime>
#include <unistd.h>

#include "cgi430_can_driver/controlcan.h"

class CGI430_CAN : public rclcpp::Node
{
public:
    CGI430_CAN(const std::string & node_name, const rclcpp::NodeOptions & node_options);
    ~CGI430_CAN() = default;

private:
    std::string _out_imu_topic;
    double out_hz_;

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;
    sensor_msgs::msg::Imu  imu_data;

private:
    //CAN卡相关
    VCI_BOARD_INFO pInfo1[4];          //用来获取设备信息

    int receive_length_;
    VCI_CAN_OBJ rec[50];  //CAN帧结构体
    int len_=200;         //设定用来接收的帧结构体数组长度

    std::thread cgi430_rec_thread_;

private:
    void init_param();
    void CanPortInit();
    void ReceiveDate();
    void CGI430_Analyze_func();

};

#endif
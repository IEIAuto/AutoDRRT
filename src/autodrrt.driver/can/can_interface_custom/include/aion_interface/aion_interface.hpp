#ifndef AIONINTERFACE__AIONINTERFACE_HPP_
#define AIONINTERFACE__AIONINTERFACE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "autoware_auto_control_msgs/msg/ackermann_control_command.hpp"
#include "autoware_auto_vehicle_msgs/msg/gear_command.hpp"
#include "autoware_auto_vehicle_msgs/msg/velocity_report.hpp"
#include "autoware_auto_vehicle_msgs/msg/gear_report.hpp"
#include "autoware_auto_vehicle_msgs/msg/steering_report.hpp"

#include "can_interface_custom/controlcan.h"
#include "can_interface_custom/can_param.hpp"

#include <iostream>
#include <cstdio>
#include <string>
#include <cmath>
#include <chrono>
#include <thread>
#include <deque>
#include <unistd.h>
#include <mutex>

using namespace std::chrono_literals;

class AionInterface : public rclcpp::Node
{
public:
    AionInterface(const std::string & node_name, const rclcpp::NodeOptions & node_options);
    ~AionInterface();
    
private:
    /* subscriber */
    //From Autoware
    rclcpp::Subscription<autoware_auto_control_msgs::msg::AckermannControlCommand>::SharedPtr control_cmd_sub_;
    rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::GearCommand>::SharedPtr gear_cmd_sub_;
    
    /* publishers */
    // To Autoware
    rclcpp::Publisher<autoware_auto_vehicle_msgs::msg::VelocityReport>::SharedPtr vehicle_twist_pub_;
    rclcpp::Publisher<autoware_auto_vehicle_msgs::msg::GearReport>::SharedPtr gear_status_pub_;
    rclcpp::Publisher<autoware_auto_vehicle_msgs::msg::SteeringReport>::SharedPtr steering_status_pub_;

    rclcpp::TimerBase::SharedPtr pub_report_timer_;
    void pubReportOnTimer();

    rclcpp::TimerBase::SharedPtr pub_command_timer_;
    void pubCommandOnTimer();

    /* input values  msg:sub form autoware control*/
    autoware_auto_control_msgs::msg::AckermannControlCommand::ConstSharedPtr control_cmd_ptr_;
    autoware_auto_vehicle_msgs::msg::GearCommand::ConstSharedPtr gear_cmd_ptr_;

    rclcpp::Time control_command_received_time_;

    /* callbacks */
    void callbackControlCmd(const autoware_auto_control_msgs::msg::AckermannControlCommand::ConstSharedPtr msg);
    void callbackGearCmd(const autoware_auto_vehicle_msgs::msg::GearCommand::ConstSharedPtr msg);

    /* fuctions */
    // void publishCommands();
    float CalCtrlByacc(float acc_);
    std::optional<int32_t> toAutowareShiftReport(const uint8_t);
    float SteerToWheelAngle(float);
    std::optional<uint8_t> GearAutowareToAion(const uint8_t);
    void send_adcu(float steerangreq, float autotrqwhlreq, float brakereq, uint8_t gearlvlreq, uint8_t latctrlreq, uint8_t lngctrlreq);

    /* ros param*/
    float pub_rate_;
    float report_dt_;
    float command_dt_;
    std::string base_frame_id_;

    std::deque<SCU_Mode> scu_mode_v;

    std::shared_ptr<uint8_t> scu_latctrmode_ptr_;
    std::shared_ptr<uint8_t> scu_lngctrmode_ptr_;
    std::shared_ptr<float>  velocity_report_ptr_;
    std::shared_ptr<float>  steer_report_ptr_;
    std::shared_ptr<uint8_t>  gear_report_ptr_;


private:
     /*CAN CARD */
    VCI_BOARD_INFO pInfo[5];

    int receive_length_;
    VCI_CAN_OBJ rec[300];    //CAN帧结构体
    int len_=299;           //设定用来接收的帧结构体数组长度

    VCI_CAN_OBJ vco[3];

    AION_ADCU_1 adcu_1={0,0,0,0,0,0};
    AION_ADCU_2 adcu_2={0,0,0,0,0};

    /*Thread 相关*/
    std::thread can_rec_thread_;
    // std::thread can_send_thread_;
    std::thread ros_rec_thread_;

    std::mutex mux_;
    std::mutex mux_scu_;

    /*Can Card Fuction*/
    void handleCanFunc();
    void can_rec_func();
    // void can_send_func();
    void checkCanCardStatus();
};






#endif
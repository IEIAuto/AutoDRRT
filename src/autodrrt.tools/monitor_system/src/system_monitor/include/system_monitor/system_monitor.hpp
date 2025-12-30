#ifndef SYSTEM_MONITOR__SYSTEM_MONITOR_HPP_
#define SYSTEM_MONITOR__SYSTEM_MONITOR_HPP_

#include <iostream>
#include <string>
#include <chrono>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <fstream>
#include <sstream>
#include <map>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <diagnostic_updater/diagnostic_updater.hpp>
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

struct CpuStatus 
{
    double load;
    double temperature;
    double frequency;
};

struct GpuStatus 
{
    double load;
    double temperature;
    double frequency;
}; 

struct MemoryStats{
    double load;
    double  temperature_celsius;
    std::string uptime;
    int read_mb;
    int write_mb;

};

struct NetworkStats
{
    double rx_rate; //下载速度(kb/s)
    double tx_rate; //上传速度(kb/s)
};

class SystemMonitor : public rclcpp::Node
{
public:
    SystemMonitor(const std::string & node_name, const rclcpp::NodeOptions & node_options);
    ~SystemMonitor();

private:
    void publsih_stats();
    CpuStatus get_cpu_status();
    GpuStatus get_gpu_status();
    MemoryStats get_memory_status();
    NetworkStats get_network_usage(const std::string &interface);

private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr sys_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;

    std::string network_card_id_;
    

};




#endif
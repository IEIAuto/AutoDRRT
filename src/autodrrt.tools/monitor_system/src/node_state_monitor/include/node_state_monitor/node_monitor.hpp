#ifndef NODE_STATE_MONITOR__NODE_MONITOR_HPP_
#define NODE_STATE_MONITOR__NODE_MONITOR_HPP_

#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <set>
#include <vector>
#include <sstream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <diagnostic_updater/diagnostic_updater.hpp>
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"


using namespace std::chrono_literals;

class NodeMonitor : public rclcpp::Node
{
public:
    NodeMonitor(const std::string & node_name, const rclcpp::NodeOptions & node_options);
    ~NodeMonitor();

private:
    std::set<std::string> monitored_nodes_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostic_pub_;

    double vx_threshold_; 

private:
    std::set<std::string> get_current_ros2_nodes();
    void check_nodes();

};





#endif
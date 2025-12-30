/*Author : INSPUR
  Data : 2025.04.16
  Description ：node state monitor
*/
#include "node_state_monitor/node_monitor.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    auto node = std::make_shared<NodeMonitor>("node_monitor_node", node_options);
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
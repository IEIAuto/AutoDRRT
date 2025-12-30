/*Author : INSPUR
  Data : 2025.03.28
  Description ：system monitor
*/
#include "system_monitor/system_monitor.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    auto node = std::make_shared<SystemMonitor>("system_monitor_node", node_options);
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
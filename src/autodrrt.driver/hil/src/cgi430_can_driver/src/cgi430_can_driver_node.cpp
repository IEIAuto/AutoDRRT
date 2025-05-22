/*Author : INSPUR
  Data : 2024.11.19
  Description ：华测CGI430 CAN报文解析
*/
#include "cgi430_can_driver/cgi430_can_driver_core.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    auto node = std::make_shared<Cgi430Can>("cgi430_can_driver_node", node_options);
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
/*Author : INSPUR
  Data : 2024.01.31
  Description ：华测CGI430 CAN报文解析
*/

#include "cgi430_can_driver/cgi430_can_driver_core.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;

    auto node = std::make_shared<CGI430_CAN>("cgi430_can_driver_node", node_options);

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
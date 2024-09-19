/*Author : lyl
  Data : 2024.01.09
  Description ：大陆408 CAN报文解析 可视化
*/

#include "ARS408_driver/ARS408_CAN_core.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;

    auto node = std::make_shared<ARS408_CAN>("ARS408_CAN_node", node_options);

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
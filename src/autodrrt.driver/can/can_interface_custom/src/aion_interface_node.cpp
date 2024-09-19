#include "can_interface_custom/can_interface_custom.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    auto node = std::make_shared<AionInterface>("can_interface_custom_node", node_options);
    rclcpp::spin(node);
    rclcpp::shutdown();

}
#include "aion_interface/aion_interface.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    auto node = std::make_shared<AionInterface>("aion_interface_node", node_options);
    rclcpp::spin(node);
    rclcpp::shutdown();

}
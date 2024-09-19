#include "cgi610_driver/cgi610_driver_core.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    auto node = std::make_shared<CGI610_Analyze>("cgi610_driver_node", node_options);

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
} 
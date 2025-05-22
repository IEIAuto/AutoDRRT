/*Author : lyl
  Data : 2024.04.09
  Description : multi lidar calibration
*/

#include <rclcpp/rclcpp.hpp>
#include <multi_lidar_calibration/multi_lidar_calibration.hpp>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;

    auto node = std::make_shared<MULTI_LIDAR_CALIBRATION>("multi_lidar_calibration_node", node_options);

    rclcpp::spin(node);
    rclcpp::shutdown();
}
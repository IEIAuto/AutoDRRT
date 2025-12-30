#include "node_state_monitor/node_monitor.hpp"

NodeMonitor::NodeMonitor(const std::string & node_name, const rclcpp::NodeOptions & node_options)
: rclcpp::Node(node_name, node_options)
{
    
    this->declare_parameter<std::vector<std::string>>("monitored_nodes",{});
    auto node_list = this->get_parameter("monitored_nodes").as_string_array();
    monitored_nodes_.insert(node_list.begin(), node_list.end());

    diagnostic_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
    timer_ = this->create_wall_timer(5s, std::bind(&NodeMonitor::check_nodes, this));
    RCLCPP_INFO(this->get_logger(),"System Monitor Node has started.");
    vx_threshold_ = declare_parameter<double>("vx_threshold");
    RCLCPP_INFO(this->get_logger(),"vx_threshold_: %f", vx_threshold_);
    
    for(const auto& name : monitored_nodes_){
        RCLCPP_INFO(this->get_logger(), "Monitoring node: %s", name.c_str());
    }
}

NodeMonitor::~NodeMonitor(){}

std::set<std::string> NodeMonitor::get_current_ros2_nodes()
{
    std::set<std::string> result;
    FILE* pipe = popen("ros2 node list", "r");
    if (!pipe) {
        RCLCPP_ERROR(this->get_logger(), "Failed to run 'ros2 node list'");
        return result;
    }

    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line(buffer);
        line.erase(line.find_last_not_of(" \n\r\t") + 1);  // trim
        if (!line.empty()) {
            result.insert(line);
        }
    }
    pclose(pipe);
    return result;

}

void NodeMonitor::check_nodes()
{
    auto current_nodes = get_current_ros2_nodes();

    diagnostic_msgs::msg::DiagnosticArray diag_array;
    diag_array.header.stamp = this->get_clock()->now();

    for (const auto& node : monitored_nodes_) {
        diagnostic_msgs::msg::DiagnosticStatus status;
        status.name = "NodeMonitor: " + node;

        if (current_nodes.find(node) == current_nodes.end()) {
            status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
            status.message = "Node missing!";
            RCLCPP_WARN(this->get_logger(), "[ALERT] Node [%s] is missing!", node.c_str());
        } else {
            status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
            status.message = "Node is alive.";
            RCLCPP_INFO(this->get_logger(), "Node [%s] is alive.", node.c_str());
        }

        diag_array.status.push_back(status);
    }

    diagnostic_pub_->publish(diag_array);
    
}
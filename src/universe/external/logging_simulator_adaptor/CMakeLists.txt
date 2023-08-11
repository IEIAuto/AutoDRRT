cmake_minimum_required(VERSION 3.14)
project(logging_simulator_adaptor)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_add_library(logging_simulator_adaptor SHARED
  data_transfer.cpp
  data_transfer_awsim.cpp
)
# ament_auto_add_executable(logging_simulator_adaptor
#   data_transfer.cpp
# )

rclcpp_components_register_node(logging_simulator_adaptor
  PLUGIN "logging_simulator_adator::DataTransfer"
  EXECUTABLE dataTransfer_node
)

rclcpp_components_register_node(logging_simulator_adaptor
  PLUGIN "logging_simulator_adator::DataTransferAwsim"
  EXECUTABLE dataTransfer_awsim_node
)
ament_auto_package(INSTALL_TO_SHARE)

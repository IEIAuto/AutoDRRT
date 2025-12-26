#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/component_manager.hpp>
#include <memory>
#include <string>
#include <unordered_set>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <vector>
#include <iostream>

using rclcpp_components::ComponentManager;

void set_rt_properties(int prio, const std::unordered_set<size_t> & affinity)
{
  struct sched_param sched_param = { 0 };
  sched_param.sched_priority = prio;
  sched_setscheduler(0, SCHED_FIFO, &sched_param);

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (const auto cpu : affinity) {
    CPU_SET(cpu, &cpuset);
  }
  sched_setaffinity(0, sizeof(cpuset), &cpuset);
}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    set_rt_properties(60,{3,4});

    auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
    rclcpp::NodeOptions node_options;

    auto manager = std::make_shared<ComponentManager>(
    std::weak_ptr<rclcpp::Executor>(executor),
    "custom_pointcloud_container",
    node_options);

    executor->add_node(manager);
    executor->spin();

    rclcpp::shutdown();

    return 0;
}
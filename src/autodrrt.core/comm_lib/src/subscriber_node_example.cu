#include "cuda_ipc_api.h"
#include <rclcpp/rclcpp.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cstring>

enum class SubscriberMode {
    CPU_ONLY,
    GPU_ONLY,
    BOTH
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    SubscriberMode mode = SubscriberMode::CPU_ONLY;  
    bool enable_stats = false;  
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            std::string mode_str = argv[i + 1];
            if (mode_str == "GPU" || mode_str == "gpu") {
                mode = SubscriberMode::GPU_ONLY;
            } else if (mode_str == "BOTH" || mode_str == "both") {
                mode = SubscriberMode::BOTH;
            } else {
                mode = SubscriberMode::CPU_ONLY;
            }
            i++;
        } else if (strcmp(argv[i], "--stats") == 0) {
            enable_stats = true;
        }
    }
    
    auto node = std::make_shared<rclcpp::Node>("cuda_ipc_subscriber");
    
    std::string mode_str = (mode == SubscriberMode::CPU_ONLY) ? "CPU" :
                          (mode == SubscriberMode::GPU_ONLY) ? "GPU" : "BOTH";
    RCLCPP_INFO(node->get_logger(), "Subscriber node started");
    RCLCPP_INFO(node->get_logger(), "Mode: %s", mode_str.c_str());
    
    CudaIpcSubscriber subscriber(node);
    
    if (!subscriber.is_initialized()) {
        RCLCPP_ERROR(node->get_logger(), "Failed to initialize subscriber");
        return 1;
    }
    
    size_t receive_count = 0;
    std::vector<int64_t> e2e_latencies;
    std::vector<int64_t> meta_open_times;
    std::vector<int64_t> mapping_times;
    std::vector<int64_t> transfer_times;
    
    if (mode == SubscriberMode::CPU_ONLY || mode == SubscriberMode::BOTH) {
        subscriber.set_cpu_callback([node, enable_stats, &receive_count, 
                                     &e2e_latencies, &meta_open_times, &mapping_times, &transfer_times,
                                     &subscriber](const void* data, size_t size, const shm_meta& meta) {
            
            auto callback_time = std::chrono::high_resolution_clock::now();
            int64_t process_end_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                callback_time.time_since_epoch()).count();
            
            int64_t e2e_latency_ns = -1;
            int64_t e2e_latency_us = -1;
            if (process_end_timestamp_ns > 0 && meta.timestamp_ns > 0) {
                e2e_latency_ns = process_end_timestamp_ns - meta.timestamp_ns;
                e2e_latency_us = e2e_latency_ns / 1000;
            }
            
            int64_t meta_open_us = 0, mapping_us = 0, transfer_us = 0;
            subscriber.get_last_timing_us(meta_open_us, mapping_us, transfer_us);
            
            receive_count++;
            
            if (enable_stats) {
                e2e_latencies.push_back(e2e_latency_us);
                meta_open_times.push_back(meta_open_us);
                mapping_times.push_back(mapping_us);
                transfer_times.push_back(transfer_us);
            }
            
            if (!enable_stats || receive_count % 10 == 0 || receive_count == 1) {
                
                int64_t subscriber_total_us = meta_open_us + mapping_us + transfer_us;
                
                int64_t publisher_shm_us = meta.shm_write_us;
                int64_t publisher_metadata_us = meta.metadata_us;
                int64_t publisher_ros_pub_us = meta.ros_pub_us;
                bool has_publisher_timings = (publisher_shm_us > 0 || publisher_metadata_us > 0 || publisher_ros_pub_us > 0);
                
                int64_t estimated_publisher_us = (e2e_latency_us > subscriber_total_us) ? 
                                                 (e2e_latency_us - subscriber_total_us) : 0;
                
                int64_t estimated_ros2_transport_us = 30;
                if (has_publisher_timings) {
                    estimated_ros2_transport_us = std::max<int64_t>(5, publisher_ros_pub_us);
                }
                
                int64_t publisher_actual_us = has_publisher_timings ?
                    (publisher_shm_us + publisher_metadata_us + publisher_ros_pub_us) :
                    std::max<int64_t>(0, estimated_publisher_us - estimated_ros2_transport_us);
                
                RCLCPP_INFO(node->get_logger(), 
                           "Received CPU data #%zu: size=%zu\n"
                           "  E2E Latency Breakdown:\n"
                           "    Total E2E:        %ld us\n"
                           "    Publisher side:   %s%ld us (shm_write=%ld, metadata=%ld, ros_pub=%ld)\n"
                           "    ROS2 transport:   ~%ld us\n"
                           "    Subscriber side:  %ld us (meta_open=%ld, mapping=%ld, transfer=%ld)",
                           receive_count, size, e2e_latency_us,
                           has_publisher_timings ? "" : "~",
                           publisher_actual_us,
                           publisher_shm_us, publisher_metadata_us, publisher_ros_pub_us,
                           estimated_ros2_transport_us,
                           subscriber_total_us, meta_open_us, mapping_us, transfer_us);
            }
            
            if (!enable_stats) {
                const uint8_t* bytes = static_cast<const uint8_t*>(data);
                std::cout << "  First 10 bytes: ";
                for (size_t i = 0; i < std::min(size, size_t(10)); i++) {
                    std::cout << (int)bytes[i] << " ";
                }
                std::cout << std::endl;
            }
            
            if (meta.width > 0 && meta.height > 0) {
                RCLCPP_INFO(node->get_logger(),
                           "  Image metadata: %zux%zux%zu",
                           meta.width, meta.height, meta.channels);
            }
            
            if (enable_stats && receive_count >= 100 && receive_count % 100 == 0) {
                
                auto calc_stats = [](const std::vector<int64_t>& times) -> std::tuple<int64_t, int64_t, int64_t, int64_t> {
                    if (times.empty()) return {0, 0, 0, 0};
                    int64_t min_val = *std::min_element(times.begin(), times.end());
                    int64_t max_val = *std::max_element(times.begin(), times.end());
                    int64_t sum = 0;
                    for (auto t : times) sum += t;
                    int64_t avg = sum / times.size();
                    std::vector<int64_t> sorted = times;
                    std::sort(sorted.begin(), sorted.end());
                    int64_t median = sorted[sorted.size() / 2];
                    return {min_val, max_val, avg, median};
                };
                
                auto [e2e_min, e2e_max, e2e_avg, e2e_median] = calc_stats(e2e_latencies);
                auto [meta_min, meta_max, meta_avg, meta_median] = calc_stats(meta_open_times);
                auto [map_min, map_max, map_avg, map_median] = calc_stats(mapping_times);
                auto [trans_min, trans_max, trans_avg, trans_median] = calc_stats(transfer_times);
                
                std::cout << "\n========================================\n";
                std::cout << "Statistics (after " << receive_count << " receives):\n";
                std::cout << "========================================\n";
                std::cout << "End-to-End Latency (us): Min=" << e2e_min << ", Max=" << e2e_max 
                          << ", Avg=" << e2e_avg << ", Median=" << e2e_median << "\n";
                std::cout << "Meta Open (us): Min=" << meta_min << ", Max=" << meta_max 
                          << ", Avg=" << meta_avg << ", Median=" << meta_median << "\n";
                std::cout << "Mapping (us): Min=" << map_min << ", Max=" << map_max 
                          << ", Avg=" << map_avg << ", Median=" << map_median << "\n";
                std::cout << "Transfer (us): Min=" << trans_min << ", Max=" << trans_max 
                          << ", Avg=" << trans_avg << ", Median=" << trans_median << "\n";
                std::cout << "========================================\n\n";
            }
        });
    }
    
    if (mode == SubscriberMode::GPU_ONLY || mode == SubscriberMode::BOTH) {
        subscriber.set_gpu_callback([node, enable_stats, &receive_count,
                                     &e2e_latencies, &meta_open_times, &mapping_times, &transfer_times,
                                     &subscriber](void* data, size_t size, const shm_meta& meta) {
            
            auto callback_time = std::chrono::high_resolution_clock::now();
            int64_t process_end_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                callback_time.time_since_epoch()).count();
            
            int64_t e2e_latency_ns = -1;
            int64_t e2e_latency_us = -1;
            if (process_end_timestamp_ns > 0 && meta.timestamp_ns > 0) {
                e2e_latency_ns = process_end_timestamp_ns - meta.timestamp_ns;
                e2e_latency_us = e2e_latency_ns / 1000;
            }
            
            int64_t meta_open_us = 0, mapping_us = 0, transfer_us = 0;
            subscriber.get_last_timing_us(meta_open_us, mapping_us, transfer_us);
            
            receive_count++;
            
            if (enable_stats) {
                e2e_latencies.push_back(e2e_latency_us);
                meta_open_times.push_back(meta_open_us);
                mapping_times.push_back(mapping_us);
                transfer_times.push_back(transfer_us);
            }
            
            if (!enable_stats || receive_count % 10 == 0 || receive_count == 1) {
                
                int64_t subscriber_total_us = meta_open_us + mapping_us + transfer_us;
                
                int64_t publisher_shm_us = meta.shm_write_us;
                int64_t publisher_metadata_us = meta.metadata_us;
                int64_t publisher_ros_pub_us = meta.ros_pub_us;
                bool has_publisher_timings = (publisher_shm_us > 0 || publisher_metadata_us > 0 || publisher_ros_pub_us > 0);
                
                int64_t estimated_publisher_us = (e2e_latency_us > subscriber_total_us) ? 
                                                 (e2e_latency_us - subscriber_total_us) : 0;
                
                int64_t estimated_ros2_transport_us = 30;
                if (has_publisher_timings) {
                    estimated_ros2_transport_us = std::max<int64_t>(5, publisher_ros_pub_us);
                }
                
                int64_t publisher_actual_us = has_publisher_timings ?
                    (publisher_shm_us + publisher_metadata_us + publisher_ros_pub_us) :
                    std::max<int64_t>(0, estimated_publisher_us - estimated_ros2_transport_us);
                
                RCLCPP_INFO(node->get_logger(),
                           "Received GPU data #%zu: size=%zu\n"
                           "  E2E Latency Breakdown:\n"
                           "    Total E2E:        %ld us\n"
                           "    Publisher side:   %s%ld us (shm_write=%ld, metadata=%ld, ros_pub=%ld)\n"
                           "    ROS2 transport:   ~%ld us\n"
                           "    Subscriber side:  %ld us (meta_open=%ld, mapping=%ld, transfer=%ld)",
                           receive_count, size, e2e_latency_us,
                           has_publisher_timings ? "" : "~",
                           publisher_actual_us,
                           publisher_shm_us, publisher_metadata_us, publisher_ros_pub_us,
                           estimated_ros2_transport_us,
                           subscriber_total_us, meta_open_us, mapping_us, transfer_us);
            }
            
            if (!enable_stats) {
                
                uint8_t host_data[10];
                size_t copy_size = std::min(size, size_t(10));
                cudaError_t err = cudaMemcpy(host_data, data, copy_size, cudaMemcpyDeviceToHost);
                
                if (err == cudaSuccess) {
                    std::cout << "  First 10 bytes (from GPU): ";
                    for (size_t i = 0; i < copy_size; i++) {
                        std::cout << (int)host_data[i] << " ";
                    }
                    std::cout << std::endl;
                } else {
                    RCLCPP_WARN(node->get_logger(), 
                               "Failed to copy GPU data: %s", cudaGetErrorString(err));
                }
            }
            
            if (meta.width > 0 && meta.height > 0) {
                RCLCPP_INFO(node->get_logger(),
                           "  Image metadata: %zux%zux%zu",
                           meta.width, meta.height, meta.channels);
            }
        });
    }
    
    subscriber.start();
    RCLCPP_INFO(node->get_logger(), "Subscriber started, waiting for messages...");
    
    rclcpp::spin(node);
    
    subscriber.stop();
    
    if (enable_stats && !e2e_latencies.empty()) {
        auto calc_stats = [](const std::vector<int64_t>& times) -> std::tuple<int64_t, int64_t, int64_t, int64_t> {
            if (times.empty()) return {0, 0, 0, 0};
            int64_t min_val = *std::min_element(times.begin(), times.end());
            int64_t max_val = *std::max_element(times.begin(), times.end());
            int64_t sum = 0;
            for (auto t : times) sum += t;
            int64_t avg = sum / times.size();
            std::vector<int64_t> sorted = times;
            std::sort(sorted.begin(), sorted.end());
            int64_t median = sorted[sorted.size() / 2];
            return {min_val, max_val, avg, median};
        };
        
        auto [e2e_min, e2e_max, e2e_avg, e2e_median] = calc_stats(e2e_latencies);
        auto [meta_min, meta_max, meta_avg, meta_median] = calc_stats(meta_open_times);
        auto [map_min, map_max, map_avg, map_median] = calc_stats(mapping_times);
        auto [trans_min, trans_max, trans_avg, trans_median] = calc_stats(transfer_times);
        
        std::cout << "\n========================================\n";
        std::cout << "Final Statistics (total " << receive_count << " receives):\n";
        std::cout << "========================================\n";
        std::cout << "End-to-End Latency (us):\n";
        std::cout << "  Min:    " << std::setw(10) << e2e_min << " us\n";
        std::cout << "  Max:    " << std::setw(10) << e2e_max << " us\n";
        std::cout << "  Avg:    " << std::setw(10) << e2e_avg << " us\n";
        std::cout << "  Median: " << std::setw(10) << e2e_median << " us\n";
        std::cout << "\nMeta Open (us): Min=" << meta_min << ", Max=" << meta_max 
                  << ", Avg=" << meta_avg << ", Median=" << meta_median << "\n";
        std::cout << "Mapping (us): Min=" << map_min << ", Max=" << map_max 
                  << ", Avg=" << map_avg << ", Median=" << map_median << "\n";
        std::cout << "Transfer (us): Min=" << trans_min << ", Max=" << trans_max 
                  << ", Avg=" << trans_avg << ", Median=" << trans_median << "\n";
        std::cout << "========================================\n";
    }
    
    rclcpp::shutdown();
    return 0;
}

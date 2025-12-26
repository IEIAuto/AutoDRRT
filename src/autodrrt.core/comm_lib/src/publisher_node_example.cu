#include "cuda_ipc_api.h"
#include <rclcpp/rclcpp.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iomanip>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    size_t data_size = 8 * 1024 * 1024;  
    DataLocation location = DataLocation::CPU;  
    size_t max_publish_count = 0;  
    double publish_rate_hz = 10.0;  
    bool use_direct_mode = false;  
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            data_size = std::atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--location") == 0 && i + 1 < argc) {
            if (strcmp(argv[i + 1], "GPU") == 0 || strcmp(argv[i + 1], "gpu") == 0) {
                location = DataLocation::GPU;
            } else {
                location = DataLocation::CPU;
            }
            i++;
        } else if (strcmp(argv[i], "--direct") == 0 || strcmp(argv[i], "-d") == 0) {
            use_direct_mode = true;
            location = DataLocation::GPU;  
        } else if (strcmp(argv[i], "--count") == 0 && i + 1 < argc) {
            max_publish_count = std::atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--rate") == 0 && i + 1 < argc) {
            publish_rate_hz = std::atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--hz") == 0 && i + 1 < argc) {
            publish_rate_hz = std::atof(argv[i + 1]);
            i++;
        }
    }
    
    int64_t publish_interval_ms = static_cast<int64_t>(1000.0 / publish_rate_hz);
    double actual_interval_ms = 1000.0 / publish_rate_hz;
    
    auto node = std::make_shared<rclcpp::Node>("cuda_ipc_publisher");
    RCLCPP_INFO(node->get_logger(), "Publisher node started");
    RCLCPP_INFO(node->get_logger(), "Data size: %zu bytes (%.2f MB)", 
                data_size, data_size / (1024.0 * 1024.0));
    RCLCPP_INFO(node->get_logger(), "Location: %s", (location == DataLocation::CPU ? "CPU" : "GPU"));
    RCLCPP_INFO(node->get_logger(), "Publish rate: %.2f Hz (interval: %.2f ms)", 
                publish_rate_hz, actual_interval_ms);
    if (max_publish_count > 0) {
        RCLCPP_INFO(node->get_logger(), "Publish count: %zu", max_publish_count);
    } else {
        RCLCPP_INFO(node->get_logger(), "Publish count: infinite (Ctrl+C to stop)");
    }
    
    PublishOptions pub_options;
    if (location == DataLocation::GPU) {
        
        pub_options.use_zero_copy = true;
        RCLCPP_INFO(node->get_logger(), "GPU mode: zero-copy enabled. Data will be generated directly in mapped shared memory (no copy needed).");
    }
    if (use_direct_mode) {
        pub_options.use_zero_copy = true;
        RCLCPP_INFO(node->get_logger(), "Using direct mode: data will be generated directly in mapped shared memory (zero-copy, no GPU->GPU copy)");
    }
    CudaIpcPublisher publisher(node, pub_options);
    
    if (!publisher.is_initialized()) {
        RCLCPP_ERROR(node->get_logger(), "Failed to initialize publisher");
        return 1;
    }
    
    void* data = nullptr;
    void* direct_gpu_buf = nullptr;  
    
    if (location == DataLocation::GPU) {
        
        direct_gpu_buf = publisher.get_gpu_buffer(data_size);
        if (direct_gpu_buf) {
            use_direct_mode = true;
            RCLCPP_INFO(node->get_logger(), 
                       "GPU mode with zero-copy: Using direct mode. "
                       "Data will be generated directly in mapped shared memory (no copy needed).");
        } else {
            
            RCLCPP_WARN(node->get_logger(), 
                       "Zero-copy not available, falling back to normal GPU mode (with GPU->GPU copy).");
            cudaError_t err = cudaMalloc(&data, data_size);
            if (err != cudaSuccess) {
                RCLCPP_ERROR(node->get_logger(), "Failed to allocate GPU memory: %s", 
                            cudaGetErrorString(err));
                return 1;
            }
            RCLCPP_INFO(node->get_logger(), 
                       "GPU data ready: %zu bytes (will be copied to shared memory)", 
                       data_size);
        }
    } else if (location == DataLocation::CPU) {
        
        data = malloc(data_size);
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < data_size; i++) {
            bytes[i] = static_cast<uint8_t>(i % 256);
        }
        RCLCPP_INFO(node->get_logger(), "Prepared CPU data: %zu bytes (data is in CPU memory)", data_size);
    } else {
        
        cudaError_t err = cudaMalloc(&data, data_size);
        if (err != cudaSuccess) {
            RCLCPP_ERROR(node->get_logger(), "Failed to allocate GPU memory: %s", 
                        cudaGetErrorString(err));
            return 1;
        }
        
        RCLCPP_INFO(node->get_logger(), 
                   "GPU data ready: %zu bytes (data is already in GPU memory)", 
                   data_size);
    }
    
    size_t publish_count = 0;
    std::vector<int64_t> publish_times;
    auto test_start = std::chrono::high_resolution_clock::now();
    
    if (max_publish_count > 0) {
        RCLCPP_INFO(node->get_logger(), "Starting to publish data (%zu times)...", max_publish_count);
        publish_times.reserve(max_publish_count);
    } else {
        RCLCPP_INFO(node->get_logger(), "Starting to publish data (infinite loop, Ctrl+C to stop)...");
    }
    
    auto generate_data_kernel = [](uint8_t* buf, size_t size, uint8_t seed) {
        size_t threads_per_block = 256;
        size_t blocks = (size + threads_per_block - 1) / threads_per_block;
        
        cudaMemset(buf, seed, size);
    };
    
    while (rclcpp::ok() && (max_publish_count == 0 || publish_count < max_publish_count)) {
        publish_count++;
        
        bool success = false;
        auto iteration_start = std::chrono::high_resolution_clock::now();
        
        int64_t prepare_time_us = 0;
        int64_t publish_api_time_us = 0;
        int64_t spin_time_us = 0;
        
        auto prepare_start = std::chrono::high_resolution_clock::now();
        bool direct_path = (use_direct_mode && direct_gpu_buf);
        
        if (direct_path) {
            
            uint8_t seed = static_cast<uint8_t>(publish_count % 256);
            generate_data_kernel(static_cast<uint8_t*>(direct_gpu_buf), data_size, seed);
            
            cudaDeviceSynchronize();  
        } else {
            
            if (location == DataLocation::CPU) {
                uint8_t* bytes = static_cast<uint8_t*>(data);
                bytes[0] = static_cast<uint8_t>(publish_count % 256);
            } else {
                uint8_t first_byte = static_cast<uint8_t>(publish_count % 256);
                cudaMemcpy(data, &first_byte, 1, cudaMemcpyHostToDevice);
            }
        }
        
        auto prepare_end = std::chrono::high_resolution_clock::now();
        prepare_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            prepare_end - prepare_start).count();
        
        auto publish_api_start = std::chrono::high_resolution_clock::now();
        if (direct_path) {
            success = publisher.publish_direct(data_size);
        } else {
            success = publisher.publish(data, data_size, location);
        }
        auto publish_api_end = std::chrono::high_resolution_clock::now();
        publish_api_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            publish_api_end - publish_api_start).count();
        
        auto iteration_end = std::chrono::high_resolution_clock::now();
        int64_t total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            iteration_end - iteration_start).count();
        
        auto spin_start = std::chrono::high_resolution_clock::now();
        rclcpp::spin_some(node);
        auto spin_end = std::chrono::high_resolution_clock::now();
        spin_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            spin_end - spin_start).count();
        
        if (success) {
            int64_t publish_time = publisher.get_last_publish_time_us();
            
            if (max_publish_count > 0) {
                publish_times.push_back(publish_time);
            }
            
            if (max_publish_count == 0 || publish_count % 10 == 0 || publish_count == 1) {
                RCLCPP_INFO(node->get_logger(), 
                           "Published #%zu: size=%zu, location=%s, prepare=%ld us, publish_api=%ld us, spin=%ld us, shm_write=%ld us, total=%ld us",
                           publish_count, data_size,
                           (location == DataLocation::CPU ? "CPU" : "GPU"),
                           prepare_time_us, publish_api_time_us, spin_time_us,
                           publish_time, total_time_us);
            }
        } else {
            RCLCPP_ERROR(node->get_logger(), "Failed to publish data #%zu", publish_count);
        }
        
        auto next_publish_time = iteration_start + std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(1.0 / publish_rate_hz));
        auto current_time = std::chrono::high_resolution_clock::now();
        
        if (current_time < next_publish_time) {
            std::this_thread::sleep_until(next_publish_time);
        }
    }
    
    if (max_publish_count > 0 && !publish_times.empty()) {
        auto test_end = std::chrono::high_resolution_clock::now();
        int64_t total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            test_end - test_start).count();
        
        int64_t min_time = *std::min_element(publish_times.begin(), publish_times.end());
        int64_t max_time = *std::max_element(publish_times.begin(), publish_times.end());
        int64_t sum_time = 0;
        for (auto t : publish_times) sum_time += t;
        int64_t avg_time = sum_time / publish_times.size();
        
        std::vector<int64_t> sorted_times = publish_times;
        std::sort(sorted_times.begin(), sorted_times.end());
        int64_t median_time = sorted_times[sorted_times.size() / 2];
        
        std::cout << "\n========================================\n";
        std::cout << "Publisher Statistics:\n";
        std::cout << "========================================\n";
        std::cout << "Total publishes: " << publish_times.size() << "\n";
        std::cout << "Total time: " << total_time_us << " us (" 
                  << total_time_us / 1000.0 << " ms)\n";
        std::cout << "SHM Write Time (us):\n";
        std::cout << "  Min:    " << std::setw(10) << min_time << " us\n";
        std::cout << "  Max:    " << std::setw(10) << max_time << " us\n";
        std::cout << "  Avg:    " << std::setw(10) << avg_time << " us\n";
        std::cout << "  Median: " << std::setw(10) << median_time << " us\n";
        std::cout << "Throughput: " << std::fixed << std::setprecision(2)
                  << (data_size * publish_times.size() * 1000000.0) / (total_time_us * 1024.0 * 1024.0)
                  << " MB/s\n";
        std::cout << "========================================\n";
    }
    
    if (location == DataLocation::CPU) {
        free(data);
    } else {
        cudaFree(data);
    }
    
    rclcpp::shutdown();
    return 0;
}

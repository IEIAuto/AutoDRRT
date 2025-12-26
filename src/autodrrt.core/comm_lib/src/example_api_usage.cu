#include "cuda_ipc_api.h"
#include <rclcpp/rclcpp.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <chrono>

void example_cpu_to_cpu() {
    std::cout << "\n=== 示例 1: CPU-CPU 数据传输 ===\n" << std::endl;
    
    rclcpp::init(0, nullptr);
    
    auto pub_node = std::make_shared<rclcpp::Node>("cpu_publisher");
    RCLCPP_INFO(pub_node->get_logger(), "Publisher node created");
    CudaIpcPublisher publisher(pub_node);
    
    auto sub_node = std::make_shared<rclcpp::Node>("cpu_subscriber");
    CudaIpcSubscriber subscriber(sub_node);
    
    subscriber.set_cpu_callback([](const void* data, size_t size, const shm_meta& meta) {
        std::cout << "收到 CPU 数据: size=" << size << " bytes" << std::endl;
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        std::cout << "前 10 个字节: ";
        for (size_t i = 0; i < std::min(size, size_t(10)); i++) {
            std::cout << (int)bytes[i] << " ";
        }
        std::cout << std::endl;
    });
    
    subscriber.start();
    
    std::thread sub_thread([&sub_node]() {
        rclcpp::spin(sub_node);
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    const size_t data_size = 8 * 1024 * 1024;  
    uint8_t cpu_data[data_size];
    for (size_t i = 0; i < data_size; i++) {
        cpu_data[i] = static_cast<uint8_t>(i % 256);
    }
    
    std::cout << "发布 CPU 数据..." << std::endl;
    publisher.publish(cpu_data, data_size, DataLocation::CPU);
    
    rclcpp::spin_some(pub_node);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    rclcpp::shutdown();
    sub_thread.join();
}

void example_cpu_to_gpu() {
    std::cout << "\n=== 示例 2: CPU-GPU 数据传输 ===\n" << std::endl;
    
    rclcpp::init(0, nullptr);
    
    auto pub_node = std::make_shared<rclcpp::Node>("cpu_gpu_publisher");
    CudaIpcPublisher publisher(pub_node);
    
    auto sub_node = std::make_shared<rclcpp::Node>("cpu_gpu_subscriber");
    CudaIpcSubscriber subscriber(sub_node);
    
    subscriber.set_gpu_callback([](void* data, size_t size, const shm_meta& meta) {
        std::cout << "收到 GPU 数据: size=" << size << " bytes" << std::endl;
        
        uint8_t* gpu_data = static_cast<uint8_t*>(data);
        uint8_t host_data[10];
        cudaMemcpy(host_data, gpu_data, 10, cudaMemcpyDeviceToHost);
        
        std::cout << "GPU 数据前 10 个字节: ";
        for (int i = 0; i < 10; i++) {
            std::cout << (int)host_data[i] << " ";
        }
        std::cout << std::endl;
    });
    
    subscriber.start();
    
    std::thread sub_thread([&sub_node]() {
        rclcpp::spin(sub_node);
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    const size_t data_size = 8 * 1024 * 1024;  
    uint8_t cpu_data[data_size];
    for (size_t i = 0; i < data_size; i++) {
        cpu_data[i] = static_cast<uint8_t>(i % 256);
    }
    
    std::cout << "发布 CPU 数据（将映射到 GPU）..." << std::endl;
    publisher.publish(cpu_data, data_size, DataLocation::CPU);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    rclcpp::shutdown();
    sub_thread.join();
}

void example_gpu_to_gpu() {
    std::cout << "\n=== 示例 3: GPU-GPU 数据传输 ===\n" << std::endl;
    
    rclcpp::init(0, nullptr);
    
    auto pub_node = std::make_shared<rclcpp::Node>("gpu_publisher");
    CudaIpcPublisher publisher(pub_node);
    
    auto sub_node = std::make_shared<rclcpp::Node>("gpu_subscriber");
    CudaIpcSubscriber subscriber(sub_node);
    
    subscriber.set_gpu_callback([](void* data, size_t size, const shm_meta& meta) {
        std::cout << "收到 GPU 数据: size=" << size << " bytes" << std::endl;
        
        uint8_t* gpu_data = static_cast<uint8_t*>(data);
        uint8_t host_data[10];
        cudaMemcpy(host_data, gpu_data, 10, cudaMemcpyDeviceToHost);
        
        std::cout << "GPU 数据前 10 个字节: ";
        for (int i = 0; i < 10; i++) {
            std::cout << (int)host_data[i] << " ";
        }
        std::cout << std::endl;
    });
    
    subscriber.start();
    
    std::thread sub_thread([&sub_node]() {
        rclcpp::spin(sub_node);
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    const size_t data_size = 8 * 1024 * 1024;  
    void* gpu_data = nullptr;
    cudaMalloc(&gpu_data, data_size);
    
    uint8_t host_init[data_size];
    for (size_t i = 0; i < data_size; i++) {
        host_init[i] = static_cast<uint8_t>(i % 256);
    }
    cudaMemcpy(gpu_data, host_init, data_size, cudaMemcpyHostToDevice);
    
    std::cout << "发布 GPU 数据..." << std::endl;
    publisher.publish(gpu_data, data_size, DataLocation::GPU);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    cudaFree(gpu_data);
    rclcpp::shutdown();
    sub_thread.join();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " <example_number>" << std::endl;
        std::cout << "  1 - CPU-CPU 数据传输" << std::endl;
        std::cout << "  2 - CPU-GPU 数据传输" << std::endl;
        std::cout << "  3 - GPU-GPU 数据传输" << std::endl;
        return 1;
    }
    
    int example = std::atoi(argv[1]);
    
    switch (example) {
        case 1:
            example_cpu_to_cpu();
            break;
        case 2:
            example_cpu_to_gpu();
            break;
        case 3:
            example_gpu_to_gpu();
            break;
        default:
            std::cout << "无效的示例编号: " << example << std::endl;
            return 1;
    }
    
    return 0;
}

#ifndef CUDA_IPC_API_H
#define CUDA_IPC_API_H

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <functional>
#include <memory>
#include <string>
#include <cstddef>
#include <cstdint>
#include "ipc_common.h"

class CudaIpcPublisherImpl;
class CudaIpcSubscriberImpl;

enum class DataLocation {
    CPU,  
    GPU   
};

struct PublishOptions {
    std::string topic_name = "cuda_ipc_ready";  
    std::string shm_name_prefix = "/cuda_ipc_buffer";  
    bool use_zero_copy = true;  
    size_t max_buffer_size = 0;  
};

struct SubscribeOptions {
    std::string topic_name = "cuda_ipc_ready";  
    bool use_zero_copy = true;  
    bool auto_cleanup = true;  
};

using CpuDataCallback = std::function<void(
    const void* data,
    size_t size,
    const shm_meta& metadata
)>;

using GpuDataCallback = std::function<void(
    void* data,
    size_t size,
    const shm_meta& metadata
)>;

class CudaIpcPublisher {
public:
    
    CudaIpcPublisher(
        rclcpp::Node::SharedPtr node,
        const PublishOptions& options = PublishOptions()
    );
    
    ~CudaIpcPublisher();
    
    bool publish(const void* data, size_t size, DataLocation location);
    
    bool publish(
        const void* data,
        size_t size,
        DataLocation location,
        size_t width,
        size_t height,
        size_t channels
    );
    
    bool is_initialized() const;
    
    int64_t get_last_publish_time_us() const;
    
    void* get_host_buffer(size_t size = 0);
    
    void* get_gpu_buffer(size_t size = 0);
    
    bool publish_direct(
        size_t size,
        size_t width = 0,
        size_t height = 0,
        size_t channels = 0
    );

private:
    std::unique_ptr<CudaIpcPublisherImpl> impl_;  
};

class CudaIpcSubscriber {
public:
    
    CudaIpcSubscriber(
        rclcpp::Node::SharedPtr node,
        const SubscribeOptions& options = SubscribeOptions()
    );
    
    ~CudaIpcSubscriber();
    
    void set_cpu_callback(CpuDataCallback callback);
    
    void set_gpu_callback(GpuDataCallback callback);
    
    void start();
    
    void stop();
    
    bool is_initialized() const;
    
    bool get_last_timing_us(int64_t& meta_open, int64_t& mapping, int64_t& transfer) const;
    
    int64_t get_last_process_end_timestamp_ns() const;

private:
    std::unique_ptr<CudaIpcSubscriberImpl> impl_;  
};

#endif 

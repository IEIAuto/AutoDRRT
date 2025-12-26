#include "cuda_ipc_api.h"
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <sstream>

class CudaIpcPublisherImpl {
public:
    CudaIpcPublisherImpl(
        rclcpp::Node::SharedPtr node,
        const PublishOptions& options
    ) : node_(node), options_(options), initialized_(false), last_publish_time_us_(-1) {
        
        publisher_ = node_->create_publisher<std_msgs::msg::UInt64>(options_.topic_name, 10);
        
        RCLCPP_INFO(node_->get_logger(), "Publisher initialized. Topic: %s", options_.topic_name.c_str());
        
        initialized_ = true;
    }
    
    ~CudaIpcPublisherImpl() {
        cleanup_shared_memory();
    }
    
    bool publish(const void* data, size_t size, DataLocation location) {
        return publish(data, size, location, 0, 0, 0);
    }
    
    bool publish(
        const void* data,
        size_t size,
        DataLocation location,
        size_t width,
        size_t height,
        size_t channels
    ) {
        if (!initialized_ || !data || size == 0) {
            RCLCPP_ERROR(node_->get_logger(), "Invalid parameters for publish");
            return false;
        }
        
        auto publish_start_time = std::chrono::high_resolution_clock::now();
        auto publish_start_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            publish_start_time.time_since_epoch()).count();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (!host_buf_ || current_size_ != size) {
            cleanup_shared_memory();
            if (!setup_shared_memory(size)) {
                return false;
            }
        }
        
        const char* copy_mode_str = "";
        int64_t copy_time_us = 0;
        int64_t sync_time_us = 0;
        
        if (location == DataLocation::CPU) {
            
            auto copy_start = std::chrono::high_resolution_clock::now();
            memcpy(host_buf_, data, size);
            auto copy_end = std::chrono::high_resolution_clock::now();
            copy_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                copy_end - copy_start).count();
            copy_mode_str = "CPU->SHM";
        } else {
            
            if (use_gpu_memory_) {
                
                auto copy_start = std::chrono::high_resolution_clock::now();
                cudaMemcpy(host_buf_, data, size, cudaMemcpyDeviceToHost);
                auto copy_end = std::chrono::high_resolution_clock::now();
                copy_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    copy_end - copy_start).count();
                
                auto sync_start = std::chrono::high_resolution_clock::now();
                cudaDeviceSynchronize();
                auto sync_end = std::chrono::high_resolution_clock::now();
                sync_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    sync_end - sync_start).count();
                copy_mode_str = "GPU->Host->SHM (explicit copy)";
            } else {
                
                if (dev_ptr_) {
                    auto copy_start = std::chrono::high_resolution_clock::now();
                    cudaMemcpy(dev_ptr_, data, size, cudaMemcpyDeviceToDevice);
                    auto copy_end = std::chrono::high_resolution_clock::now();
                    copy_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        copy_end - copy_start).count();
                    
                    auto sync_start = std::chrono::high_resolution_clock::now();
                    cudaDeviceSynchronize();
                    auto sync_end = std::chrono::high_resolution_clock::now();
                    sync_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        sync_end - sync_start).count();
                    copy_mode_str = "GPU->SHM (zero-copy)";
                } else {
                    
                    auto copy_start = std::chrono::high_resolution_clock::now();
                    cudaMemcpy(host_buf_, data, size, cudaMemcpyDeviceToHost);
                    auto copy_end = std::chrono::high_resolution_clock::now();
                    copy_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        copy_end - copy_start).count();
                    
                    auto sync_start = std::chrono::high_resolution_clock::now();
                    cudaDeviceSynchronize();
                    auto sync_end = std::chrono::high_resolution_clock::now();
                    sync_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        sync_end - sync_start).count();
                    copy_mode_str = "GPU->Host->SHM (fallback)";
                }
            }
        }
        
        auto shm_write_end = std::chrono::high_resolution_clock::now();
        auto shm_write_time = std::chrono::duration_cast<std::chrono::microseconds>(
            shm_write_end - start_time).count();
        
        auto metadata_start = std::chrono::high_resolution_clock::now();
        
        auto timestamp_ns = publish_start_timestamp_ns;
        
        publish_count_++;
        meta_->width = width;
        meta_->height = height;
        meta_->channels = channels;
        meta_->data_size = size;
        meta_->timestamp_ns = timestamp_ns;
        meta_->publish_index = publish_count_;
        meta_->ready = 1;
        meta_->ack = 0;
        
        auto metadata_end = std::chrono::high_resolution_clock::now();
        auto metadata_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            metadata_end - metadata_start).count();
        
        auto ros_pub_start = std::chrono::high_resolution_clock::now();
        auto msg = std_msgs::msg::UInt64();
        msg.data = timestamp_ns;
        publisher_->publish(msg);
        auto ros_pub_end = std::chrono::high_resolution_clock::now();
        auto ros_pub_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            ros_pub_end - ros_pub_start).count();
        
        last_publish_time_us_ = shm_write_time + metadata_time_us + ros_pub_time_us;
        
        if (location == DataLocation::GPU && !use_gpu_memory_) {
            
            RCLCPP_INFO(node_->get_logger(), 
                        "Published data #%zu, size=%zu bytes, location=%s, mode=%s, "
                        "copy=%ld us, sync=%ld us, shm_write=%ld us, metadata=%ld us, ros_pub=%ld us, total=%ld us",
                        publish_count_, size, 
                        (location == DataLocation::CPU ? "CPU" : "GPU"),
                        copy_mode_str,
                        copy_time_us, sync_time_us, shm_write_time, metadata_time_us, ros_pub_time_us, last_publish_time_us_);
        } else {
            
            RCLCPP_INFO(node_->get_logger(), 
                        "Published data #%zu, size=%zu bytes, location=%s, mode=%s, "
                        "shm_write=%ld us, metadata=%ld us, ros_pub=%ld us, total=%ld us",
                        publish_count_, size, 
                        (location == DataLocation::CPU ? "CPU" : "GPU"),
                        copy_mode_str,
                        shm_write_time, metadata_time_us, ros_pub_time_us, last_publish_time_us_);
        }
        
        return true;
    }
    
    bool is_initialized() const {
        return initialized_;
    }
    
    int64_t get_last_publish_time_us() const {
        return last_publish_time_us_;
    }
    
    void* get_host_buffer(size_t size = 0) {
        
        if (size > 0 && (!host_buf_ || current_size_ != size)) {
            if (!setup_shared_memory(size)) {
                return nullptr;
            }
        }
        
        if (initialized_ && host_buf_) {
            return host_buf_;
        }
        return nullptr;
    }
    
    void* get_gpu_buffer(size_t size = 0) {
        
        if (size > 0 && (!host_buf_ || current_size_ != size)) {
            if (!setup_shared_memory(size)) {
                return nullptr;
            }
        }
        
        if (initialized_ && !use_gpu_memory_ && dev_ptr_) {
            return dev_ptr_;
        }
        return nullptr;
    }
    
    bool publish_direct(
        size_t size,
        size_t width = 0,
        size_t height = 0,
        size_t channels = 0
    ) {
        if (!initialized_ || size == 0) {
            RCLCPP_ERROR(node_->get_logger(), "Invalid parameters for publish_direct");
            return false;
        }
        
        if (use_gpu_memory_ || !dev_ptr_) {
            RCLCPP_ERROR(node_->get_logger(), 
                        "publish_direct() can only be used in zero-copy mode. "
                        "Use publish() with DataLocation::GPU instead.");
            return false;
        }
        
        if (!host_buf_ || current_size_ != size) {
            cleanup_shared_memory();
            if (!setup_shared_memory(size)) {
                return false;
            }
        }
        
        auto publish_start_time = std::chrono::high_resolution_clock::now();
        auto publish_start_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            publish_start_time.time_since_epoch()).count();
        
        auto sync_start = std::chrono::high_resolution_clock::now();
        cudaDeviceSynchronize();  
        auto sync_end = std::chrono::high_resolution_clock::now();
        int64_t sync_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            sync_end - sync_start).count();
        
        int64_t shm_write_time_us = 0;
        
        auto metadata_start = std::chrono::high_resolution_clock::now();
        auto timestamp_ns = publish_start_timestamp_ns;
        
        publish_count_++;
        meta_->width = width;
        meta_->height = height;
        meta_->channels = channels;
        meta_->data_size = size;
        meta_->timestamp_ns = timestamp_ns;
        meta_->publish_index = publish_count_;
        meta_->ready = 1;
        meta_->ack = 0;
        auto metadata_end = std::chrono::high_resolution_clock::now();
        int64_t metadata_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            metadata_end - metadata_start).count();
        
        auto ros_pub_start = std::chrono::high_resolution_clock::now();
        auto msg = std_msgs::msg::UInt64();
        msg.data = timestamp_ns;
        publisher_->publish(msg);
        auto ros_pub_end = std::chrono::high_resolution_clock::now();
        int64_t ros_pub_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            ros_pub_end - ros_pub_start).count();
        
        meta_->shm_write_us = shm_write_time_us;
        meta_->metadata_us = metadata_time_us;
        meta_->ros_pub_us = ros_pub_time_us;
        
        int64_t publish_op_time_us = metadata_time_us + ros_pub_time_us;
        last_publish_time_us_ = sync_time_us + publish_op_time_us;  
        
        RCLCPP_INFO(node_->get_logger(), 
                    "Published data #%zu (direct), size=%zu bytes, mode=GPU->SHM (zero-copy, no copy), "
                    "sync=%ld us, metadata=%ld us, ros_pub=%ld us, total=%ld us",
                    publish_count_, size, sync_time_us, metadata_time_us, ros_pub_time_us, last_publish_time_us_);
        
        return true;
    }

private:
    bool setup_shared_memory(size_t size) {
        
        std::stringstream ss;
        ss << options_.shm_name_prefix << "_" << getpid();
        std::string shm_name = ss.str();
        
        int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd == -1) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to create shared memory: %s", strerror(errno));
            return false;
        }
        
        if (ftruncate(fd, size) == -1) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to truncate shared memory: %s", strerror(errno));
            close(fd);
            return false;
        }
        
        host_buf_ = mmap(nullptr, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        
        if (host_buf_ == MAP_FAILED) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to mmap shared memory: %s", strerror(errno));
            return false;
        }
        
        current_size_ = size;
        shm_name_ = shm_name;
        
        if (options_.use_zero_copy) {
            cudaError_t err = cudaHostRegister(host_buf_, size, cudaHostRegisterMapped);
            if (err == cudaSuccess) {
                err = cudaHostGetDevicePointer(&dev_ptr_, host_buf_, 0);
                if (err == cudaSuccess) {
                    use_gpu_memory_ = false;
                    RCLCPP_INFO(node_->get_logger(), "Zero-copy mapping enabled for GPU data (GPU can directly access shared memory)");
                } else {
                    cudaHostUnregister(host_buf_);
                    use_gpu_memory_ = true;
                    RCLCPP_WARN(node_->get_logger(), "Zero-copy mapping failed (cudaHostGetDevicePointer), falling back to explicit copy mode");
                }
            } else {
                use_gpu_memory_ = true;
                RCLCPP_WARN(node_->get_logger(), "Zero-copy mapping failed (cudaHostRegister: %s), falling back to explicit copy mode", 
                           cudaGetErrorString(err));
            }
        } else {
            use_gpu_memory_ = true;
            RCLCPP_INFO(node_->get_logger(), "Using explicit copy mode for GPU data (GPU -> Host -> Shared Memory)");
        }
        
        int meta_fd = shm_open(META_SHM_NAME, O_CREAT|O_RDWR, 0666);
        if (meta_fd == -1) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to create meta shared memory: %s", strerror(errno));
            return false;
        }
        
        if (ftruncate(meta_fd, sizeof(shm_meta)) == -1) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to truncate meta shared memory: %s", strerror(errno));
            close(meta_fd);
            return false;
        }
        
        meta_ = (shm_meta*)mmap(nullptr, sizeof(shm_meta), PROT_READ|PROT_WRITE, MAP_SHARED, meta_fd, 0);
        close(meta_fd);
        
        if (meta_ == MAP_FAILED) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to mmap meta shared memory: %s", strerror(errno));
            return false;
        }
        
        strncpy(meta_->shm_name, shm_name.c_str(), sizeof(meta_->shm_name) - 1);
        meta_->shm_name[sizeof(meta_->shm_name) - 1] = '\0';
        meta_->ready = 0;
        meta_->ack = 0;
        
        return true;
    }
    
    void cleanup_shared_memory() {
        if (host_buf_ != nullptr && host_buf_ != MAP_FAILED) {
            if (use_gpu_memory_) {
                
            } else {
                cudaHostUnregister(host_buf_);
            }
            munmap(host_buf_, current_size_);
            if (!shm_name_.empty()) {
                shm_unlink(shm_name_.c_str());
            }
            host_buf_ = nullptr;
        }
        
        if (meta_ != nullptr && meta_ != MAP_FAILED) {
            munmap(meta_, sizeof(shm_meta));
            meta_ = nullptr;
        }
    }
    
    rclcpp::Node::SharedPtr node_;
    PublishOptions options_;
    rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr publisher_;
    
    void* host_buf_ = nullptr;
    void* dev_ptr_ = nullptr;
    shm_meta* meta_ = nullptr;
    std::string shm_name_;
    size_t current_size_ = 0;
    bool use_gpu_memory_ = false;
    bool initialized_ = false;
    size_t publish_count_ = 0;
    int64_t last_publish_time_us_ = -1;
};

class CudaIpcSubscriberImpl {
public:
    CudaIpcSubscriberImpl(
        rclcpp::Node::SharedPtr node,
        const SubscribeOptions& options
    ) : node_(node), options_(options), initialized_(false),
        last_meta_open_us_(-1), last_mapping_us_(-1), last_transfer_us_(-1) {
        
        subscription_ = node_->create_subscription<std_msgs::msg::UInt64>(
            options_.topic_name, 10,
            std::bind(&CudaIpcSubscriberImpl::message_callback, this, std::placeholders::_1));
        
        initialized_ = true;
    }
    
    ~CudaIpcSubscriberImpl() {
        cleanup_shared_memory();
    }
    
    void set_cpu_callback(CpuDataCallback callback) {
        cpu_callback_ = callback;
    }
    
    void set_gpu_callback(GpuDataCallback callback) {
        gpu_callback_ = callback;
    }
    
    void start() {
        
        RCLCPP_INFO(node_->get_logger(), "Subscriber started, waiting for messages...");
    }
    
    void stop() {
        
        subscription_.reset();
    }
    
    bool is_initialized() const {
        return initialized_;
    }
    
    bool get_last_timing_us(int64_t& meta_open, int64_t& mapping, int64_t& transfer) const {
        if (last_meta_open_us_ < 0) {
            return false;
        }
        meta_open = last_meta_open_us_;
        mapping = last_mapping_us_;
        transfer = last_transfer_us_;
        return true;
    }
    
    int64_t get_last_process_end_timestamp_ns() const {
        return last_process_end_timestamp_ns_;
    }

private:
    void message_callback(const std_msgs::msg::UInt64::SharedPtr msg) {
        auto total_start = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point t_meta_open_end, t_mapping_end, t_transfer_start, t_transfer_end;
        
        bool meta_writable = true;
        int meta_fd = shm_open(META_SHM_NAME, O_RDWR, 0666);
        if (meta_fd == -1) {
            RCLCPP_WARN(node_->get_logger(), "Open meta shm O_RDWR failed: %s. Try O_RDONLY.", strerror(errno));
            meta_fd = shm_open(META_SHM_NAME, O_RDONLY, 0666);
            meta_writable = false;
            if (meta_fd == -1) {
                RCLCPP_ERROR(node_->get_logger(), "Failed to open meta shm: %s", strerror(errno));
                return;
            }
        }
        
        int prot = meta_writable ? (PROT_READ|PROT_WRITE) : PROT_READ;
        struct shm_meta* meta = (struct shm_meta*)mmap(nullptr, sizeof(struct shm_meta), prot, MAP_SHARED, meta_fd, 0);
        close(meta_fd);
        
        if (meta == MAP_FAILED) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to mmap meta shm: %s", strerror(errno));
            return;
        }
        
        t_meta_open_end = std::chrono::high_resolution_clock::now();
        
        uint64_t expected_ts = msg->data;
        int retry = 0;
        const int max_retries = 200;  
        
        while (meta->ready != 1) {
            if (retry++ > max_retries) {
                RCLCPP_ERROR(node_->get_logger(), "Timeout waiting for meta ready (retries: %d)", retry);
                munmap(meta, sizeof(struct shm_meta));
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        retry = 0;
        while (meta->timestamp_ns != expected_ts) {
            
            if (meta->timestamp_ns > expected_ts) {
                
                break;  
            }
            
            if (retry++ > 10) {  
                
                RCLCPP_DEBUG(node_->get_logger(), 
                           "Timestamp not updated yet: expected=%llu, got=%llu (using current data)",
                           expected_ts, meta->timestamp_ns);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        void* host_buf = ensure_persistent_mapping(meta->shm_name, meta->data_size);
        if (!host_buf) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to ensure persistent mapping for %s", meta->shm_name);
            munmap(meta, sizeof(struct shm_meta));
            return;
        }
        
        t_mapping_end = std::chrono::high_resolution_clock::now();
        
        auto ms = [](const std::chrono::high_resolution_clock::time_point &a, 
                     const std::chrono::high_resolution_clock::time_point &b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
        };
        
        last_meta_open_us_ = ms(total_start, t_meta_open_end);
        last_mapping_us_ = ms(t_meta_open_end, t_mapping_end);
        
        if (cpu_callback_) {
            
            cpu_callback_(host_buf, meta->data_size, *meta);
        }
        
        if (gpu_callback_) {
            
            void* dev_ptr_to_use = nullptr;
            if (!persistent_use_device_copy_) {
                
                t_transfer_start = std::chrono::high_resolution_clock::now();
                
                if (persistent_dev_ptr_ && meta->data_size > 0) {
                    
                    uint8_t test_byte = 0;
                    cudaError_t err = cudaMemcpy(&test_byte, persistent_dev_ptr_, 1, cudaMemcpyDeviceToHost);
                    if (err == cudaSuccess) {
                        
                        cudaDeviceSynchronize();
                    }
                    
                }
                
                t_transfer_end = std::chrono::high_resolution_clock::now();
                dev_ptr_to_use = persistent_dev_ptr_;
                last_transfer_us_ = ms(t_transfer_start, t_transfer_end);
            } else {
                
                t_transfer_start = std::chrono::high_resolution_clock::now();
                cudaMemcpy(persistent_dev_buffer_, host_buf, meta->data_size, cudaMemcpyHostToDevice);
                t_transfer_end = std::chrono::high_resolution_clock::now();
                dev_ptr_to_use = persistent_dev_buffer_;
                last_transfer_us_ = ms(t_transfer_start, t_transfer_end);
            }
            
            if (dev_ptr_to_use) {
                gpu_callback_(dev_ptr_to_use, meta->data_size, *meta);
            }
        }
        
        auto process_end_time = std::chrono::high_resolution_clock::now();
        last_process_end_timestamp_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
            process_end_time.time_since_epoch()).count();
        
        if (meta_writable) {
            meta->ready = 0;
        }
        munmap(meta, sizeof(struct shm_meta));
    }
    
    void* ensure_persistent_mapping(const char* shm_name, size_t size) {
        
        if (persistent_host_buf_ != nullptr && persistent_shm_name_ == shm_name && persistent_host_size_ == size) {
            return persistent_host_buf_;
        }
        
        if (persistent_host_buf_ != nullptr) {
            if (persistent_registered_) {
                cudaHostUnregister(persistent_host_buf_);
                persistent_registered_ = false;
                persistent_dev_ptr_ = nullptr;
            }
            if (persistent_dev_buffer_ != nullptr) {
                cudaFree(persistent_dev_buffer_);
                persistent_dev_buffer_ = nullptr;
            }
            munmap(persistent_host_buf_, persistent_host_size_);
            persistent_host_buf_ = nullptr;
            persistent_host_size_ = 0;
            persistent_shm_name_.clear();
        }
        
        int fd = shm_open(shm_name, O_RDWR, 0666);
        if (fd == -1) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to open data shared memory: %s", strerror(errno));
            return nullptr;
        }
        
        void* host_buf = mmap(nullptr, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        
        if (host_buf == MAP_FAILED) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to mmap data shared memory: %s", strerror(errno));
            return nullptr;
        }
        
        if (options_.use_zero_copy) {
            cudaError_t err = cudaHostRegister(host_buf, size, cudaHostRegisterMapped);
            if (err == cudaSuccess) {
                err = cudaHostGetDevicePointer(&persistent_dev_ptr_, host_buf, 0);
                if (err == cudaSuccess) {
                    persistent_host_buf_ = host_buf;
                    persistent_host_size_ = size;
                    persistent_shm_name_ = std::string(shm_name);
                    persistent_registered_ = true;
                    persistent_use_device_copy_ = false;
                    return host_buf;
                } else {
                    cudaHostUnregister(host_buf);
                }
            }
        }
        
        void* d_buf = nullptr;
        cudaError_t merr = cudaMalloc(&d_buf, size);
        if (merr != cudaSuccess) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to allocate device buffer: %s", cudaGetErrorString(merr));
            munmap(host_buf, size);
            return nullptr;
        }
        
        persistent_host_buf_ = host_buf;
        persistent_host_size_ = size;
        persistent_shm_name_ = std::string(shm_name);
        persistent_registered_ = false;
        persistent_dev_ptr_ = nullptr;
        persistent_dev_buffer_ = d_buf;
        persistent_use_device_copy_ = true;
        
        return host_buf;
    }
    
    void cleanup_shared_memory() {
        if (persistent_registered_ && persistent_host_buf_ != nullptr) {
            cudaHostUnregister(persistent_host_buf_);
            persistent_registered_ = false;
            persistent_dev_ptr_ = nullptr;
        }
        if (persistent_dev_buffer_ != nullptr) {
            cudaFree(persistent_dev_buffer_);
            persistent_dev_buffer_ = nullptr;
        }
        if (persistent_host_buf_ != nullptr) {
            munmap(persistent_host_buf_, persistent_host_size_);
            persistent_host_buf_ = nullptr;
            persistent_host_size_ = 0;
            persistent_shm_name_.clear();
        }
    }
    
    rclcpp::Node::SharedPtr node_;
    SubscribeOptions options_;
    rclcpp::Subscription<std_msgs::msg::UInt64>::SharedPtr subscription_;
    
    CpuDataCallback cpu_callback_;
    GpuDataCallback gpu_callback_;
    
    void* persistent_host_buf_ = nullptr;
    size_t persistent_host_size_ = 0;
    std::string persistent_shm_name_;
    bool persistent_registered_ = false;
    void* persistent_dev_ptr_ = nullptr;
    void* persistent_dev_buffer_ = nullptr;
    bool persistent_use_device_copy_ = false;
    
    bool initialized_ = false;
    int64_t last_meta_open_us_ = -1;
    int64_t last_mapping_us_ = -1;
    int64_t last_transfer_us_ = -1;
    int64_t last_process_end_timestamp_ns_ = -1;  
};

CudaIpcPublisher::CudaIpcPublisher(
    rclcpp::Node::SharedPtr node,
    const PublishOptions& options
) : impl_(std::make_unique<CudaIpcPublisherImpl>(node, options)) {
}

CudaIpcPublisher::~CudaIpcPublisher() = default;

bool CudaIpcPublisher::publish(const void* data, size_t size, DataLocation location) {
    return impl_->publish(data, size, location);
}

bool CudaIpcPublisher::publish(
    const void* data,
    size_t size,
    DataLocation location,
    size_t width,
    size_t height,
    size_t channels
) {
    return impl_->publish(data, size, location, width, height, channels);
}

bool CudaIpcPublisher::is_initialized() const {
    return impl_->is_initialized();
}

int64_t CudaIpcPublisher::get_last_publish_time_us() const {
    return impl_->get_last_publish_time_us();
}

void* CudaIpcPublisher::get_gpu_buffer(size_t size) {
    return impl_->get_gpu_buffer(size);
}

bool CudaIpcPublisher::publish_direct(
    size_t size,
    size_t width,
    size_t height,
    size_t channels
) {
    return impl_->publish_direct(size, width, height, channels);
}

CudaIpcSubscriber::CudaIpcSubscriber(
    rclcpp::Node::SharedPtr node,
    const SubscribeOptions& options
) : impl_(std::make_unique<CudaIpcSubscriberImpl>(node, options)) {
}

CudaIpcSubscriber::~CudaIpcSubscriber() = default;

void CudaIpcSubscriber::set_cpu_callback(CpuDataCallback callback) {
    impl_->set_cpu_callback(callback);
}

void CudaIpcSubscriber::set_gpu_callback(GpuDataCallback callback) {
    impl_->set_gpu_callback(callback);
}

void CudaIpcSubscriber::start() {
    impl_->start();
}

void CudaIpcSubscriber::stop() {
    impl_->stop();
}

bool CudaIpcSubscriber::is_initialized() const {
    return impl_->is_initialized();
}

bool CudaIpcSubscriber::get_last_timing_us(int64_t& meta_open, int64_t& mapping, int64_t& transfer) const {
    return impl_->get_last_timing_us(meta_open, mapping, transfer);
}

int64_t CudaIpcSubscriber::get_last_process_end_timestamp_ns() const {
    return impl_->get_last_process_end_timestamp_ns();
}

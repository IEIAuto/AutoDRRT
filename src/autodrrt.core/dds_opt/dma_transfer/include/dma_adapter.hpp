#ifndef DMA_ADAPTER_HPP_
#define DMA_ADAPTER_HPP_
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <vector>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/mman.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include "rclcpp/rclcpp.hpp"
#include <dma_customer_msg/msg/dmamessage.hpp>
#include <map>
#include <mutex>
#include <thread>
#include <chrono>

#include "opencv2/opencv.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <fstream>

template<typename MessageT>
class DmaAdapter {
public:
    ~DmaAdapter()
    {
        close(mem_fd);
    }

    DmaAdapter()
    {
        mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
        if(mem_fd < 0)
        {
            std::cout << "Failed to open /dev/mem" << std::endl;
            exit(-1);
        }
    }

    void imageToJPG(const sensor_msgs::msg::Image& image_msg, const std::string& output_file) {
        // Step 1: Convert sensor_msgs::msg::Image to cv::Mat using cv_bridge
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "cv_bridge exception: %s", e.what());
            return;
        }

        // Step 2: Encode the cv::Mat as a JPG image
        std::vector<uchar> buf;
        cv::imencode(".jpg", cv_ptr->image, buf);

        // Step 3: Save the encoded data to a file
        std::ofstream output_file_stream(output_file, std::ios::binary);
        output_file_stream.write(reinterpret_cast<const char*>(buf.data()), buf.size());
        output_file_stream.close();
    }

    void set_physical_address(const uintptr_t & addr){
        physical_address = addr;
    }
    
    void publish_via_dma(rclcpp::Node * node, const MessageT& msg, const std::string & topic){
        std::lock_guard<std::mutex> lock(publish_lock);

        auto currentTime = std::chrono::high_resolution_clock::now();
        auto milliseconds_start = std::chrono::duration_cast<std::chrono::microseconds>(currentTime.time_since_epoch());
        std::cout << "==start time is :" << std::to_string(milliseconds_start.count()) << " us==" << std::endl;

        auto pub_pair = publishers_.find(topic);
        if (pub_pair == publishers_.end())
        {
            publishers_[topic] = node->create_publisher<dma_customer_msg::msg::Dmamessage>(topic, 1);
        }
        auto notify_msg = std::make_unique<dma_customer_msg::msg::Dmamessage>();        
        std::vector<uint8_t> serialized_data;

        currentTime = std::chrono::high_resolution_clock::now();
        auto milliseconds_prepare = std::chrono::duration_cast<std::chrono::microseconds>(currentTime.time_since_epoch());
        std::cout << "==prepare time is :" << std::to_string(milliseconds_prepare.count() - milliseconds_start.count()) << " us==" << std::endl;

        serialize(msg, serialized_data);

        currentTime = std::chrono::high_resolution_clock::now();
        auto milliseconds_serialized = std::chrono::duration_cast<std::chrono::microseconds>(currentTime.time_since_epoch());
        std::cout << "==serialize time is :" << std::to_string(milliseconds_serialized.count() - milliseconds_prepare.count()) << " us==" << std::endl;

        notify_msg->data = serialized_data.size();
        notify_msg->header.stamp = node->now();
        notify_msg->header.frame_id = "base_link";
        dma_write(serialized_data);

        currentTime = std::chrono::high_resolution_clock::now();
        auto milliseconds_dma_write = std::chrono::duration_cast<std::chrono::microseconds>(currentTime.time_since_epoch());
        std::cout << "==dma_write time is :" << std::to_string(milliseconds_dma_write.count() - milliseconds_serialized.count()) << " us==" << std::endl;

        publishers_[topic]->publish(std::move(notify_msg));

        currentTime = std::chrono::high_resolution_clock::now();
        auto milliseconds_total = std::chrono::duration_cast<std::chrono::microseconds>(currentTime.time_since_epoch());
        std::cout << "==totaltime is :" << std::to_string(milliseconds_total.count() - milliseconds_start.count()) << " us==\n" << std::endl;

    }
    
    void create_subscription_via_dma(rclcpp::Node * node,const std::string & topic, void(*func)(const MessageT& messageg)){
        auto sub_pair = subscriptions_.find(topic);
        if (sub_pair == subscriptions_.end())
        {
            subscriptions_[topic] = node->create_subscription<dma_customer_msg::msg::Dmamessage>(topic, 1,
            [node, this, func](const dma_customer_msg::msg::Dmamessage::ConstSharedPtr msg) {
                //code
                size_t msg_size = msg->data;
                
                rclcpp::Time msgTimeStamp = msg->header.stamp;
                rclcpp::Time currentTimeStamp = node->now();
                rclcpp::Duration timeDiff = currentTimeStamp - msgTimeStamp;
                std::cout << "==dma trans time is :" << std::to_string(timeDiff.nanoseconds()/1000) << " us==" << std::endl;
                
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto milliseconds_start = std::chrono::duration_cast<std::chrono::microseconds>(currentTime.time_since_epoch());
                std::cout << "==start time is :" << std::to_string(milliseconds_start.count()) << " us==" << std::endl;

                std::vector<uint8_t> serialized_data_from_dma = dma_read(msg_size);

                currentTime = std::chrono::high_resolution_clock::now();
                auto milliseconds_dma_read = std::chrono::duration_cast<std::chrono::microseconds>(currentTime.time_since_epoch());
                std::cout << "==dma_read time is :" << std::to_string(milliseconds_dma_read.count() - milliseconds_start.count()) << " us==" << std::endl;

                MessageT received_data;
                deserialize(serialized_data_from_dma, received_data);

                imageToJPG(received_data, "/home/orin/swpld/dma_transfer/dump/output.jpg");

                currentTime = std::chrono::high_resolution_clock::now();
                auto milliseconds_deserialized = std::chrono::duration_cast<std::chrono::microseconds>(currentTime.time_since_epoch());
                std::cout << "==milliseconds_deserialized time is :" << std::to_string(milliseconds_deserialized.count() - milliseconds_dma_read.count()) << " us==" << std::endl;

                std::cout << "==total time is :" << std::to_string(milliseconds_deserialized.count() - milliseconds_start.count()) << " us==" << std::endl;
                func(received_data);
            });
        }
    }


private:
    void serialize(const MessageT& msg, std::vector<uint8_t> & serialized_data) {
        rclcpp::Serialization<MessageT> serializer;
        rclcpp::SerializedMessage serialized_msg;
        serializer.serialize_message(&msg, &serialized_msg);
        serialized_data.assign(serialized_msg.get_rcl_serialized_message().buffer,
                               serialized_msg.get_rcl_serialized_message().buffer +
                               serialized_msg.get_rcl_serialized_message().buffer_length);
    }

    void deserialize(const std::vector<uint8_t>& serialized_data, MessageT& received_msg_data) {
        rclcpp::Serialization<MessageT> deserializer;
       
        rclcpp::SerializedMessage serialized_msg;
        serialized_msg.reserve(serialized_data.size());
        serialized_msg.get_rcl_serialized_message().buffer_length = serialized_data.size();
        serialized_msg.get_rcl_serialized_message().buffer = reinterpret_cast<uint8_t*>(malloc(serialized_data.size()));
        memcpy(serialized_msg.get_rcl_serialized_message().buffer, serialized_data.data(), serialized_data.size());
        // 反序列化消息
        deserializer.deserialize_message(&serialized_msg, &received_msg_data);
    }

    std::vector<uint8_t> convertToUint8Vector(const std::vector<uint32_t>& input, size_t actualSize) {
        std::vector<uint8_t> convert_vector;
        for (const uint32_t & value : input) {
            for (size_t j = 0; j < 4; ++j) {
                uint8_t byte = static_cast<uint8_t>((value >> (8 * j)) & 0xFF);
                convert_vector.push_back(byte);
            }
        }
        std::vector<uint8_t> output(convert_vector.begin(), convert_vector.begin() + actualSize);
        return output;
    }
    
    std::vector<uint32_t> convertToUint32Vector(const std::vector<uint8_t>& input) {
        std::vector<uint32_t> output;
        
        size_t inputSize = input.size();
        // size_t paddingSize = (4 - (inputSize % 4)) % 4; // 计算需要添加的零字节数量
        size_t paddingSize = static_cast<size_t>(std::pow(2, std::ceil(std::log2(inputSize))));
        std::vector<uint8_t> paddedInput = input; // 复制输入向量
        
        // 添加零字节填充
        for (size_t i = inputSize; i < paddingSize; ++i) {
            paddedInput.push_back(0);
        }
        output.reserve(paddedInput.size() / 4); // 预分配输出向量的空间
        // 将输入添加到输出
        for (size_t i = 0; i < paddedInput.size(); i += 4) {
            uint32_t value = 0;
            
            for (size_t j = 0; j < 4; ++j) {
                value |= static_cast<uint32_t>(paddedInput[i + j]) << (8 * j);
            }
            
            output.push_back(value);
        }
        
        return output;
    }

    void dma_write(const std::vector<uint8_t>& serialized_data)
    {
        std::vector<uint32_t> serialized_data_u32 = convertToUint32Vector(serialized_data);
        size_t size = serialized_data_u32.size() * sizeof(uint32_t);
        uint32_t *mapped_mem = reinterpret_cast<uint32_t *>(mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, physical_address));
        if(mapped_mem == MAP_FAILED)
        {
            std::cout << "Failed to mmap physical memory" << std::endl;
            close(mem_fd);
            return;
        }
        memcpy(mapped_mem, serialized_data_u32.data(), size);
        munmap(mapped_mem, size);
    }

    std::vector<uint8_t> dma_read(const size_t & uint8_vector_size)
    {
        
        std::vector<uint8_t> serialized_data_u8;
        std::vector<uint32_t> serialized_data_u32;
        size_t size = static_cast<size_t>(std::pow(2, std::ceil(std::log2(uint8_vector_size))));
        serialized_data_u32.resize(size);
        uint32_t *mapped_mem = reinterpret_cast<uint32_t *>(mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, physical_address));
        if(mapped_mem == MAP_FAILED)
        {
            std::cout << "Failed to mmap physical memory" << std::endl;
            close(mem_fd);
            return serialized_data_u8;
        }
        memcpy(serialized_data_u32.data(), mapped_mem, size);
        serialized_data_u8 = convertToUint8Vector(serialized_data_u32, uint8_vector_size);
        // printf("recved data:\n");
        // for (size_t i = 0; i < serialized_data_u8.size(); i ++)
        // {
        //     printf("%d", serialized_data_u8[i]);
        // }
        munmap(mapped_mem, size);
        return serialized_data_u8;
    }

private:
    static uintptr_t physical_address;
    rclcpp::Node* node_;
    std::map<std::string,rclcpp::Publisher<dma_customer_msg::msg::Dmamessage>::SharedPtr> publishers_; 
    std::map<std::string,rclcpp::Subscription<dma_customer_msg::msg::Dmamessage>::SharedPtr> subscriptions_; 
    std::mutex publish_lock;
    int mem_fd;
};

template<typename MessageT>
uintptr_t DmaAdapter<MessageT>::physical_address = 0x2b28000000;
#endif //DMA_ADAPTER_HPP_

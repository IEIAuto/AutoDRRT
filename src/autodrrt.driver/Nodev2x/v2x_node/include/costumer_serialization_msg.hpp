#ifndef CONSTUMER_SERIALIZATION_MSG_HPP_
#define CONSTUMER_SERIALIZATION_MSG_HPP_

#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <classformsg/msg/msgdemo.hpp>

#include <vector>

template<typename MessageT>
class MessageSerializer {
public:
    static void serialize(const MessageT& msg, std::vector<uint8_t>& serialized_data) {
        rclcpp::Serialization<MessageT> serializer;
        rclcpp::SerializedMessage serialized_msg;
        serializer.serialize_message(&msg, &serialized_msg);
        serialized_data.assign(serialized_msg.get_rcl_serialized_message().buffer,
                               serialized_msg.get_rcl_serialized_message().buffer +
                               serialized_msg.get_rcl_serialized_message().buffer_length);
    }

    static void deserialize(const std::vector<uint8_t>& serialized_data, MessageT& received_msg_data) {
        rclcpp::Serialization<MessageT> deserializer;
       
        rclcpp::SerializedMessage serialized_msg;
        serialized_msg.reserve(serialized_data.size());
        serialized_msg.get_rcl_serialized_message().buffer_length = serialized_data.size();
        serialized_msg.get_rcl_serialized_message().buffer = reinterpret_cast<uint8_t*>(malloc(serialized_data.size()));
        memcpy(serialized_msg.get_rcl_serialized_message().buffer, serialized_data.data(), serialized_data.size());
        deserializer.deserialize_message(&serialized_msg, &received_msg_data);
    }

};
#endif //CONSTUMER_SERIALIZATION_MSG_HPP_

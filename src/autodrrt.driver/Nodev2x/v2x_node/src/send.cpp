// Copyright 2021 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "node_v2x.hpp"
#include "costumer_serialization_msg.hpp"

#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>

char *SERVERIP = (char *)"192.168.4.102";
#define SEND_PORT 7201

int cli_sock = -1;

#define ERR_EXIT(m)         \
    do                      \
    {                       \
        perror(m);          \
        exit(EXIT_FAILURE); \
    } while (0)

namespace node_v2x
{

    double getime()
    {
        struct timespec time1 = {0, 0};
        clock_gettime(CLOCK_REALTIME, &time1);
        return (time1.tv_sec + time1.tv_nsec * 0.000000001);
    }

    void udp_client_thread(std::vector<uint8_t> serialized_data)
    {
        std::mutex mtx;
        mtx.lock();
        struct sockaddr_in servaddr;

        if ((cli_sock = socket(PF_INET, SOCK_DGRAM, 0)) < 0)
        {
            ERR_EXIT("socket");
        }

        memset(&servaddr, 0, sizeof(servaddr));
        servaddr.sin_family      = AF_INET;
        servaddr.sin_port        = htons(SEND_PORT);
        servaddr.sin_addr.s_addr = inet_addr(SERVERIP);

        char msg_char[4100] = {};

        long unsigned ser_len = 0;
        for (uint8_t byte: serialized_data) {
            msg_char[ser_len] = static_cast<char>(byte);
            ser_len++;
        }
        msg_char[ser_len] = '\0';

        for (long unsigned i = 0; i < ser_len; i++) {
            printf("%d", msg_char[i]);
        }
        printf("\n\n");

        sendto(cli_sock, msg_char, (ser_len + 1), 0, (struct sockaddr *)&servaddr, sizeof(servaddr));
        memset(&msg_char[0], '\0', sizeof(msg_char));

        close(cli_sock);
        mtx.unlock();
    }

    template <typename typeT>
    void send2OBU(std::string topic_name, const typeT &msg)
    {
        std::vector<uint8_t> nakemsg;
        while (nakemsg.empty() == false) {nakemsg.pop_back();}

        MessageSerializer<typeT>::serialize(msg, nakemsg);

        classformsg::msg::Msgdemo combmsg;
        combmsg.topic     = topic_name;
        combmsg.type      = abi::__cxa_demangle(typeid(typeT).name(), NULL, NULL, NULL);
        combmsg.data      = nakemsg;
        combmsg.timestamp = getime();

        std::vector<uint8_t> serialized_data;
        while (serialized_data.empty() == false) {serialized_data.pop_back();}

        MessageSerializer<classformsg::msg::Msgdemo>::serialize(combmsg, serialized_data);

        udp_client_thread(serialized_data);
    }

    NodeV2X1::NodeV2X1(const rclcpp::NodeOptions &options)
        : Node("node_v2x", options)
    {

        rclcpp::QoS qos(rclcpp::KeepLast(10));
        qos.reliable();
        qos.deadline(rclcpp::Duration(1, 0));

        sub_pose = this->create_subscription<nav_msgs::msg::Odometry>("/localization/kinematic_state", rclcpp::QoS(1), [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
                //code
                std::cout << "I heard sth. from pose" << std::endl;
                send2OBU<nav_msgs::msg::Odometry>("/localization/kinematic_state", *msg);
            });

        sub_object = this->create_subscription<autoware_auto_perception_msgs::msg::PredictedObjects>("/perception/object_recognition/objects", rclcpp::QoS(1), [this](const autoware_auto_perception_msgs::msg::PredictedObjects::ConstSharedPtr msg) {
                //code
                std::cout << "I heard sth. from object" << std::endl;
                send2OBU<autoware_auto_perception_msgs::msg::PredictedObjects>("/perception/object_recognition/objects", *msg);
            });
    }

} // namespace node_v2x

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(node_v2x::NodeV2X1)

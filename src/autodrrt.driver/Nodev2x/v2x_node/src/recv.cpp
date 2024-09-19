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
#include <boost/lexical_cast.hpp>

#define RECV_PORT 8110

int ser_sock = -1;

#define ERR_EXIT(m)         \
    do                      \
    {                       \
        perror(m);          \
        exit(EXIT_FAILURE); \
    } while(0)

#define TYPE_NAME(x) #x

namespace node_v2x
{

template <typename typeT>
class topic_customer {
public:
    topic_customer(typename rclcpp::Publisher<typeT>::SharedPtr Anytopic_ptr, std::string topicname): topic_ptr(Anytopic_ptr), topic_name(topicname) {};

public:
    typename rclcpp::Publisher<typeT>::SharedPtr topic_ptr;
    std::string topic_name;
};

class topic_msg {
public:
    topic_msg(std::string typeinstring, std::string custypeinstring, std::string topicname): first(typeinstring), second(custypeinstring), third(topicname) {}; 

public:
    std::string first; 
    std::string second;
    std::string third;
};

class Topicvec {
public:
    template <typename typeU>
    void add(topic_customer<typeU> new_topic) {
        topic_vec.emplace_back(new_topic);
        std::string str1 = abi::__cxa_demangle(typeid(typeU).name(), NULL, NULL, NULL);
        std::string str2 = abi::__cxa_demangle(typeid(topic_customer<typeU>).name(), NULL, NULL, NULL);
	topic_msg Retrievalmap(str1, str2, new_topic.topic_name);
        type_vec.emplace_back(Retrievalmap);
    }

    void clear() {
        while (topic_vec.empty() == false) {topic_vec.pop_back();}
        while (type_vec.empty() == false) {type_vec.pop_back();}
    } 

public:
    std::vector<std::any> topic_vec;
    std::vector<topic_msg> type_vec;
};

class Base_Operator {
public:
    virtual void try_type(std::any item, classformsg::msg::Msgdemo combmsg) {
        try {
            int intValue = std::any_cast<int>(item);
            std::cout << intValue << std::endl;

            for (uint8_t byte: combmsg.data) {
		std::cout << byte;;
            }
            std::cout << std::endl;

        } catch (const std::bad_any_cast& e) {}	
    }

    virtual ~Base_Operator() {}
};

template<typename typeY>
class try_sth: public Base_Operator {
public:
    void try_type(std::any item, classformsg::msg::Msgdemo combmsg) override {
        auto value = std::any_cast<topic_customer<typeY>>(item); 
        MessageSerializer<typeY>::deserialize(combmsg.data, restored_topic);
        value.topic_ptr -> publish(restored_topic);
    }

public:
    typeY restored_topic;
};


std::map<std::string, std::shared_ptr<Base_Operator>> deserialize_map;

template<typename typeZ>
void register_type() {
    std::shared_ptr<Base_Operator> deserialized_operator = std::make_shared<try_sth<typeZ>>();
    deserialize_map[TYPE_NAME(typeZ)] = deserialized_operator;
}


void *udp_server_thread(void *arg)
{
    char recvbuf[4100] = {};
    struct sockaddr_in servaddr;
    socklen_t addr_len;
    long unsigned recv_len = 0;
    
    Topicvec *tpset = (Topicvec*) arg;

    if ((ser_sock = socket(PF_INET, SOCK_DGRAM, 0)) < 0)
    {
        ERR_EXIT("socket error");
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family      = AF_INET;
    servaddr.sin_port        = htons(RECV_PORT);
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(ser_sock, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
    {
        ERR_EXIT("bind error");
    }
    while (1)
    {
        addr_len = sizeof(servaddr);
        memset(recvbuf, '\0', sizeof(recvbuf));
        recv_len = recvfrom(ser_sock, recvbuf, sizeof(recvbuf), 0, (struct sockaddr *)&servaddr, &addr_len);
    
        if (recv_len <= 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            ERR_EXIT("recvfrom error");
        }
        else if(recv_len > 0)
        { 
            std::cout << "recv_len = " << recv_len << std::endl;

            printf("recv msg:\n");
            for (long unsigned i = 0; i < recv_len; i++) {
                printf("%d", recvbuf[i]);
            }
            printf("\n\n");

            std::vector<uint8_t> serialized_data;
            while (serialized_data.empty() == false) {serialized_data.pop_back();}

            for (long unsigned i = 0; i < recv_len; i++)
            {
                serialized_data.emplace_back(static_cast<uint8_t>(recvbuf[i]));
            }

            classformsg::msg::Msgdemo combmsg;
            MessageSerializer<classformsg::msg::Msgdemo>::deserialize(serialized_data, combmsg);

            for (const auto& typeptr: tpset->type_vec) {
                if (combmsg.type == typeptr.second && combmsg.topic == typeptr.third) {
                    for (const auto& item: tpset->topic_vec) {
                        if (abi::__cxa_demangle(item.type().name(), NULL, NULL, NULL) == typeptr.first) {
                            deserialize_map[combmsg.type]->try_type(item, combmsg); 
                        }
                    }
                }
            }

        }
    }
    close(ser_sock);
    return NULL;
}

NodeV2X2::NodeV2X2(const rclcpp::NodeOptions & options)
: Node("node_v2x", options)
{

    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.reliable();
    qos.deadline(rclcpp::Duration(1, 0));

    pub_pose = this->create_publisher<nav_msgs::msg::Odometry>("remote_posetwist_fusion", rclcpp::QoS(1));
    pub_object = this->create_publisher<autoware_auto_perception_msgs::msg::PredictedObjects>("remote_object_recognition", rclcpp::QoS(1));

    std::string pub_pose_topicname_original = "/localization/kinematic_state";
    std::string pub_object_topicname_original = "/perception/object_recognition/objects";

    topic_customer<nav_msgs::msg::Odometry> pub1(pub_pose, pub_pose_topicname_original);
    topic_customer<autoware_auto_perception_msgs::msg::PredictedObjects> pub2(pub_object, pub_object_topicname_original);

    Topicvec *pubset = new Topicvec();
    pubset->clear();
    pubset->add(pub1);
    pubset->add(pub2);

    register_type<nav_msgs::msg::Odometry>();
    register_type<autoware_auto_perception_msgs::msg::PredictedObjects>();

    pthread_t t2;

    int err = pthread_create(&t2, NULL, udp_server_thread, pubset);
    if (err != 0)
    {
        printf("thread_create Failed:%s\n", strerror(errno));
    }
    else
    {
        printf("thread_create success\n");
    }
    pthread_detach(t2);

    pause();

}

}  // namespace node_v2x

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(node_v2x::NodeV2X2)

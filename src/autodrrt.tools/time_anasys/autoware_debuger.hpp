    #ifndef AUTOWARE_DEBUGGER_H
    #define AUTOWARE_DEBUGGER_H
    #include "std_msgs/msg/string.hpp"
    #include <rclcpp/rclcpp.hpp>

   
        #define INIT_PUBLISH_DEBUGGER_MICRO rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_pub;
        #define INIT_STAMP_STRING std::string debuger_string;\
                long long timeStamp_debug_start = 0;
        #define INIT_EXTRA_STAMP_STRING std::string debuger_output_string;

        #define CREATE_PUBLISH_DEBUGGER_MICRO const rclcpp::QoS & qos_debug = rclcpp::QoS(rclcpp::KeepLast(1000));\
                debug_pub = this->create_publisher<std_msgs::msg::String>("/debug_topic_consumer", qos_debug);

        
        #define GET_STAMP(x) debuger_string = std::to_string(rclcpp::Time(x->header.stamp).seconds());
        #define GET_STAMP_VIA_ENTITY(x) debuger_string = std::to_string(rclcpp::Time(x.header.stamp).seconds());
        #define GET_STAMP_EXTRA(x) debuger_output_string = std::to_string(rclcpp::Time(x->header.stamp).seconds());
        #define GET_STAMP_VIA_ENTITY_EXTRA(x) debuger_output_string = std::to_string(rclcpp::Time(x.header.stamp).seconds());
        #define GET_STAMP_VIA_TIME_ENTITY_EXTRA(x) debuger_output_string = std::to_string(rclcpp::Time(x.stamp).seconds());
        
        #define SET_STAMP_IN_CALLBACK timeStamp_debug_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
       


        #define START_TO_PUBLISH_DEBUGGER_MICRO  auto timeStamp_debug =std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();\
                std::string namespace_(this->get_namespace());\ 
                std::string out_str(namespace_ + "/" + this->get_name() + ":" + std::to_string(timeStamp_debug));\
                RCLCPP_INFO(rclcpp::get_logger("debug_topic_consumer"), out_str.c_str());
              


        #define START_TO_PUBLISH_DEBUGGER_WITH_STMP_MICRO auto timeStamp_debug =std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();\
                std::string namespace_(this->get_namespace());\ 
                long long temp_res = timeStamp_debug - timeStamp_debug_start;\
                std::string result(std::to_string(temp_res));\
                std::string out_str(namespace_ + "/" + this->get_name() + ":" + std::to_string(timeStamp_debug) + ":" + result + ":" + debuger_string);\
                RCLCPP_INFO(rclcpp::get_logger("debug_topic_consumer"), out_str.c_str());
                // std::string out_str_compare(namespace_ + "/" + this->get_name() + ":" + std::to_string(timeStamp_debug) +":" + timeStamp_debug_start + ":" + debuger_string);\
                // RCLCPP_INFO(rclcpp::get_logger("debug_topic_node"), out_str_compare.c_str());



        #define START_TO_PUBLISH_DEBUGGER_WITH_STMP_EXTRA_MICRO  auto timeStamp_debug =std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();\
                std::string namespace_(this->get_namespace());\
                 long long temp_res = timeStamp_debug - timeStamp_debug_start;\
                std::string result(std::to_string(temp_res));\
                std::string out_str(namespace_ + "/" + this->get_name() + ":" + std::to_string(timeStamp_debug) + ":" + result + ":" + debuger_string  + ":" + debuger_output_string);\
                RCLCPP_INFO(rclcpp::get_logger("debug_topic_consumer"), out_str.c_str());
                // std::string out_str_compare(namespace_ + "/" + this->get_name() + ":" + std::to_string(timeStamp_debug) + ":" + timeStamp_debug_start + ":" + debuger_string  + ":" + debuger_output_string);\
                // RCLCPP_INFO(rclcpp::get_logger("debug_topic_node"), out_str_compare.c_str());


        #endif
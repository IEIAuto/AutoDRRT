// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dma_customer_msg:msg/Dmamessage.idl
// generated code does not contain a copyright notice

#ifndef DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__BUILDER_HPP_
#define DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dma_customer_msg/msg/detail/dmamessage__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dma_customer_msg
{

namespace msg
{

namespace builder
{

class Init_Dmamessage_header
{
public:
  explicit Init_Dmamessage_header(::dma_customer_msg::msg::Dmamessage & msg)
  : msg_(msg)
  {}
  ::dma_customer_msg::msg::Dmamessage header(::dma_customer_msg::msg::Dmamessage::_header_type arg)
  {
    msg_.header = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dma_customer_msg::msg::Dmamessage msg_;
};

class Init_Dmamessage_data
{
public:
  Init_Dmamessage_data()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Dmamessage_header data(::dma_customer_msg::msg::Dmamessage::_data_type arg)
  {
    msg_.data = std::move(arg);
    return Init_Dmamessage_header(msg_);
  }

private:
  ::dma_customer_msg::msg::Dmamessage msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dma_customer_msg::msg::Dmamessage>()
{
  return dma_customer_msg::msg::builder::Init_Dmamessage_data();
}

}  // namespace dma_customer_msg

#endif  // DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__BUILDER_HPP_

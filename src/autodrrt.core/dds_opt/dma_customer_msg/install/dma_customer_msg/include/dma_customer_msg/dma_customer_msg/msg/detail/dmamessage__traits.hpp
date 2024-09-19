// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dma_customer_msg:msg/Dmamessage.idl
// generated code does not contain a copyright notice

#ifndef DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__TRAITS_HPP_
#define DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dma_customer_msg/msg/detail/dmamessage__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace dma_customer_msg
{

namespace msg
{

inline void to_flow_style_yaml(
  const Dmamessage & msg,
  std::ostream & out)
{
  out << "{";
  // member: data
  {
    out << "data: ";
    rosidl_generator_traits::value_to_yaml(msg.data, out);
    out << ", ";
  }

  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Dmamessage & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: data
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "data: ";
    rosidl_generator_traits::value_to_yaml(msg.data, out);
    out << "\n";
  }

  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Dmamessage & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace dma_customer_msg

namespace rosidl_generator_traits
{

[[deprecated("use dma_customer_msg::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const dma_customer_msg::msg::Dmamessage & msg,
  std::ostream & out, size_t indentation = 0)
{
  dma_customer_msg::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dma_customer_msg::msg::to_yaml() instead")]]
inline std::string to_yaml(const dma_customer_msg::msg::Dmamessage & msg)
{
  return dma_customer_msg::msg::to_yaml(msg);
}

template<>
inline const char * data_type<dma_customer_msg::msg::Dmamessage>()
{
  return "dma_customer_msg::msg::Dmamessage";
}

template<>
inline const char * name<dma_customer_msg::msg::Dmamessage>()
{
  return "dma_customer_msg/msg/Dmamessage";
}

template<>
struct has_fixed_size<dma_customer_msg::msg::Dmamessage>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<dma_customer_msg::msg::Dmamessage>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<dma_customer_msg::msg::Dmamessage>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__TRAITS_HPP_

// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dma_customer_msg:msg/Dmamessage.idl
// generated code does not contain a copyright notice

#ifndef DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__STRUCT_HPP_
#define DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__dma_customer_msg__msg__Dmamessage __attribute__((deprecated))
#else
# define DEPRECATED__dma_customer_msg__msg__Dmamessage __declspec(deprecated)
#endif

namespace dma_customer_msg
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Dmamessage_
{
  using Type = Dmamessage_<ContainerAllocator>;

  explicit Dmamessage_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->data = 0ull;
    }
  }

  explicit Dmamessage_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->data = 0ull;
    }
  }

  // field types and members
  using _data_type =
    uint64_t;
  _data_type data;
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;

  // setters for named parameter idiom
  Type & set__data(
    const uint64_t & _arg)
  {
    this->data = _arg;
    return *this;
  }
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dma_customer_msg::msg::Dmamessage_<ContainerAllocator> *;
  using ConstRawPtr =
    const dma_customer_msg::msg::Dmamessage_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dma_customer_msg::msg::Dmamessage_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dma_customer_msg::msg::Dmamessage_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dma_customer_msg::msg::Dmamessage_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dma_customer_msg::msg::Dmamessage_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dma_customer_msg::msg::Dmamessage_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dma_customer_msg::msg::Dmamessage_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dma_customer_msg::msg::Dmamessage_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dma_customer_msg::msg::Dmamessage_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dma_customer_msg__msg__Dmamessage
    std::shared_ptr<dma_customer_msg::msg::Dmamessage_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dma_customer_msg__msg__Dmamessage
    std::shared_ptr<dma_customer_msg::msg::Dmamessage_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Dmamessage_ & other) const
  {
    if (this->data != other.data) {
      return false;
    }
    if (this->header != other.header) {
      return false;
    }
    return true;
  }
  bool operator!=(const Dmamessage_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Dmamessage_

// alias to use template instance with default allocator
using Dmamessage =
  dma_customer_msg::msg::Dmamessage_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace dma_customer_msg

#endif  // DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__STRUCT_HPP_

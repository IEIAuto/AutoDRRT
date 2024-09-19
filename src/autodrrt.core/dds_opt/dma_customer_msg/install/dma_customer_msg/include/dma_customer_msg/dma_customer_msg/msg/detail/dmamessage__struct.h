// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dma_customer_msg:msg/Dmamessage.idl
// generated code does not contain a copyright notice

#ifndef DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__STRUCT_H_
#define DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"

/// Struct defined in msg/Dmamessage in the package dma_customer_msg.
/**
  * dmamessage.msg
 */
typedef struct dma_customer_msg__msg__Dmamessage
{
  /// string type
  uint64_t data;
  std_msgs__msg__Header header;
} dma_customer_msg__msg__Dmamessage;

// Struct for a sequence of dma_customer_msg__msg__Dmamessage.
typedef struct dma_customer_msg__msg__Dmamessage__Sequence
{
  dma_customer_msg__msg__Dmamessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dma_customer_msg__msg__Dmamessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__STRUCT_H_

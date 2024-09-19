// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from dma_customer_msg:msg/Dmamessage.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "dma_customer_msg/msg/detail/dmamessage__rosidl_typesupport_introspection_c.h"
#include "dma_customer_msg/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "dma_customer_msg/msg/detail/dmamessage__functions.h"
#include "dma_customer_msg/msg/detail/dmamessage__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  dma_customer_msg__msg__Dmamessage__init(message_memory);
}

void dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_fini_function(void * message_memory)
{
  dma_customer_msg__msg__Dmamessage__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_message_member_array[2] = {
  {
    "data",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dma_customer_msg__msg__Dmamessage, data),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dma_customer_msg__msg__Dmamessage, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_message_members = {
  "dma_customer_msg__msg",  // message namespace
  "Dmamessage",  // message name
  2,  // number of fields
  sizeof(dma_customer_msg__msg__Dmamessage),
  dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_message_member_array,  // message members
  dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_init_function,  // function to initialize message memory (memory has to be allocated)
  dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_message_type_support_handle = {
  0,
  &dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_dma_customer_msg
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, dma_customer_msg, msg, Dmamessage)() {
  dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_message_type_support_handle.typesupport_identifier) {
    dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &dma_customer_msg__msg__Dmamessage__rosidl_typesupport_introspection_c__Dmamessage_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

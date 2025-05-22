// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from dma_customer_msg:msg/Dmamessage.idl
// generated code does not contain a copyright notice

#ifndef DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__FUNCTIONS_H_
#define DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "dma_customer_msg/msg/rosidl_generator_c__visibility_control.h"

#include "dma_customer_msg/msg/detail/dmamessage__struct.h"

/// Initialize msg/Dmamessage message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * dma_customer_msg__msg__Dmamessage
 * )) before or use
 * dma_customer_msg__msg__Dmamessage__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_dma_customer_msg
bool
dma_customer_msg__msg__Dmamessage__init(dma_customer_msg__msg__Dmamessage * msg);

/// Finalize msg/Dmamessage message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dma_customer_msg
void
dma_customer_msg__msg__Dmamessage__fini(dma_customer_msg__msg__Dmamessage * msg);

/// Create msg/Dmamessage message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * dma_customer_msg__msg__Dmamessage__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dma_customer_msg
dma_customer_msg__msg__Dmamessage *
dma_customer_msg__msg__Dmamessage__create();

/// Destroy msg/Dmamessage message.
/**
 * It calls
 * dma_customer_msg__msg__Dmamessage__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dma_customer_msg
void
dma_customer_msg__msg__Dmamessage__destroy(dma_customer_msg__msg__Dmamessage * msg);

/// Check for msg/Dmamessage message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dma_customer_msg
bool
dma_customer_msg__msg__Dmamessage__are_equal(const dma_customer_msg__msg__Dmamessage * lhs, const dma_customer_msg__msg__Dmamessage * rhs);

/// Copy a msg/Dmamessage message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_dma_customer_msg
bool
dma_customer_msg__msg__Dmamessage__copy(
  const dma_customer_msg__msg__Dmamessage * input,
  dma_customer_msg__msg__Dmamessage * output);

/// Initialize array of msg/Dmamessage messages.
/**
 * It allocates the memory for the number of elements and calls
 * dma_customer_msg__msg__Dmamessage__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_dma_customer_msg
bool
dma_customer_msg__msg__Dmamessage__Sequence__init(dma_customer_msg__msg__Dmamessage__Sequence * array, size_t size);

/// Finalize array of msg/Dmamessage messages.
/**
 * It calls
 * dma_customer_msg__msg__Dmamessage__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dma_customer_msg
void
dma_customer_msg__msg__Dmamessage__Sequence__fini(dma_customer_msg__msg__Dmamessage__Sequence * array);

/// Create array of msg/Dmamessage messages.
/**
 * It allocates the memory for the array and calls
 * dma_customer_msg__msg__Dmamessage__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dma_customer_msg
dma_customer_msg__msg__Dmamessage__Sequence *
dma_customer_msg__msg__Dmamessage__Sequence__create(size_t size);

/// Destroy array of msg/Dmamessage messages.
/**
 * It calls
 * dma_customer_msg__msg__Dmamessage__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dma_customer_msg
void
dma_customer_msg__msg__Dmamessage__Sequence__destroy(dma_customer_msg__msg__Dmamessage__Sequence * array);

/// Check for msg/Dmamessage message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dma_customer_msg
bool
dma_customer_msg__msg__Dmamessage__Sequence__are_equal(const dma_customer_msg__msg__Dmamessage__Sequence * lhs, const dma_customer_msg__msg__Dmamessage__Sequence * rhs);

/// Copy an array of msg/Dmamessage messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_dma_customer_msg
bool
dma_customer_msg__msg__Dmamessage__Sequence__copy(
  const dma_customer_msg__msg__Dmamessage__Sequence * input,
  dma_customer_msg__msg__Dmamessage__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // DMA_CUSTOMER_MSG__MSG__DETAIL__DMAMESSAGE__FUNCTIONS_H_

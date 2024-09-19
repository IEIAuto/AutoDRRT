// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from dma_customer_msg:msg/Dmamessage.idl
// generated code does not contain a copyright notice
#include "dma_customer_msg/msg/detail/dmamessage__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
dma_customer_msg__msg__Dmamessage__init(dma_customer_msg__msg__Dmamessage * msg)
{
  if (!msg) {
    return false;
  }
  // data
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    dma_customer_msg__msg__Dmamessage__fini(msg);
    return false;
  }
  return true;
}

void
dma_customer_msg__msg__Dmamessage__fini(dma_customer_msg__msg__Dmamessage * msg)
{
  if (!msg) {
    return;
  }
  // data
  // header
  std_msgs__msg__Header__fini(&msg->header);
}

bool
dma_customer_msg__msg__Dmamessage__are_equal(const dma_customer_msg__msg__Dmamessage * lhs, const dma_customer_msg__msg__Dmamessage * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // data
  if (lhs->data != rhs->data) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  return true;
}

bool
dma_customer_msg__msg__Dmamessage__copy(
  const dma_customer_msg__msg__Dmamessage * input,
  dma_customer_msg__msg__Dmamessage * output)
{
  if (!input || !output) {
    return false;
  }
  // data
  output->data = input->data;
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  return true;
}

dma_customer_msg__msg__Dmamessage *
dma_customer_msg__msg__Dmamessage__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dma_customer_msg__msg__Dmamessage * msg = (dma_customer_msg__msg__Dmamessage *)allocator.allocate(sizeof(dma_customer_msg__msg__Dmamessage), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dma_customer_msg__msg__Dmamessage));
  bool success = dma_customer_msg__msg__Dmamessage__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dma_customer_msg__msg__Dmamessage__destroy(dma_customer_msg__msg__Dmamessage * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dma_customer_msg__msg__Dmamessage__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dma_customer_msg__msg__Dmamessage__Sequence__init(dma_customer_msg__msg__Dmamessage__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dma_customer_msg__msg__Dmamessage * data = NULL;

  if (size) {
    data = (dma_customer_msg__msg__Dmamessage *)allocator.zero_allocate(size, sizeof(dma_customer_msg__msg__Dmamessage), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dma_customer_msg__msg__Dmamessage__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dma_customer_msg__msg__Dmamessage__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
dma_customer_msg__msg__Dmamessage__Sequence__fini(dma_customer_msg__msg__Dmamessage__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      dma_customer_msg__msg__Dmamessage__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

dma_customer_msg__msg__Dmamessage__Sequence *
dma_customer_msg__msg__Dmamessage__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dma_customer_msg__msg__Dmamessage__Sequence * array = (dma_customer_msg__msg__Dmamessage__Sequence *)allocator.allocate(sizeof(dma_customer_msg__msg__Dmamessage__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dma_customer_msg__msg__Dmamessage__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dma_customer_msg__msg__Dmamessage__Sequence__destroy(dma_customer_msg__msg__Dmamessage__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dma_customer_msg__msg__Dmamessage__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dma_customer_msg__msg__Dmamessage__Sequence__are_equal(const dma_customer_msg__msg__Dmamessage__Sequence * lhs, const dma_customer_msg__msg__Dmamessage__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dma_customer_msg__msg__Dmamessage__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dma_customer_msg__msg__Dmamessage__Sequence__copy(
  const dma_customer_msg__msg__Dmamessage__Sequence * input,
  dma_customer_msg__msg__Dmamessage__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dma_customer_msg__msg__Dmamessage);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dma_customer_msg__msg__Dmamessage * data =
      (dma_customer_msg__msg__Dmamessage *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dma_customer_msg__msg__Dmamessage__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dma_customer_msg__msg__Dmamessage__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dma_customer_msg__msg__Dmamessage__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

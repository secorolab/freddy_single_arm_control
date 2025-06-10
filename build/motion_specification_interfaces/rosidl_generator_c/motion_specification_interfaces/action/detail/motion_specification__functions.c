// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from motion_specification_interfaces:action/MotionSpecification.idl
// generated code does not contain a copyright notice
#include "motion_specification_interfaces/action/detail/motion_specification__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `motion_specification`
#include "rosidl_runtime_c/string_functions.h"

bool
motion_specification_interfaces__action__MotionSpecification_Goal__init(motion_specification_interfaces__action__MotionSpecification_Goal * msg)
{
  if (!msg) {
    return false;
  }
  // motion_specification
  if (!rosidl_runtime_c__String__init(&msg->motion_specification)) {
    motion_specification_interfaces__action__MotionSpecification_Goal__fini(msg);
    return false;
  }
  return true;
}

void
motion_specification_interfaces__action__MotionSpecification_Goal__fini(motion_specification_interfaces__action__MotionSpecification_Goal * msg)
{
  if (!msg) {
    return;
  }
  // motion_specification
  rosidl_runtime_c__String__fini(&msg->motion_specification);
}

bool
motion_specification_interfaces__action__MotionSpecification_Goal__are_equal(const motion_specification_interfaces__action__MotionSpecification_Goal * lhs, const motion_specification_interfaces__action__MotionSpecification_Goal * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // motion_specification
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->motion_specification), &(rhs->motion_specification)))
  {
    return false;
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_Goal__copy(
  const motion_specification_interfaces__action__MotionSpecification_Goal * input,
  motion_specification_interfaces__action__MotionSpecification_Goal * output)
{
  if (!input || !output) {
    return false;
  }
  // motion_specification
  if (!rosidl_runtime_c__String__copy(
      &(input->motion_specification), &(output->motion_specification)))
  {
    return false;
  }
  return true;
}

motion_specification_interfaces__action__MotionSpecification_Goal *
motion_specification_interfaces__action__MotionSpecification_Goal__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_Goal * msg = (motion_specification_interfaces__action__MotionSpecification_Goal *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_Goal), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(motion_specification_interfaces__action__MotionSpecification_Goal));
  bool success = motion_specification_interfaces__action__MotionSpecification_Goal__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
motion_specification_interfaces__action__MotionSpecification_Goal__destroy(motion_specification_interfaces__action__MotionSpecification_Goal * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    motion_specification_interfaces__action__MotionSpecification_Goal__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
motion_specification_interfaces__action__MotionSpecification_Goal__Sequence__init(motion_specification_interfaces__action__MotionSpecification_Goal__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_Goal * data = NULL;

  if (size) {
    data = (motion_specification_interfaces__action__MotionSpecification_Goal *)allocator.zero_allocate(size, sizeof(motion_specification_interfaces__action__MotionSpecification_Goal), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = motion_specification_interfaces__action__MotionSpecification_Goal__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        motion_specification_interfaces__action__MotionSpecification_Goal__fini(&data[i - 1]);
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
motion_specification_interfaces__action__MotionSpecification_Goal__Sequence__fini(motion_specification_interfaces__action__MotionSpecification_Goal__Sequence * array)
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
      motion_specification_interfaces__action__MotionSpecification_Goal__fini(&array->data[i]);
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

motion_specification_interfaces__action__MotionSpecification_Goal__Sequence *
motion_specification_interfaces__action__MotionSpecification_Goal__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_Goal__Sequence * array = (motion_specification_interfaces__action__MotionSpecification_Goal__Sequence *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_Goal__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = motion_specification_interfaces__action__MotionSpecification_Goal__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
motion_specification_interfaces__action__MotionSpecification_Goal__Sequence__destroy(motion_specification_interfaces__action__MotionSpecification_Goal__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    motion_specification_interfaces__action__MotionSpecification_Goal__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
motion_specification_interfaces__action__MotionSpecification_Goal__Sequence__are_equal(const motion_specification_interfaces__action__MotionSpecification_Goal__Sequence * lhs, const motion_specification_interfaces__action__MotionSpecification_Goal__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_Goal__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_Goal__Sequence__copy(
  const motion_specification_interfaces__action__MotionSpecification_Goal__Sequence * input,
  motion_specification_interfaces__action__MotionSpecification_Goal__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(motion_specification_interfaces__action__MotionSpecification_Goal);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    motion_specification_interfaces__action__MotionSpecification_Goal * data =
      (motion_specification_interfaces__action__MotionSpecification_Goal *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!motion_specification_interfaces__action__MotionSpecification_Goal__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          motion_specification_interfaces__action__MotionSpecification_Goal__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_Goal__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
motion_specification_interfaces__action__MotionSpecification_Result__init(motion_specification_interfaces__action__MotionSpecification_Result * msg)
{
  if (!msg) {
    return false;
  }
  // motion_successful
  return true;
}

void
motion_specification_interfaces__action__MotionSpecification_Result__fini(motion_specification_interfaces__action__MotionSpecification_Result * msg)
{
  if (!msg) {
    return;
  }
  // motion_successful
}

bool
motion_specification_interfaces__action__MotionSpecification_Result__are_equal(const motion_specification_interfaces__action__MotionSpecification_Result * lhs, const motion_specification_interfaces__action__MotionSpecification_Result * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // motion_successful
  if (lhs->motion_successful != rhs->motion_successful) {
    return false;
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_Result__copy(
  const motion_specification_interfaces__action__MotionSpecification_Result * input,
  motion_specification_interfaces__action__MotionSpecification_Result * output)
{
  if (!input || !output) {
    return false;
  }
  // motion_successful
  output->motion_successful = input->motion_successful;
  return true;
}

motion_specification_interfaces__action__MotionSpecification_Result *
motion_specification_interfaces__action__MotionSpecification_Result__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_Result * msg = (motion_specification_interfaces__action__MotionSpecification_Result *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_Result), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(motion_specification_interfaces__action__MotionSpecification_Result));
  bool success = motion_specification_interfaces__action__MotionSpecification_Result__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
motion_specification_interfaces__action__MotionSpecification_Result__destroy(motion_specification_interfaces__action__MotionSpecification_Result * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    motion_specification_interfaces__action__MotionSpecification_Result__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
motion_specification_interfaces__action__MotionSpecification_Result__Sequence__init(motion_specification_interfaces__action__MotionSpecification_Result__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_Result * data = NULL;

  if (size) {
    data = (motion_specification_interfaces__action__MotionSpecification_Result *)allocator.zero_allocate(size, sizeof(motion_specification_interfaces__action__MotionSpecification_Result), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = motion_specification_interfaces__action__MotionSpecification_Result__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        motion_specification_interfaces__action__MotionSpecification_Result__fini(&data[i - 1]);
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
motion_specification_interfaces__action__MotionSpecification_Result__Sequence__fini(motion_specification_interfaces__action__MotionSpecification_Result__Sequence * array)
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
      motion_specification_interfaces__action__MotionSpecification_Result__fini(&array->data[i]);
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

motion_specification_interfaces__action__MotionSpecification_Result__Sequence *
motion_specification_interfaces__action__MotionSpecification_Result__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_Result__Sequence * array = (motion_specification_interfaces__action__MotionSpecification_Result__Sequence *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_Result__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = motion_specification_interfaces__action__MotionSpecification_Result__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
motion_specification_interfaces__action__MotionSpecification_Result__Sequence__destroy(motion_specification_interfaces__action__MotionSpecification_Result__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    motion_specification_interfaces__action__MotionSpecification_Result__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
motion_specification_interfaces__action__MotionSpecification_Result__Sequence__are_equal(const motion_specification_interfaces__action__MotionSpecification_Result__Sequence * lhs, const motion_specification_interfaces__action__MotionSpecification_Result__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_Result__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_Result__Sequence__copy(
  const motion_specification_interfaces__action__MotionSpecification_Result__Sequence * input,
  motion_specification_interfaces__action__MotionSpecification_Result__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(motion_specification_interfaces__action__MotionSpecification_Result);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    motion_specification_interfaces__action__MotionSpecification_Result * data =
      (motion_specification_interfaces__action__MotionSpecification_Result *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!motion_specification_interfaces__action__MotionSpecification_Result__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          motion_specification_interfaces__action__MotionSpecification_Result__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_Result__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `tcp_position`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
motion_specification_interfaces__action__MotionSpecification_Feedback__init(motion_specification_interfaces__action__MotionSpecification_Feedback * msg)
{
  if (!msg) {
    return false;
  }
  // tcp_position
  if (!rosidl_runtime_c__float__Sequence__init(&msg->tcp_position, 0)) {
    motion_specification_interfaces__action__MotionSpecification_Feedback__fini(msg);
    return false;
  }
  return true;
}

void
motion_specification_interfaces__action__MotionSpecification_Feedback__fini(motion_specification_interfaces__action__MotionSpecification_Feedback * msg)
{
  if (!msg) {
    return;
  }
  // tcp_position
  rosidl_runtime_c__float__Sequence__fini(&msg->tcp_position);
}

bool
motion_specification_interfaces__action__MotionSpecification_Feedback__are_equal(const motion_specification_interfaces__action__MotionSpecification_Feedback * lhs, const motion_specification_interfaces__action__MotionSpecification_Feedback * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // tcp_position
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->tcp_position), &(rhs->tcp_position)))
  {
    return false;
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_Feedback__copy(
  const motion_specification_interfaces__action__MotionSpecification_Feedback * input,
  motion_specification_interfaces__action__MotionSpecification_Feedback * output)
{
  if (!input || !output) {
    return false;
  }
  // tcp_position
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->tcp_position), &(output->tcp_position)))
  {
    return false;
  }
  return true;
}

motion_specification_interfaces__action__MotionSpecification_Feedback *
motion_specification_interfaces__action__MotionSpecification_Feedback__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_Feedback * msg = (motion_specification_interfaces__action__MotionSpecification_Feedback *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_Feedback), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(motion_specification_interfaces__action__MotionSpecification_Feedback));
  bool success = motion_specification_interfaces__action__MotionSpecification_Feedback__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
motion_specification_interfaces__action__MotionSpecification_Feedback__destroy(motion_specification_interfaces__action__MotionSpecification_Feedback * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    motion_specification_interfaces__action__MotionSpecification_Feedback__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence__init(motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_Feedback * data = NULL;

  if (size) {
    data = (motion_specification_interfaces__action__MotionSpecification_Feedback *)allocator.zero_allocate(size, sizeof(motion_specification_interfaces__action__MotionSpecification_Feedback), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = motion_specification_interfaces__action__MotionSpecification_Feedback__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        motion_specification_interfaces__action__MotionSpecification_Feedback__fini(&data[i - 1]);
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
motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence__fini(motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence * array)
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
      motion_specification_interfaces__action__MotionSpecification_Feedback__fini(&array->data[i]);
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

motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence *
motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence * array = (motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence__destroy(motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence__are_equal(const motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence * lhs, const motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_Feedback__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence__copy(
  const motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence * input,
  motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(motion_specification_interfaces__action__MotionSpecification_Feedback);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    motion_specification_interfaces__action__MotionSpecification_Feedback * data =
      (motion_specification_interfaces__action__MotionSpecification_Feedback *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!motion_specification_interfaces__action__MotionSpecification_Feedback__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          motion_specification_interfaces__action__MotionSpecification_Feedback__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_Feedback__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `goal_id`
#include "unique_identifier_msgs/msg/detail/uuid__functions.h"
// Member `goal`
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__init(motion_specification_interfaces__action__MotionSpecification_SendGoal_Request * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__fini(msg);
    return false;
  }
  // goal
  if (!motion_specification_interfaces__action__MotionSpecification_Goal__init(&msg->goal)) {
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__fini(msg);
    return false;
  }
  return true;
}

void
motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__fini(motion_specification_interfaces__action__MotionSpecification_SendGoal_Request * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
  // goal
  motion_specification_interfaces__action__MotionSpecification_Goal__fini(&msg->goal);
}

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__are_equal(const motion_specification_interfaces__action__MotionSpecification_SendGoal_Request * lhs, const motion_specification_interfaces__action__MotionSpecification_SendGoal_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__are_equal(
      &(lhs->goal_id), &(rhs->goal_id)))
  {
    return false;
  }
  // goal
  if (!motion_specification_interfaces__action__MotionSpecification_Goal__are_equal(
      &(lhs->goal), &(rhs->goal)))
  {
    return false;
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__copy(
  const motion_specification_interfaces__action__MotionSpecification_SendGoal_Request * input,
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__copy(
      &(input->goal_id), &(output->goal_id)))
  {
    return false;
  }
  // goal
  if (!motion_specification_interfaces__action__MotionSpecification_Goal__copy(
      &(input->goal), &(output->goal)))
  {
    return false;
  }
  return true;
}

motion_specification_interfaces__action__MotionSpecification_SendGoal_Request *
motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Request * msg = (motion_specification_interfaces__action__MotionSpecification_SendGoal_Request *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Request));
  bool success = motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__destroy(motion_specification_interfaces__action__MotionSpecification_SendGoal_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence__init(motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Request * data = NULL;

  if (size) {
    data = (motion_specification_interfaces__action__MotionSpecification_SendGoal_Request *)allocator.zero_allocate(size, sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__fini(&data[i - 1]);
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
motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence__fini(motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence * array)
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
      motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__fini(&array->data[i]);
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

motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence *
motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence * array = (motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence__destroy(motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence__are_equal(const motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence * lhs, const motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence__copy(
  const motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence * input,
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Request * data =
      (motion_specification_interfaces__action__MotionSpecification_SendGoal_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__init(motion_specification_interfaces__action__MotionSpecification_SendGoal_Response * msg)
{
  if (!msg) {
    return false;
  }
  // accepted
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__fini(msg);
    return false;
  }
  return true;
}

void
motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__fini(motion_specification_interfaces__action__MotionSpecification_SendGoal_Response * msg)
{
  if (!msg) {
    return;
  }
  // accepted
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
}

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__are_equal(const motion_specification_interfaces__action__MotionSpecification_SendGoal_Response * lhs, const motion_specification_interfaces__action__MotionSpecification_SendGoal_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // accepted
  if (lhs->accepted != rhs->accepted) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__copy(
  const motion_specification_interfaces__action__MotionSpecification_SendGoal_Response * input,
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // accepted
  output->accepted = input->accepted;
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  return true;
}

motion_specification_interfaces__action__MotionSpecification_SendGoal_Response *
motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Response * msg = (motion_specification_interfaces__action__MotionSpecification_SendGoal_Response *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Response));
  bool success = motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__destroy(motion_specification_interfaces__action__MotionSpecification_SendGoal_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence__init(motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Response * data = NULL;

  if (size) {
    data = (motion_specification_interfaces__action__MotionSpecification_SendGoal_Response *)allocator.zero_allocate(size, sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__fini(&data[i - 1]);
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
motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence__fini(motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence * array)
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
      motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__fini(&array->data[i]);
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

motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence *
motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence * array = (motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence__destroy(motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence__are_equal(const motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence * lhs, const motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence__copy(
  const motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence * input,
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Response * data =
      (motion_specification_interfaces__action__MotionSpecification_SendGoal_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `info`
#include "service_msgs/msg/detail/service_event_info__functions.h"
// Member `request`
// Member `response`
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__init(motion_specification_interfaces__action__MotionSpecification_SendGoal_Event * msg)
{
  if (!msg) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__init(&msg->info)) {
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__fini(msg);
    return false;
  }
  // request
  if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence__init(&msg->request, 0)) {
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__fini(msg);
    return false;
  }
  // response
  if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence__init(&msg->response, 0)) {
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__fini(msg);
    return false;
  }
  return true;
}

void
motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__fini(motion_specification_interfaces__action__MotionSpecification_SendGoal_Event * msg)
{
  if (!msg) {
    return;
  }
  // info
  service_msgs__msg__ServiceEventInfo__fini(&msg->info);
  // request
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence__fini(&msg->request);
  // response
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence__fini(&msg->response);
}

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__are_equal(const motion_specification_interfaces__action__MotionSpecification_SendGoal_Event * lhs, const motion_specification_interfaces__action__MotionSpecification_SendGoal_Event * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__are_equal(
      &(lhs->info), &(rhs->info)))
  {
    return false;
  }
  // request
  if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence__are_equal(
      &(lhs->request), &(rhs->request)))
  {
    return false;
  }
  // response
  if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence__are_equal(
      &(lhs->response), &(rhs->response)))
  {
    return false;
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__copy(
  const motion_specification_interfaces__action__MotionSpecification_SendGoal_Event * input,
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Event * output)
{
  if (!input || !output) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__copy(
      &(input->info), &(output->info)))
  {
    return false;
  }
  // request
  if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence__copy(
      &(input->request), &(output->request)))
  {
    return false;
  }
  // response
  if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence__copy(
      &(input->response), &(output->response)))
  {
    return false;
  }
  return true;
}

motion_specification_interfaces__action__MotionSpecification_SendGoal_Event *
motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Event * msg = (motion_specification_interfaces__action__MotionSpecification_SendGoal_Event *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Event), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Event));
  bool success = motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__destroy(motion_specification_interfaces__action__MotionSpecification_SendGoal_Event * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence__init(motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Event * data = NULL;

  if (size) {
    data = (motion_specification_interfaces__action__MotionSpecification_SendGoal_Event *)allocator.zero_allocate(size, sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Event), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__fini(&data[i - 1]);
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
motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence__fini(motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence * array)
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
      motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__fini(&array->data[i]);
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

motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence *
motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence * array = (motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence__destroy(motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence__are_equal(const motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence * lhs, const motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence__copy(
  const motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence * input,
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(motion_specification_interfaces__action__MotionSpecification_SendGoal_Event);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    motion_specification_interfaces__action__MotionSpecification_SendGoal_Event * data =
      (motion_specification_interfaces__action__MotionSpecification_SendGoal_Event *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `goal_id`
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__functions.h"

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Request__init(motion_specification_interfaces__action__MotionSpecification_GetResult_Request * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    motion_specification_interfaces__action__MotionSpecification_GetResult_Request__fini(msg);
    return false;
  }
  return true;
}

void
motion_specification_interfaces__action__MotionSpecification_GetResult_Request__fini(motion_specification_interfaces__action__MotionSpecification_GetResult_Request * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
}

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Request__are_equal(const motion_specification_interfaces__action__MotionSpecification_GetResult_Request * lhs, const motion_specification_interfaces__action__MotionSpecification_GetResult_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__are_equal(
      &(lhs->goal_id), &(rhs->goal_id)))
  {
    return false;
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Request__copy(
  const motion_specification_interfaces__action__MotionSpecification_GetResult_Request * input,
  motion_specification_interfaces__action__MotionSpecification_GetResult_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__copy(
      &(input->goal_id), &(output->goal_id)))
  {
    return false;
  }
  return true;
}

motion_specification_interfaces__action__MotionSpecification_GetResult_Request *
motion_specification_interfaces__action__MotionSpecification_GetResult_Request__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_GetResult_Request * msg = (motion_specification_interfaces__action__MotionSpecification_GetResult_Request *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Request));
  bool success = motion_specification_interfaces__action__MotionSpecification_GetResult_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
motion_specification_interfaces__action__MotionSpecification_GetResult_Request__destroy(motion_specification_interfaces__action__MotionSpecification_GetResult_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    motion_specification_interfaces__action__MotionSpecification_GetResult_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence__init(motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_GetResult_Request * data = NULL;

  if (size) {
    data = (motion_specification_interfaces__action__MotionSpecification_GetResult_Request *)allocator.zero_allocate(size, sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = motion_specification_interfaces__action__MotionSpecification_GetResult_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        motion_specification_interfaces__action__MotionSpecification_GetResult_Request__fini(&data[i - 1]);
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
motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence__fini(motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence * array)
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
      motion_specification_interfaces__action__MotionSpecification_GetResult_Request__fini(&array->data[i]);
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

motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence *
motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence * array = (motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence__destroy(motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence__are_equal(const motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence * lhs, const motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence__copy(
  const motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence * input,
  motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    motion_specification_interfaces__action__MotionSpecification_GetResult_Request * data =
      (motion_specification_interfaces__action__MotionSpecification_GetResult_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          motion_specification_interfaces__action__MotionSpecification_GetResult_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `result`
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Response__init(motion_specification_interfaces__action__MotionSpecification_GetResult_Response * msg)
{
  if (!msg) {
    return false;
  }
  // status
  // result
  if (!motion_specification_interfaces__action__MotionSpecification_Result__init(&msg->result)) {
    motion_specification_interfaces__action__MotionSpecification_GetResult_Response__fini(msg);
    return false;
  }
  return true;
}

void
motion_specification_interfaces__action__MotionSpecification_GetResult_Response__fini(motion_specification_interfaces__action__MotionSpecification_GetResult_Response * msg)
{
  if (!msg) {
    return;
  }
  // status
  // result
  motion_specification_interfaces__action__MotionSpecification_Result__fini(&msg->result);
}

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Response__are_equal(const motion_specification_interfaces__action__MotionSpecification_GetResult_Response * lhs, const motion_specification_interfaces__action__MotionSpecification_GetResult_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // status
  if (lhs->status != rhs->status) {
    return false;
  }
  // result
  if (!motion_specification_interfaces__action__MotionSpecification_Result__are_equal(
      &(lhs->result), &(rhs->result)))
  {
    return false;
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Response__copy(
  const motion_specification_interfaces__action__MotionSpecification_GetResult_Response * input,
  motion_specification_interfaces__action__MotionSpecification_GetResult_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // status
  output->status = input->status;
  // result
  if (!motion_specification_interfaces__action__MotionSpecification_Result__copy(
      &(input->result), &(output->result)))
  {
    return false;
  }
  return true;
}

motion_specification_interfaces__action__MotionSpecification_GetResult_Response *
motion_specification_interfaces__action__MotionSpecification_GetResult_Response__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_GetResult_Response * msg = (motion_specification_interfaces__action__MotionSpecification_GetResult_Response *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Response));
  bool success = motion_specification_interfaces__action__MotionSpecification_GetResult_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
motion_specification_interfaces__action__MotionSpecification_GetResult_Response__destroy(motion_specification_interfaces__action__MotionSpecification_GetResult_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    motion_specification_interfaces__action__MotionSpecification_GetResult_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence__init(motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_GetResult_Response * data = NULL;

  if (size) {
    data = (motion_specification_interfaces__action__MotionSpecification_GetResult_Response *)allocator.zero_allocate(size, sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = motion_specification_interfaces__action__MotionSpecification_GetResult_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        motion_specification_interfaces__action__MotionSpecification_GetResult_Response__fini(&data[i - 1]);
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
motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence__fini(motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence * array)
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
      motion_specification_interfaces__action__MotionSpecification_GetResult_Response__fini(&array->data[i]);
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

motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence *
motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence * array = (motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence__destroy(motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence__are_equal(const motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence * lhs, const motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence__copy(
  const motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence * input,
  motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    motion_specification_interfaces__action__MotionSpecification_GetResult_Response * data =
      (motion_specification_interfaces__action__MotionSpecification_GetResult_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          motion_specification_interfaces__action__MotionSpecification_GetResult_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `info`
// already included above
// #include "service_msgs/msg/detail/service_event_info__functions.h"
// Member `request`
// Member `response`
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Event__init(motion_specification_interfaces__action__MotionSpecification_GetResult_Event * msg)
{
  if (!msg) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__init(&msg->info)) {
    motion_specification_interfaces__action__MotionSpecification_GetResult_Event__fini(msg);
    return false;
  }
  // request
  if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence__init(&msg->request, 0)) {
    motion_specification_interfaces__action__MotionSpecification_GetResult_Event__fini(msg);
    return false;
  }
  // response
  if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence__init(&msg->response, 0)) {
    motion_specification_interfaces__action__MotionSpecification_GetResult_Event__fini(msg);
    return false;
  }
  return true;
}

void
motion_specification_interfaces__action__MotionSpecification_GetResult_Event__fini(motion_specification_interfaces__action__MotionSpecification_GetResult_Event * msg)
{
  if (!msg) {
    return;
  }
  // info
  service_msgs__msg__ServiceEventInfo__fini(&msg->info);
  // request
  motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence__fini(&msg->request);
  // response
  motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence__fini(&msg->response);
}

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Event__are_equal(const motion_specification_interfaces__action__MotionSpecification_GetResult_Event * lhs, const motion_specification_interfaces__action__MotionSpecification_GetResult_Event * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__are_equal(
      &(lhs->info), &(rhs->info)))
  {
    return false;
  }
  // request
  if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence__are_equal(
      &(lhs->request), &(rhs->request)))
  {
    return false;
  }
  // response
  if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence__are_equal(
      &(lhs->response), &(rhs->response)))
  {
    return false;
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Event__copy(
  const motion_specification_interfaces__action__MotionSpecification_GetResult_Event * input,
  motion_specification_interfaces__action__MotionSpecification_GetResult_Event * output)
{
  if (!input || !output) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__copy(
      &(input->info), &(output->info)))
  {
    return false;
  }
  // request
  if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence__copy(
      &(input->request), &(output->request)))
  {
    return false;
  }
  // response
  if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence__copy(
      &(input->response), &(output->response)))
  {
    return false;
  }
  return true;
}

motion_specification_interfaces__action__MotionSpecification_GetResult_Event *
motion_specification_interfaces__action__MotionSpecification_GetResult_Event__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_GetResult_Event * msg = (motion_specification_interfaces__action__MotionSpecification_GetResult_Event *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Event), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Event));
  bool success = motion_specification_interfaces__action__MotionSpecification_GetResult_Event__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
motion_specification_interfaces__action__MotionSpecification_GetResult_Event__destroy(motion_specification_interfaces__action__MotionSpecification_GetResult_Event * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    motion_specification_interfaces__action__MotionSpecification_GetResult_Event__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence__init(motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_GetResult_Event * data = NULL;

  if (size) {
    data = (motion_specification_interfaces__action__MotionSpecification_GetResult_Event *)allocator.zero_allocate(size, sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Event), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = motion_specification_interfaces__action__MotionSpecification_GetResult_Event__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        motion_specification_interfaces__action__MotionSpecification_GetResult_Event__fini(&data[i - 1]);
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
motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence__fini(motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence * array)
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
      motion_specification_interfaces__action__MotionSpecification_GetResult_Event__fini(&array->data[i]);
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

motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence *
motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence * array = (motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence__destroy(motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence__are_equal(const motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence * lhs, const motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Event__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence__copy(
  const motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence * input,
  motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(motion_specification_interfaces__action__MotionSpecification_GetResult_Event);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    motion_specification_interfaces__action__MotionSpecification_GetResult_Event * data =
      (motion_specification_interfaces__action__MotionSpecification_GetResult_Event *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Event__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          motion_specification_interfaces__action__MotionSpecification_GetResult_Event__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_GetResult_Event__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `goal_id`
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__functions.h"
// Member `feedback`
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"

bool
motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__init(motion_specification_interfaces__action__MotionSpecification_FeedbackMessage * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__fini(msg);
    return false;
  }
  // feedback
  if (!motion_specification_interfaces__action__MotionSpecification_Feedback__init(&msg->feedback)) {
    motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__fini(msg);
    return false;
  }
  return true;
}

void
motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__fini(motion_specification_interfaces__action__MotionSpecification_FeedbackMessage * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
  // feedback
  motion_specification_interfaces__action__MotionSpecification_Feedback__fini(&msg->feedback);
}

bool
motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__are_equal(const motion_specification_interfaces__action__MotionSpecification_FeedbackMessage * lhs, const motion_specification_interfaces__action__MotionSpecification_FeedbackMessage * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__are_equal(
      &(lhs->goal_id), &(rhs->goal_id)))
  {
    return false;
  }
  // feedback
  if (!motion_specification_interfaces__action__MotionSpecification_Feedback__are_equal(
      &(lhs->feedback), &(rhs->feedback)))
  {
    return false;
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__copy(
  const motion_specification_interfaces__action__MotionSpecification_FeedbackMessage * input,
  motion_specification_interfaces__action__MotionSpecification_FeedbackMessage * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__copy(
      &(input->goal_id), &(output->goal_id)))
  {
    return false;
  }
  // feedback
  if (!motion_specification_interfaces__action__MotionSpecification_Feedback__copy(
      &(input->feedback), &(output->feedback)))
  {
    return false;
  }
  return true;
}

motion_specification_interfaces__action__MotionSpecification_FeedbackMessage *
motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_FeedbackMessage * msg = (motion_specification_interfaces__action__MotionSpecification_FeedbackMessage *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_FeedbackMessage), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(motion_specification_interfaces__action__MotionSpecification_FeedbackMessage));
  bool success = motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__destroy(motion_specification_interfaces__action__MotionSpecification_FeedbackMessage * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence__init(motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_FeedbackMessage * data = NULL;

  if (size) {
    data = (motion_specification_interfaces__action__MotionSpecification_FeedbackMessage *)allocator.zero_allocate(size, sizeof(motion_specification_interfaces__action__MotionSpecification_FeedbackMessage), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__fini(&data[i - 1]);
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
motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence__fini(motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence * array)
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
      motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__fini(&array->data[i]);
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

motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence *
motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence * array = (motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence *)allocator.allocate(sizeof(motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence__destroy(motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence__are_equal(const motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence * lhs, const motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence__copy(
  const motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence * input,
  motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(motion_specification_interfaces__action__MotionSpecification_FeedbackMessage);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    motion_specification_interfaces__action__MotionSpecification_FeedbackMessage * data =
      (motion_specification_interfaces__action__MotionSpecification_FeedbackMessage *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from motion_specification_interfaces:action/MotionSpecification.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "motion_specification_interfaces/action/motion_specification.h"


#ifndef MOTION_SPECIFICATION_INTERFACES__ACTION__DETAIL__MOTION_SPECIFICATION__STRUCT_H_
#define MOTION_SPECIFICATION_INTERFACES__ACTION__DETAIL__MOTION_SPECIFICATION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'motion_specification'
#include "rosidl_runtime_c/string.h"

/// Struct defined in action/MotionSpecification in the package motion_specification_interfaces.
typedef struct motion_specification_interfaces__action__MotionSpecification_Goal
{
  rosidl_runtime_c__String motion_specification;
} motion_specification_interfaces__action__MotionSpecification_Goal;

// Struct for a sequence of motion_specification_interfaces__action__MotionSpecification_Goal.
typedef struct motion_specification_interfaces__action__MotionSpecification_Goal__Sequence
{
  motion_specification_interfaces__action__MotionSpecification_Goal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} motion_specification_interfaces__action__MotionSpecification_Goal__Sequence;

// Constants defined in the message

/// Struct defined in action/MotionSpecification in the package motion_specification_interfaces.
typedef struct motion_specification_interfaces__action__MotionSpecification_Result
{
  bool motion_successful;
} motion_specification_interfaces__action__MotionSpecification_Result;

// Struct for a sequence of motion_specification_interfaces__action__MotionSpecification_Result.
typedef struct motion_specification_interfaces__action__MotionSpecification_Result__Sequence
{
  motion_specification_interfaces__action__MotionSpecification_Result * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} motion_specification_interfaces__action__MotionSpecification_Result__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'tcp_position'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in action/MotionSpecification in the package motion_specification_interfaces.
typedef struct motion_specification_interfaces__action__MotionSpecification_Feedback
{
  rosidl_runtime_c__float__Sequence tcp_position;
} motion_specification_interfaces__action__MotionSpecification_Feedback;

// Struct for a sequence of motion_specification_interfaces__action__MotionSpecification_Feedback.
typedef struct motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence
{
  motion_specification_interfaces__action__MotionSpecification_Feedback * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} motion_specification_interfaces__action__MotionSpecification_Feedback__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'goal'
#include "motion_specification_interfaces/action/detail/motion_specification__struct.h"

/// Struct defined in action/MotionSpecification in the package motion_specification_interfaces.
typedef struct motion_specification_interfaces__action__MotionSpecification_SendGoal_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
  motion_specification_interfaces__action__MotionSpecification_Goal goal;
} motion_specification_interfaces__action__MotionSpecification_SendGoal_Request;

// Struct for a sequence of motion_specification_interfaces__action__MotionSpecification_SendGoal_Request.
typedef struct motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence
{
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in action/MotionSpecification in the package motion_specification_interfaces.
typedef struct motion_specification_interfaces__action__MotionSpecification_SendGoal_Response
{
  bool accepted;
  builtin_interfaces__msg__Time stamp;
} motion_specification_interfaces__action__MotionSpecification_SendGoal_Response;

// Struct for a sequence of motion_specification_interfaces__action__MotionSpecification_SendGoal_Response.
typedef struct motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence
{
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__request__MAX_SIZE = 1
};
// response
enum
{
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__response__MAX_SIZE = 1
};

/// Struct defined in action/MotionSpecification in the package motion_specification_interfaces.
typedef struct motion_specification_interfaces__action__MotionSpecification_SendGoal_Event
{
  service_msgs__msg__ServiceEventInfo info;
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__Sequence request;
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__Sequence response;
} motion_specification_interfaces__action__MotionSpecification_SendGoal_Event;

// Struct for a sequence of motion_specification_interfaces__action__MotionSpecification_SendGoal_Event.
typedef struct motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence
{
  motion_specification_interfaces__action__MotionSpecification_SendGoal_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"

/// Struct defined in action/MotionSpecification in the package motion_specification_interfaces.
typedef struct motion_specification_interfaces__action__MotionSpecification_GetResult_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
} motion_specification_interfaces__action__MotionSpecification_GetResult_Request;

// Struct for a sequence of motion_specification_interfaces__action__MotionSpecification_GetResult_Request.
typedef struct motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence
{
  motion_specification_interfaces__action__MotionSpecification_GetResult_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'result'
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__struct.h"

/// Struct defined in action/MotionSpecification in the package motion_specification_interfaces.
typedef struct motion_specification_interfaces__action__MotionSpecification_GetResult_Response
{
  int8_t status;
  motion_specification_interfaces__action__MotionSpecification_Result result;
} motion_specification_interfaces__action__MotionSpecification_GetResult_Response;

// Struct for a sequence of motion_specification_interfaces__action__MotionSpecification_GetResult_Response.
typedef struct motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence
{
  motion_specification_interfaces__action__MotionSpecification_GetResult_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
// already included above
// #include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  motion_specification_interfaces__action__MotionSpecification_GetResult_Event__request__MAX_SIZE = 1
};
// response
enum
{
  motion_specification_interfaces__action__MotionSpecification_GetResult_Event__response__MAX_SIZE = 1
};

/// Struct defined in action/MotionSpecification in the package motion_specification_interfaces.
typedef struct motion_specification_interfaces__action__MotionSpecification_GetResult_Event
{
  service_msgs__msg__ServiceEventInfo info;
  motion_specification_interfaces__action__MotionSpecification_GetResult_Request__Sequence request;
  motion_specification_interfaces__action__MotionSpecification_GetResult_Response__Sequence response;
} motion_specification_interfaces__action__MotionSpecification_GetResult_Event;

// Struct for a sequence of motion_specification_interfaces__action__MotionSpecification_GetResult_Event.
typedef struct motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence
{
  motion_specification_interfaces__action__MotionSpecification_GetResult_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} motion_specification_interfaces__action__MotionSpecification_GetResult_Event__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'feedback'
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__struct.h"

/// Struct defined in action/MotionSpecification in the package motion_specification_interfaces.
typedef struct motion_specification_interfaces__action__MotionSpecification_FeedbackMessage
{
  unique_identifier_msgs__msg__UUID goal_id;
  motion_specification_interfaces__action__MotionSpecification_Feedback feedback;
} motion_specification_interfaces__action__MotionSpecification_FeedbackMessage;

// Struct for a sequence of motion_specification_interfaces__action__MotionSpecification_FeedbackMessage.
typedef struct motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence
{
  motion_specification_interfaces__action__MotionSpecification_FeedbackMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MOTION_SPECIFICATION_INTERFACES__ACTION__DETAIL__MOTION_SPECIFICATION__STRUCT_H_

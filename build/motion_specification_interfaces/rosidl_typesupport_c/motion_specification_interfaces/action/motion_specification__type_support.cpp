// generated from rosidl_typesupport_c/resource/idl__type_support.cpp.em
// with input from motion_specification_interfaces:action/MotionSpecification.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "motion_specification_interfaces/action/detail/motion_specification__struct.h"
#include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"
#include "motion_specification_interfaces/action/detail/motion_specification__functions.h"
#include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/message_type_support_dispatch.h"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_c/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace motion_specification_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _MotionSpecification_Goal_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _MotionSpecification_Goal_type_support_ids_t;

static const _MotionSpecification_Goal_type_support_ids_t _MotionSpecification_Goal_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _MotionSpecification_Goal_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _MotionSpecification_Goal_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _MotionSpecification_Goal_type_support_symbol_names_t _MotionSpecification_Goal_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, motion_specification_interfaces, action, MotionSpecification_Goal)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, motion_specification_interfaces, action, MotionSpecification_Goal)),
  }
};

typedef struct _MotionSpecification_Goal_type_support_data_t
{
  void * data[2];
} _MotionSpecification_Goal_type_support_data_t;

static _MotionSpecification_Goal_type_support_data_t _MotionSpecification_Goal_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _MotionSpecification_Goal_message_typesupport_map = {
  2,
  "motion_specification_interfaces",
  &_MotionSpecification_Goal_message_typesupport_ids.typesupport_identifier[0],
  &_MotionSpecification_Goal_message_typesupport_symbol_names.symbol_name[0],
  &_MotionSpecification_Goal_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t MotionSpecification_Goal_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_MotionSpecification_Goal_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &motion_specification_interfaces__action__MotionSpecification_Goal__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification_Goal__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification_Goal__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace motion_specification_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_Goal)() {
  return &::motion_specification_interfaces::action::rosidl_typesupport_c::MotionSpecification_Goal_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace motion_specification_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _MotionSpecification_Result_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _MotionSpecification_Result_type_support_ids_t;

static const _MotionSpecification_Result_type_support_ids_t _MotionSpecification_Result_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _MotionSpecification_Result_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _MotionSpecification_Result_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _MotionSpecification_Result_type_support_symbol_names_t _MotionSpecification_Result_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, motion_specification_interfaces, action, MotionSpecification_Result)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, motion_specification_interfaces, action, MotionSpecification_Result)),
  }
};

typedef struct _MotionSpecification_Result_type_support_data_t
{
  void * data[2];
} _MotionSpecification_Result_type_support_data_t;

static _MotionSpecification_Result_type_support_data_t _MotionSpecification_Result_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _MotionSpecification_Result_message_typesupport_map = {
  2,
  "motion_specification_interfaces",
  &_MotionSpecification_Result_message_typesupport_ids.typesupport_identifier[0],
  &_MotionSpecification_Result_message_typesupport_symbol_names.symbol_name[0],
  &_MotionSpecification_Result_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t MotionSpecification_Result_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_MotionSpecification_Result_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &motion_specification_interfaces__action__MotionSpecification_Result__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification_Result__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification_Result__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace motion_specification_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_Result)() {
  return &::motion_specification_interfaces::action::rosidl_typesupport_c::MotionSpecification_Result_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace motion_specification_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _MotionSpecification_Feedback_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _MotionSpecification_Feedback_type_support_ids_t;

static const _MotionSpecification_Feedback_type_support_ids_t _MotionSpecification_Feedback_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _MotionSpecification_Feedback_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _MotionSpecification_Feedback_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _MotionSpecification_Feedback_type_support_symbol_names_t _MotionSpecification_Feedback_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, motion_specification_interfaces, action, MotionSpecification_Feedback)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, motion_specification_interfaces, action, MotionSpecification_Feedback)),
  }
};

typedef struct _MotionSpecification_Feedback_type_support_data_t
{
  void * data[2];
} _MotionSpecification_Feedback_type_support_data_t;

static _MotionSpecification_Feedback_type_support_data_t _MotionSpecification_Feedback_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _MotionSpecification_Feedback_message_typesupport_map = {
  2,
  "motion_specification_interfaces",
  &_MotionSpecification_Feedback_message_typesupport_ids.typesupport_identifier[0],
  &_MotionSpecification_Feedback_message_typesupport_symbol_names.symbol_name[0],
  &_MotionSpecification_Feedback_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t MotionSpecification_Feedback_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_MotionSpecification_Feedback_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &motion_specification_interfaces__action__MotionSpecification_Feedback__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification_Feedback__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification_Feedback__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace motion_specification_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_Feedback)() {
  return &::motion_specification_interfaces::action::rosidl_typesupport_c::MotionSpecification_Feedback_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace motion_specification_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _MotionSpecification_SendGoal_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _MotionSpecification_SendGoal_Request_type_support_ids_t;

static const _MotionSpecification_SendGoal_Request_type_support_ids_t _MotionSpecification_SendGoal_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _MotionSpecification_SendGoal_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _MotionSpecification_SendGoal_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _MotionSpecification_SendGoal_Request_type_support_symbol_names_t _MotionSpecification_SendGoal_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, motion_specification_interfaces, action, MotionSpecification_SendGoal_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, motion_specification_interfaces, action, MotionSpecification_SendGoal_Request)),
  }
};

typedef struct _MotionSpecification_SendGoal_Request_type_support_data_t
{
  void * data[2];
} _MotionSpecification_SendGoal_Request_type_support_data_t;

static _MotionSpecification_SendGoal_Request_type_support_data_t _MotionSpecification_SendGoal_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _MotionSpecification_SendGoal_Request_message_typesupport_map = {
  2,
  "motion_specification_interfaces",
  &_MotionSpecification_SendGoal_Request_message_typesupport_ids.typesupport_identifier[0],
  &_MotionSpecification_SendGoal_Request_message_typesupport_symbol_names.symbol_name[0],
  &_MotionSpecification_SendGoal_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t MotionSpecification_SendGoal_Request_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_MotionSpecification_SendGoal_Request_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification_SendGoal_Request__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace motion_specification_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_SendGoal_Request)() {
  return &::motion_specification_interfaces::action::rosidl_typesupport_c::MotionSpecification_SendGoal_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace motion_specification_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _MotionSpecification_SendGoal_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _MotionSpecification_SendGoal_Response_type_support_ids_t;

static const _MotionSpecification_SendGoal_Response_type_support_ids_t _MotionSpecification_SendGoal_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _MotionSpecification_SendGoal_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _MotionSpecification_SendGoal_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _MotionSpecification_SendGoal_Response_type_support_symbol_names_t _MotionSpecification_SendGoal_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, motion_specification_interfaces, action, MotionSpecification_SendGoal_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, motion_specification_interfaces, action, MotionSpecification_SendGoal_Response)),
  }
};

typedef struct _MotionSpecification_SendGoal_Response_type_support_data_t
{
  void * data[2];
} _MotionSpecification_SendGoal_Response_type_support_data_t;

static _MotionSpecification_SendGoal_Response_type_support_data_t _MotionSpecification_SendGoal_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _MotionSpecification_SendGoal_Response_message_typesupport_map = {
  2,
  "motion_specification_interfaces",
  &_MotionSpecification_SendGoal_Response_message_typesupport_ids.typesupport_identifier[0],
  &_MotionSpecification_SendGoal_Response_message_typesupport_symbol_names.symbol_name[0],
  &_MotionSpecification_SendGoal_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t MotionSpecification_SendGoal_Response_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_MotionSpecification_SendGoal_Response_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification_SendGoal_Response__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace motion_specification_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_SendGoal_Response)() {
  return &::motion_specification_interfaces::action::rosidl_typesupport_c::MotionSpecification_SendGoal_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace motion_specification_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _MotionSpecification_SendGoal_Event_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _MotionSpecification_SendGoal_Event_type_support_ids_t;

static const _MotionSpecification_SendGoal_Event_type_support_ids_t _MotionSpecification_SendGoal_Event_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _MotionSpecification_SendGoal_Event_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _MotionSpecification_SendGoal_Event_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _MotionSpecification_SendGoal_Event_type_support_symbol_names_t _MotionSpecification_SendGoal_Event_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, motion_specification_interfaces, action, MotionSpecification_SendGoal_Event)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, motion_specification_interfaces, action, MotionSpecification_SendGoal_Event)),
  }
};

typedef struct _MotionSpecification_SendGoal_Event_type_support_data_t
{
  void * data[2];
} _MotionSpecification_SendGoal_Event_type_support_data_t;

static _MotionSpecification_SendGoal_Event_type_support_data_t _MotionSpecification_SendGoal_Event_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _MotionSpecification_SendGoal_Event_message_typesupport_map = {
  2,
  "motion_specification_interfaces",
  &_MotionSpecification_SendGoal_Event_message_typesupport_ids.typesupport_identifier[0],
  &_MotionSpecification_SendGoal_Event_message_typesupport_symbol_names.symbol_name[0],
  &_MotionSpecification_SendGoal_Event_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t MotionSpecification_SendGoal_Event_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_MotionSpecification_SendGoal_Event_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification_SendGoal_Event__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace motion_specification_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_SendGoal_Event)() {
  return &::motion_specification_interfaces::action::rosidl_typesupport_c::MotionSpecification_SendGoal_Event_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/service_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
#include "service_msgs/msg/service_event_info.h"
#include "builtin_interfaces/msg/time.h"

namespace motion_specification_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{
typedef struct _MotionSpecification_SendGoal_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _MotionSpecification_SendGoal_type_support_ids_t;

static const _MotionSpecification_SendGoal_type_support_ids_t _MotionSpecification_SendGoal_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _MotionSpecification_SendGoal_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _MotionSpecification_SendGoal_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _MotionSpecification_SendGoal_type_support_symbol_names_t _MotionSpecification_SendGoal_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, motion_specification_interfaces, action, MotionSpecification_SendGoal)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, motion_specification_interfaces, action, MotionSpecification_SendGoal)),
  }
};

typedef struct _MotionSpecification_SendGoal_type_support_data_t
{
  void * data[2];
} _MotionSpecification_SendGoal_type_support_data_t;

static _MotionSpecification_SendGoal_type_support_data_t _MotionSpecification_SendGoal_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _MotionSpecification_SendGoal_service_typesupport_map = {
  2,
  "motion_specification_interfaces",
  &_MotionSpecification_SendGoal_service_typesupport_ids.typesupport_identifier[0],
  &_MotionSpecification_SendGoal_service_typesupport_symbol_names.symbol_name[0],
  &_MotionSpecification_SendGoal_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t MotionSpecification_SendGoal_service_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_MotionSpecification_SendGoal_service_typesupport_map),
  rosidl_typesupport_c__get_service_typesupport_handle_function,
  &MotionSpecification_SendGoal_Request_message_type_support_handle,
  &MotionSpecification_SendGoal_Response_message_type_support_handle,
  &MotionSpecification_SendGoal_Event_message_type_support_handle,
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_CREATE_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    motion_specification_interfaces,
    action,
    MotionSpecification_SendGoal
  ),
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_DESTROY_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    motion_specification_interfaces,
    action,
    MotionSpecification_SendGoal
  ),
  &motion_specification_interfaces__action__MotionSpecification_SendGoal__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification_SendGoal__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification_SendGoal__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace motion_specification_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_SendGoal)() {
  return &::motion_specification_interfaces::action::rosidl_typesupport_c::MotionSpecification_SendGoal_service_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace motion_specification_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _MotionSpecification_GetResult_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _MotionSpecification_GetResult_Request_type_support_ids_t;

static const _MotionSpecification_GetResult_Request_type_support_ids_t _MotionSpecification_GetResult_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _MotionSpecification_GetResult_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _MotionSpecification_GetResult_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _MotionSpecification_GetResult_Request_type_support_symbol_names_t _MotionSpecification_GetResult_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, motion_specification_interfaces, action, MotionSpecification_GetResult_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, motion_specification_interfaces, action, MotionSpecification_GetResult_Request)),
  }
};

typedef struct _MotionSpecification_GetResult_Request_type_support_data_t
{
  void * data[2];
} _MotionSpecification_GetResult_Request_type_support_data_t;

static _MotionSpecification_GetResult_Request_type_support_data_t _MotionSpecification_GetResult_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _MotionSpecification_GetResult_Request_message_typesupport_map = {
  2,
  "motion_specification_interfaces",
  &_MotionSpecification_GetResult_Request_message_typesupport_ids.typesupport_identifier[0],
  &_MotionSpecification_GetResult_Request_message_typesupport_symbol_names.symbol_name[0],
  &_MotionSpecification_GetResult_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t MotionSpecification_GetResult_Request_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_MotionSpecification_GetResult_Request_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &motion_specification_interfaces__action__MotionSpecification_GetResult_Request__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification_GetResult_Request__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification_GetResult_Request__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace motion_specification_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_GetResult_Request)() {
  return &::motion_specification_interfaces::action::rosidl_typesupport_c::MotionSpecification_GetResult_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace motion_specification_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _MotionSpecification_GetResult_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _MotionSpecification_GetResult_Response_type_support_ids_t;

static const _MotionSpecification_GetResult_Response_type_support_ids_t _MotionSpecification_GetResult_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _MotionSpecification_GetResult_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _MotionSpecification_GetResult_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _MotionSpecification_GetResult_Response_type_support_symbol_names_t _MotionSpecification_GetResult_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, motion_specification_interfaces, action, MotionSpecification_GetResult_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, motion_specification_interfaces, action, MotionSpecification_GetResult_Response)),
  }
};

typedef struct _MotionSpecification_GetResult_Response_type_support_data_t
{
  void * data[2];
} _MotionSpecification_GetResult_Response_type_support_data_t;

static _MotionSpecification_GetResult_Response_type_support_data_t _MotionSpecification_GetResult_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _MotionSpecification_GetResult_Response_message_typesupport_map = {
  2,
  "motion_specification_interfaces",
  &_MotionSpecification_GetResult_Response_message_typesupport_ids.typesupport_identifier[0],
  &_MotionSpecification_GetResult_Response_message_typesupport_symbol_names.symbol_name[0],
  &_MotionSpecification_GetResult_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t MotionSpecification_GetResult_Response_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_MotionSpecification_GetResult_Response_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &motion_specification_interfaces__action__MotionSpecification_GetResult_Response__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification_GetResult_Response__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification_GetResult_Response__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace motion_specification_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_GetResult_Response)() {
  return &::motion_specification_interfaces::action::rosidl_typesupport_c::MotionSpecification_GetResult_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace motion_specification_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _MotionSpecification_GetResult_Event_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _MotionSpecification_GetResult_Event_type_support_ids_t;

static const _MotionSpecification_GetResult_Event_type_support_ids_t _MotionSpecification_GetResult_Event_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _MotionSpecification_GetResult_Event_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _MotionSpecification_GetResult_Event_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _MotionSpecification_GetResult_Event_type_support_symbol_names_t _MotionSpecification_GetResult_Event_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, motion_specification_interfaces, action, MotionSpecification_GetResult_Event)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, motion_specification_interfaces, action, MotionSpecification_GetResult_Event)),
  }
};

typedef struct _MotionSpecification_GetResult_Event_type_support_data_t
{
  void * data[2];
} _MotionSpecification_GetResult_Event_type_support_data_t;

static _MotionSpecification_GetResult_Event_type_support_data_t _MotionSpecification_GetResult_Event_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _MotionSpecification_GetResult_Event_message_typesupport_map = {
  2,
  "motion_specification_interfaces",
  &_MotionSpecification_GetResult_Event_message_typesupport_ids.typesupport_identifier[0],
  &_MotionSpecification_GetResult_Event_message_typesupport_symbol_names.symbol_name[0],
  &_MotionSpecification_GetResult_Event_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t MotionSpecification_GetResult_Event_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_MotionSpecification_GetResult_Event_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &motion_specification_interfaces__action__MotionSpecification_GetResult_Event__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification_GetResult_Event__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification_GetResult_Event__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace motion_specification_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_GetResult_Event)() {
  return &::motion_specification_interfaces::action::rosidl_typesupport_c::MotionSpecification_GetResult_Event_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/service_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "service_msgs/msg/service_event_info.h"
// already included above
// #include "builtin_interfaces/msg/time.h"

namespace motion_specification_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{
typedef struct _MotionSpecification_GetResult_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _MotionSpecification_GetResult_type_support_ids_t;

static const _MotionSpecification_GetResult_type_support_ids_t _MotionSpecification_GetResult_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _MotionSpecification_GetResult_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _MotionSpecification_GetResult_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _MotionSpecification_GetResult_type_support_symbol_names_t _MotionSpecification_GetResult_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, motion_specification_interfaces, action, MotionSpecification_GetResult)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, motion_specification_interfaces, action, MotionSpecification_GetResult)),
  }
};

typedef struct _MotionSpecification_GetResult_type_support_data_t
{
  void * data[2];
} _MotionSpecification_GetResult_type_support_data_t;

static _MotionSpecification_GetResult_type_support_data_t _MotionSpecification_GetResult_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _MotionSpecification_GetResult_service_typesupport_map = {
  2,
  "motion_specification_interfaces",
  &_MotionSpecification_GetResult_service_typesupport_ids.typesupport_identifier[0],
  &_MotionSpecification_GetResult_service_typesupport_symbol_names.symbol_name[0],
  &_MotionSpecification_GetResult_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t MotionSpecification_GetResult_service_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_MotionSpecification_GetResult_service_typesupport_map),
  rosidl_typesupport_c__get_service_typesupport_handle_function,
  &MotionSpecification_GetResult_Request_message_type_support_handle,
  &MotionSpecification_GetResult_Response_message_type_support_handle,
  &MotionSpecification_GetResult_Event_message_type_support_handle,
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_CREATE_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    motion_specification_interfaces,
    action,
    MotionSpecification_GetResult
  ),
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_DESTROY_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    motion_specification_interfaces,
    action,
    MotionSpecification_GetResult
  ),
  &motion_specification_interfaces__action__MotionSpecification_GetResult__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification_GetResult__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification_GetResult__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace motion_specification_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_GetResult)() {
  return &::motion_specification_interfaces::action::rosidl_typesupport_c::MotionSpecification_GetResult_service_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__struct.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace motion_specification_interfaces
{

namespace action
{

namespace rosidl_typesupport_c
{

typedef struct _MotionSpecification_FeedbackMessage_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _MotionSpecification_FeedbackMessage_type_support_ids_t;

static const _MotionSpecification_FeedbackMessage_type_support_ids_t _MotionSpecification_FeedbackMessage_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _MotionSpecification_FeedbackMessage_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _MotionSpecification_FeedbackMessage_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _MotionSpecification_FeedbackMessage_type_support_symbol_names_t _MotionSpecification_FeedbackMessage_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, motion_specification_interfaces, action, MotionSpecification_FeedbackMessage)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, motion_specification_interfaces, action, MotionSpecification_FeedbackMessage)),
  }
};

typedef struct _MotionSpecification_FeedbackMessage_type_support_data_t
{
  void * data[2];
} _MotionSpecification_FeedbackMessage_type_support_data_t;

static _MotionSpecification_FeedbackMessage_type_support_data_t _MotionSpecification_FeedbackMessage_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _MotionSpecification_FeedbackMessage_message_typesupport_map = {
  2,
  "motion_specification_interfaces",
  &_MotionSpecification_FeedbackMessage_message_typesupport_ids.typesupport_identifier[0],
  &_MotionSpecification_FeedbackMessage_message_typesupport_symbol_names.symbol_name[0],
  &_MotionSpecification_FeedbackMessage_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t MotionSpecification_FeedbackMessage_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_MotionSpecification_FeedbackMessage_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification_FeedbackMessage__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace action

}  // namespace motion_specification_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_FeedbackMessage)() {
  return &::motion_specification_interfaces::action::rosidl_typesupport_c::MotionSpecification_FeedbackMessage_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

#include "action_msgs/msg/goal_status_array.h"
#include "action_msgs/srv/cancel_goal.h"
#include "motion_specification_interfaces/action/motion_specification.h"
// already included above
// #include "motion_specification_interfaces/action/detail/motion_specification__type_support.h"

static rosidl_action_type_support_t _motion_specification_interfaces__action__MotionSpecification__typesupport_c = {
  NULL, NULL, NULL, NULL, NULL,
  &motion_specification_interfaces__action__MotionSpecification__get_type_hash,
  &motion_specification_interfaces__action__MotionSpecification__get_type_description,
  &motion_specification_interfaces__action__MotionSpecification__get_type_description_sources,
};

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_action_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__ACTION_SYMBOL_NAME(
  rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification)()
{
  // Thread-safe by always writing the same values to the static struct
  _motion_specification_interfaces__action__MotionSpecification__typesupport_c.goal_service_type_support =
    ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(
    rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_SendGoal)();
  _motion_specification_interfaces__action__MotionSpecification__typesupport_c.result_service_type_support =
    ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(
    rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_GetResult)();
  _motion_specification_interfaces__action__MotionSpecification__typesupport_c.cancel_service_type_support =
    ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(
    rosidl_typesupport_c, action_msgs, srv, CancelGoal)();
  _motion_specification_interfaces__action__MotionSpecification__typesupport_c.feedback_message_type_support =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c, motion_specification_interfaces, action, MotionSpecification_FeedbackMessage)();
  _motion_specification_interfaces__action__MotionSpecification__typesupport_c.status_message_type_support =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c, action_msgs, msg, GoalStatusArray)();

  return &_motion_specification_interfaces__action__MotionSpecification__typesupport_c;
}

#ifdef __cplusplus
}
#endif

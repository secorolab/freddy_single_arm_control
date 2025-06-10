// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from motion_specification_interfaces:action/MotionSpecification.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "motion_specification_interfaces/action/motion_specification.hpp"


#ifndef MOTION_SPECIFICATION_INTERFACES__ACTION__DETAIL__MOTION_SPECIFICATION__BUILDER_HPP_
#define MOTION_SPECIFICATION_INTERFACES__ACTION__DETAIL__MOTION_SPECIFICATION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "motion_specification_interfaces/action/detail/motion_specification__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace motion_specification_interfaces
{

namespace action
{

namespace builder
{

class Init_MotionSpecification_Goal_motion_specification
{
public:
  Init_MotionSpecification_Goal_motion_specification()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::motion_specification_interfaces::action::MotionSpecification_Goal motion_specification(::motion_specification_interfaces::action::MotionSpecification_Goal::_motion_specification_type arg)
  {
    msg_.motion_specification = std::move(arg);
    return std::move(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_Goal msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::motion_specification_interfaces::action::MotionSpecification_Goal>()
{
  return motion_specification_interfaces::action::builder::Init_MotionSpecification_Goal_motion_specification();
}

}  // namespace motion_specification_interfaces


namespace motion_specification_interfaces
{

namespace action
{

namespace builder
{

class Init_MotionSpecification_Result_motion_successful
{
public:
  Init_MotionSpecification_Result_motion_successful()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::motion_specification_interfaces::action::MotionSpecification_Result motion_successful(::motion_specification_interfaces::action::MotionSpecification_Result::_motion_successful_type arg)
  {
    msg_.motion_successful = std::move(arg);
    return std::move(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_Result msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::motion_specification_interfaces::action::MotionSpecification_Result>()
{
  return motion_specification_interfaces::action::builder::Init_MotionSpecification_Result_motion_successful();
}

}  // namespace motion_specification_interfaces


namespace motion_specification_interfaces
{

namespace action
{

namespace builder
{

class Init_MotionSpecification_Feedback_tcp_position
{
public:
  Init_MotionSpecification_Feedback_tcp_position()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::motion_specification_interfaces::action::MotionSpecification_Feedback tcp_position(::motion_specification_interfaces::action::MotionSpecification_Feedback::_tcp_position_type arg)
  {
    msg_.tcp_position = std::move(arg);
    return std::move(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_Feedback msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::motion_specification_interfaces::action::MotionSpecification_Feedback>()
{
  return motion_specification_interfaces::action::builder::Init_MotionSpecification_Feedback_tcp_position();
}

}  // namespace motion_specification_interfaces


namespace motion_specification_interfaces
{

namespace action
{

namespace builder
{

class Init_MotionSpecification_SendGoal_Request_goal
{
public:
  explicit Init_MotionSpecification_SendGoal_Request_goal(::motion_specification_interfaces::action::MotionSpecification_SendGoal_Request & msg)
  : msg_(msg)
  {}
  ::motion_specification_interfaces::action::MotionSpecification_SendGoal_Request goal(::motion_specification_interfaces::action::MotionSpecification_SendGoal_Request::_goal_type arg)
  {
    msg_.goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_SendGoal_Request msg_;
};

class Init_MotionSpecification_SendGoal_Request_goal_id
{
public:
  Init_MotionSpecification_SendGoal_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MotionSpecification_SendGoal_Request_goal goal_id(::motion_specification_interfaces::action::MotionSpecification_SendGoal_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_MotionSpecification_SendGoal_Request_goal(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_SendGoal_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::motion_specification_interfaces::action::MotionSpecification_SendGoal_Request>()
{
  return motion_specification_interfaces::action::builder::Init_MotionSpecification_SendGoal_Request_goal_id();
}

}  // namespace motion_specification_interfaces


namespace motion_specification_interfaces
{

namespace action
{

namespace builder
{

class Init_MotionSpecification_SendGoal_Response_stamp
{
public:
  explicit Init_MotionSpecification_SendGoal_Response_stamp(::motion_specification_interfaces::action::MotionSpecification_SendGoal_Response & msg)
  : msg_(msg)
  {}
  ::motion_specification_interfaces::action::MotionSpecification_SendGoal_Response stamp(::motion_specification_interfaces::action::MotionSpecification_SendGoal_Response::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_SendGoal_Response msg_;
};

class Init_MotionSpecification_SendGoal_Response_accepted
{
public:
  Init_MotionSpecification_SendGoal_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MotionSpecification_SendGoal_Response_stamp accepted(::motion_specification_interfaces::action::MotionSpecification_SendGoal_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_MotionSpecification_SendGoal_Response_stamp(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_SendGoal_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::motion_specification_interfaces::action::MotionSpecification_SendGoal_Response>()
{
  return motion_specification_interfaces::action::builder::Init_MotionSpecification_SendGoal_Response_accepted();
}

}  // namespace motion_specification_interfaces


namespace motion_specification_interfaces
{

namespace action
{

namespace builder
{

class Init_MotionSpecification_SendGoal_Event_response
{
public:
  explicit Init_MotionSpecification_SendGoal_Event_response(::motion_specification_interfaces::action::MotionSpecification_SendGoal_Event & msg)
  : msg_(msg)
  {}
  ::motion_specification_interfaces::action::MotionSpecification_SendGoal_Event response(::motion_specification_interfaces::action::MotionSpecification_SendGoal_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_SendGoal_Event msg_;
};

class Init_MotionSpecification_SendGoal_Event_request
{
public:
  explicit Init_MotionSpecification_SendGoal_Event_request(::motion_specification_interfaces::action::MotionSpecification_SendGoal_Event & msg)
  : msg_(msg)
  {}
  Init_MotionSpecification_SendGoal_Event_response request(::motion_specification_interfaces::action::MotionSpecification_SendGoal_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_MotionSpecification_SendGoal_Event_response(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_SendGoal_Event msg_;
};

class Init_MotionSpecification_SendGoal_Event_info
{
public:
  Init_MotionSpecification_SendGoal_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MotionSpecification_SendGoal_Event_request info(::motion_specification_interfaces::action::MotionSpecification_SendGoal_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_MotionSpecification_SendGoal_Event_request(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_SendGoal_Event msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::motion_specification_interfaces::action::MotionSpecification_SendGoal_Event>()
{
  return motion_specification_interfaces::action::builder::Init_MotionSpecification_SendGoal_Event_info();
}

}  // namespace motion_specification_interfaces


namespace motion_specification_interfaces
{

namespace action
{

namespace builder
{

class Init_MotionSpecification_GetResult_Request_goal_id
{
public:
  Init_MotionSpecification_GetResult_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::motion_specification_interfaces::action::MotionSpecification_GetResult_Request goal_id(::motion_specification_interfaces::action::MotionSpecification_GetResult_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_GetResult_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::motion_specification_interfaces::action::MotionSpecification_GetResult_Request>()
{
  return motion_specification_interfaces::action::builder::Init_MotionSpecification_GetResult_Request_goal_id();
}

}  // namespace motion_specification_interfaces


namespace motion_specification_interfaces
{

namespace action
{

namespace builder
{

class Init_MotionSpecification_GetResult_Response_result
{
public:
  explicit Init_MotionSpecification_GetResult_Response_result(::motion_specification_interfaces::action::MotionSpecification_GetResult_Response & msg)
  : msg_(msg)
  {}
  ::motion_specification_interfaces::action::MotionSpecification_GetResult_Response result(::motion_specification_interfaces::action::MotionSpecification_GetResult_Response::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_GetResult_Response msg_;
};

class Init_MotionSpecification_GetResult_Response_status
{
public:
  Init_MotionSpecification_GetResult_Response_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MotionSpecification_GetResult_Response_result status(::motion_specification_interfaces::action::MotionSpecification_GetResult_Response::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_MotionSpecification_GetResult_Response_result(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_GetResult_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::motion_specification_interfaces::action::MotionSpecification_GetResult_Response>()
{
  return motion_specification_interfaces::action::builder::Init_MotionSpecification_GetResult_Response_status();
}

}  // namespace motion_specification_interfaces


namespace motion_specification_interfaces
{

namespace action
{

namespace builder
{

class Init_MotionSpecification_GetResult_Event_response
{
public:
  explicit Init_MotionSpecification_GetResult_Event_response(::motion_specification_interfaces::action::MotionSpecification_GetResult_Event & msg)
  : msg_(msg)
  {}
  ::motion_specification_interfaces::action::MotionSpecification_GetResult_Event response(::motion_specification_interfaces::action::MotionSpecification_GetResult_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_GetResult_Event msg_;
};

class Init_MotionSpecification_GetResult_Event_request
{
public:
  explicit Init_MotionSpecification_GetResult_Event_request(::motion_specification_interfaces::action::MotionSpecification_GetResult_Event & msg)
  : msg_(msg)
  {}
  Init_MotionSpecification_GetResult_Event_response request(::motion_specification_interfaces::action::MotionSpecification_GetResult_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_MotionSpecification_GetResult_Event_response(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_GetResult_Event msg_;
};

class Init_MotionSpecification_GetResult_Event_info
{
public:
  Init_MotionSpecification_GetResult_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MotionSpecification_GetResult_Event_request info(::motion_specification_interfaces::action::MotionSpecification_GetResult_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_MotionSpecification_GetResult_Event_request(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_GetResult_Event msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::motion_specification_interfaces::action::MotionSpecification_GetResult_Event>()
{
  return motion_specification_interfaces::action::builder::Init_MotionSpecification_GetResult_Event_info();
}

}  // namespace motion_specification_interfaces


namespace motion_specification_interfaces
{

namespace action
{

namespace builder
{

class Init_MotionSpecification_FeedbackMessage_feedback
{
public:
  explicit Init_MotionSpecification_FeedbackMessage_feedback(::motion_specification_interfaces::action::MotionSpecification_FeedbackMessage & msg)
  : msg_(msg)
  {}
  ::motion_specification_interfaces::action::MotionSpecification_FeedbackMessage feedback(::motion_specification_interfaces::action::MotionSpecification_FeedbackMessage::_feedback_type arg)
  {
    msg_.feedback = std::move(arg);
    return std::move(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_FeedbackMessage msg_;
};

class Init_MotionSpecification_FeedbackMessage_goal_id
{
public:
  Init_MotionSpecification_FeedbackMessage_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MotionSpecification_FeedbackMessage_feedback goal_id(::motion_specification_interfaces::action::MotionSpecification_FeedbackMessage::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_MotionSpecification_FeedbackMessage_feedback(msg_);
  }

private:
  ::motion_specification_interfaces::action::MotionSpecification_FeedbackMessage msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::motion_specification_interfaces::action::MotionSpecification_FeedbackMessage>()
{
  return motion_specification_interfaces::action::builder::Init_MotionSpecification_FeedbackMessage_goal_id();
}

}  // namespace motion_specification_interfaces

#endif  // MOTION_SPECIFICATION_INTERFACES__ACTION__DETAIL__MOTION_SPECIFICATION__BUILDER_HPP_

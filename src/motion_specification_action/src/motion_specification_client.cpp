#include <functional>
#include <future>
#include <memory>
#include <string>
#include <sstream>

#include "motion_specification_interfaces/action/motion_specification.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace motion_specification_action
{
class MotionSpecificationActionClient : public rclcpp::Node
{
public:
  using MotionSpecification = motion_specification_interfaces::action::MotionSpecification;
  using GoalHandleMotionSpecification = rclcpp_action::ClientGoalHandle<MotionSpecification>;

  explicit MotionSpecificationActionClient(const rclcpp::NodeOptions & options)
  : Node("motion_specification_action_client", options)
  {
    this->client_ptr_ = rclcpp_action::create_client<MotionSpecification>(
      this,
      "motion_specification");

    auto timer_callback_lambda = [this](){ return this->send_goal(); };
    this->timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      timer_callback_lambda);
  }

  void send_goal()
  {
    using namespace std::placeholders;

    this->timer_->cancel();

    if (!this->client_ptr_->wait_for_action_server()) {
      RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
      rclcpp::shutdown();
    }

    auto goal_msg = MotionSpecification::Goal();
    std::string motion_specification = "motion_specification";
    goal_msg.motion_specification = motion_specification;

    RCLCPP_INFO(this->get_logger(), "Sending goal");

    auto send_goal_options = rclcpp_action::Client<MotionSpecification>::SendGoalOptions();
    send_goal_options.goal_response_callback = [this](const GoalHandleMotionSpecification::SharedPtr & goal_handle)
    {
      if (!goal_handle) {
        RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
      } else {
        RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
      }
    };

    send_goal_options.feedback_callback = [this](
      GoalHandleMotionSpecification::SharedPtr,
      const std::shared_ptr<const MotionSpecification::Feedback> feedback)
    {
        std::stringstream ss;
        ss << "Current tcp with respect to global frame is: ";
        ss << std::fixed << std::setprecision(2);  // Ensures fixed-point notation with 2 decimal places
        
        for (float number : feedback->tcp_position) {
            ss << number << " ";
        }
    };

    send_goal_options.result_callback = [this](const GoalHandleMotionSpecification::WrappedResult & result)
    {
      switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
          break;
        case rclcpp_action::ResultCode::ABORTED:
          RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
          return;
        case rclcpp_action::ResultCode::CANCELED:
          RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
          return;
        default:
          RCLCPP_ERROR(this->get_logger(), "Unknown result code");
          return;
      }
      std::stringstream ss;
      ss << "Result received: ";
        ss << result.result->motion_successful;
      RCLCPP_INFO(this->get_logger(), ss.str().c_str());
      rclcpp::shutdown();
    };
    this->client_ptr_->async_send_goal(goal_msg, send_goal_options);
  }

private:
  rclcpp_action::Client<MotionSpecification>::SharedPtr client_ptr_;
  rclcpp::TimerBase::SharedPtr timer_;
};  // class MotionSpecificationActionClient

}  // namespace motion_specification_action

RCLCPP_COMPONENTS_REGISTER_NODE(motion_specification_action::MotionSpecificationActionClient)
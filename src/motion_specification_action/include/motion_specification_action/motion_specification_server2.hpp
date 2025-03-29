#ifndef MOTION_SPECIFICATION_ACTION_SERVER_HPP
#define MOTION_SPECIFICATION_ACTION_SERVER_HPP

#include <functional>
#include <memory>
#include <thread>
#include <csignal>

#include <motion_specification_interfaces/action/motion_specification.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <ctime>
#include <sstream>
#include <fstream>
#include <kinova_mediator/mediator.hpp>
#include <math.h>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>

// KDL libraries
#include <kdl/chain.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainfksolvervel_recursive.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainjnttojacdotsolver.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/frames.hpp>
#include <kdl/frames_io.hpp>
#include <kdl_parser/kdl_parser.hpp>

// namespace motion_specification_action
// {
//   class MotionSpecificationActionServer : public rclcpp::Node
//   {
//   public:
//     using MotionSpecification = motion_specification_interfaces::action::MotionSpecification;
//     using GoalHandleMotionSpecification = rclcpp_action::ServerGoalHandle<MotionSpecification>;

//     MOTION_SPECIFICATION_ACTION_PUBLIC
//     explicit MotionSpecificationActionServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
//     ~MotionSpecificationActionServer();

//   private:
//     rclcpp_action::Server<MotionSpecification>::SharedPtr action_server_;
//     std::atomic<bool> control_loop_active_;
//     std::thread control_loop_thread_;

//     void control_loop();
//     void execute(const std::shared_ptr<GoalHandleMotionSpecification> goal_handle);

//     // Callbacks for the action server
//     rclcpp_action::GoalResponse handle_goal(
//       const rclcpp_action::GoalUUID & uuid,
//       std::shared_ptr<const MotionSpecification::Goal> goal);

//     rclcpp_action::CancelResponse handle_cancel(
//       const std::shared_ptr<GoalHandleMotionSpecification> goal_handle);

//     void handle_accepted(
//       const std::shared_ptr<GoalHandleMotionSpecification> goal_handle);
//   };
// }  // namespace motion_specification_action

#endif  // MOTION_SPECIFICATION_ACTION_SERVER_HPP

// to solve issue with building with header file,
// refer: https://github.com/ros/ros_tutorials/blob/54c2b3c70884f957453419b91c58f89d66b4563e/turtlesim/include/turtlesim/turtle.hpp (remove visibility_control.h ?)
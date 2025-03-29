#include "motion_specification_action/motion_specification_server.hpp"

namespace motion_specification_action
{
  MotionSpecificationActionServer::MotionSpecificationActionServer(const rclcpp::NodeOptions &options)
      : Node("motion_specification_action_server", options),
        control_loop_active_(true),
        flag(0),
        motion_completed(false),
        goal_accepted_and_executing(false),
        zero_jnt_velocities(kinova_constants::NUMBER_OF_JOINTS),
        pre_condition_constraint_count(0),
        per_condition_constraint_count(0),
        post_condition_constraint_count(0),
        prevail_condition_constraint_count(0),
        motion_specification_read(0),
        frequency_of_checking_motion_specification(10), // TODO: check if this is required
        gravitational_acceleration{0.0f, 0.0f, -9.81f},
        time_period_of_complete_controller_cycle_data(0.0),
        desired_quat_GF{0.0, 0.0, 0.0, 1.0},
        measured_lin_vel_x_axis_data(0.0),
        measured_lin_vel_y_axis_data(0.0),
        measured_lin_vel_z_axis_data(0.0),
        lin_pos_sp_x_axis_data(0.0),
        lin_pos_sp_y_axis_data(0.0),
        lin_pos_sp_z_axis_data(0.0),
        lin_vel_sp_x_axis_data(0.0),
        lin_vel_sp_y_axis_data(0.0),
        lin_vel_sp_z_axis_data(0.0),
        stiffness_term_lin_x_axis_data(0.0),
        stiffness_term_lin_y_axis_data(0.0),
        stiffness_term_lin_z_axis_data(0.0),
        damping_term_x_axis_data(0.0),
        damping_term_y_axis_data(0.0),
        damping_term_z_axis_data(0.0),
        lin_pos_error_stiffness_x_axis_data(0.0),
        lin_pos_error_stiffness_y_axis_data(0.0),
        lin_pos_error_stiffness_z_axis_data(0.0),
        lin_vel_error_damping_x_axis_data(0.0),
        lin_vel_error_damping_y_axis_data(0.0),
        lin_vel_error_damping_z_axis_data(0.0),
        stiffness_damping_terms_summation_x_axis_data(0.0),
        stiffness_damping_terms_summation_y_axis_data(0.0),
        stiffness_damping_terms_summation_z_axis_data(0.0),
        apply_ee_force_x_axis_data(0.0),
        apply_ee_force_y_axis_data(0.0),
        apply_ee_force_z_axis_data(0.0),
        apply_ee_torque_x_axis_data(0.0),
        apply_ee_torque_y_axis_data(0.0),
        apply_ee_torque_z_axis_data(0.0),
        measured_roll_data(0.0),
        measured_pitch_data(0.0),
        measured_yaw_data(0.0),
        force_to_apply_x_axis(0.0),
        force_to_apply_y_axis(0.0),
        force_to_apply_z_axis(0.0),
        configuration_file_read(false),
        rne_output_jnt_torques_vector_to_set_control_mode(kinova_constants::NUMBER_OF_JOINTS, 0.0),
        arm_name("kinova_gen3_2_right")
  {
    using namespace std::placeholders;
    package_share_directory = ament_index_cpp::get_package_share_directory("motion_specification_action");
    config_file_path = package_share_directory + "/config/ms_config.yaml";
    config_file_object = YAML::LoadFile(config_file_path);
    read_config_file(config_file_object);
    zero_jnt_velocities.data.setZero();

    auto handle_goal = [this](
                           const rclcpp_action::GoalUUID &uuid,
                           std::shared_ptr<const MotionSpecification::Goal> goal)
    {
      RCLCPP_INFO(this->get_logger(), "Received motion_specification as goal: %s", goal->motion_specification.c_str());
      (void)uuid;
      return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    };

    auto handle_cancel = [this](
                             const std::shared_ptr<GoalHandleMotionSpecification> goal_handle)
    {
      RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
      (void)goal_handle;
      return rclcpp_action::CancelResponse::ACCEPT;
    };

    auto handle_accepted = [this](
                               const std::shared_ptr<GoalHandleMotionSpecification> goal_handle)
    {
      // this needs to return quickly to avoid blocking the executor,
      // so we declare a lambda function to be called inside a new thread
      auto execute_in_thread = [this, goal_handle]()
      { return this->execute(goal_handle); };
      std::thread{execute_in_thread}.detach();
    };

    this->action_server_ = rclcpp_action::create_server<MotionSpecification>(
        this,
        "motion_specification", // action name (not the node name)
        handle_goal,
        handle_cancel,
        handle_accepted);

    // Start the control loop in a separate thread
    control_loop_thread_ = std::thread([this]()
                                       { this->control_loop(); });
  }

  MotionSpecificationActionServer::~MotionSpecificationActionServer()
  {
    // Stop the control loop thread gracefully when the node is shutting down
    control_loop_active_ = false;
    if (control_loop_thread_.joinable())
    {
      control_loop_thread_.join();
    }
  }

  // Define all the functions related to the control loop

  // void MotionSpecificationActionServer::handle_signal(int sig)
  // {
  //     flag = 1;
  //     std::cout << "Received signal: " << sig << std::endl;
  // }

  void MotionSpecificationActionServer::kinova_feedback(kinova_mediator &kinova_arm_mediator,
                                                        KDL::JntArray &jnt_positions,
                                                        KDL::JntArray &jnt_velocities,
                                                        KDL::JntArray &jnt_torques)
  {
    kinova_arm_mediator.get_joint_state(jnt_positions,
                                        jnt_velocities,
                                        jnt_torques);
  }

  void MotionSpecificationActionServer::get_end_effector_pose_and_twist(KDL::JntArrayVel &jnt_velocity,
                                                                        const KDL::JntArray &jnt_positions,
                                                                        const KDL::JntArray &jnt_velocities,
                                                                        KDL::Frame &measured_endEffPose_BL,
                                                                        KDL::FrameVel &measured_endEffTwist_BL,
                                                                        KDL::Frame &measured_endEffPose_GF,
                                                                        KDL::FrameVel &measured_endEffTwist_GF,
                                                                        std::shared_ptr<KDL::ChainFkSolverPos_recursive> &fkSolverPos,
                                                                        std::shared_ptr<KDL::ChainFkSolverVel_recursive> &fkSolverVel,
                                                                        const KDL::Frame &BL_wrt_GF_frame)
  {
    jnt_velocity.q = jnt_positions;
    jnt_velocity.qdot = jnt_velocities;

    fkSolverPos->JntToCart(jnt_positions, measured_endEffPose_BL);
    fkSolverVel->JntToCart(jnt_velocity, measured_endEffTwist_BL);

    measured_endEffPose_GF = BL_wrt_GF_frame * measured_endEffPose_BL;
    measured_endEffTwist_GF = BL_wrt_GF_frame * measured_endEffTwist_BL;
  }

  void MotionSpecificationActionServer::calculate_joint_torques_RNEA(
      std::shared_ptr<KDL::ChainJntToJacDotSolver> &jacobDotSolver,
      std::shared_ptr<KDL::ChainIkSolverVel_pinv> &ikSolverAcc,
      std::shared_ptr<KDL::ChainIdSolver_RNE> &idSolver,
      KDL::JntArrayVel &jnt_velocity,
      KDL::Twist &jd_qd,
      KDL::Twist &xdd,
      KDL::Twist &xdd_minus_jd_qd,
      KDL::JntArray &jnt_accelerations,
      KDL::JntArray &jnt_positions,
      KDL::JntArray &jnt_velocities,
      KDL::Wrenches &linkWrenches_EE,
      KDL::JntArray &jnt_torques)
  {
    jacobDotSolver->JntToJacDot(jnt_velocity, jd_qd);
    xdd_minus_jd_qd = xdd - jd_qd;
    ikSolverAcc->CartToJnt(jnt_positions, xdd_minus_jd_qd, jnt_accelerations);
    idSolver->CartToJnt(jnt_positions, jnt_velocities, jnt_accelerations, linkWrenches_EE, jnt_torques);
  }

  template <size_t N>
  void MotionSpecificationActionServer::appendDataToFile_dynamic_size(std::ofstream &file, const std::vector<std::array<double, N>> &data)
  {
    for (const auto &row : data)
    {
      for (size_t i = 0; i < N; ++i)
      {
        file << row[i];
        if (i < N - 1)
        {
          file << ",";
        }
      }
      file << "\n";
    }
  }

  const std::unordered_map<std::string, constraint_type> &MotionSpecificationActionServer::getConstraintTypeMap()
  {
    static const std::unordered_map<std::string, constraint_type> constraint_type_map = {
        {"POSITION_XYZ", constraint_type::POSITION_XYZ},
        {"FORCE_XYZ", constraint_type::FORCE_XYZ},
        {"VELOCITY_XYZ", constraint_type::VELOCITY_XYZ},
        {"TORQUE_RPY", constraint_type::TORQUE_RPY},
        {"ORIENTATION_QUATERNION", constraint_type::ORIENTATION_QUATERNION},
        {"ORIENTATION_ROLL", constraint_type::ORIENTATION_ROLL},
        {"ORIENTATION_PITCH", constraint_type::ORIENTATION_PITCH},
        {"ORIENTATION_YAW", constraint_type::ORIENTATION_YAW}};
    return constraint_type_map;
  }

  const std::unordered_map<std::string, operator_type> &MotionSpecificationActionServer::getOperatorTypeMap()
  {
    static const std::unordered_map<std::string, operator_type> operator_type_map = {
        {"GREATER_THAN", operator_type::GREATER_THAN},
        {"LESS_THAN", operator_type::LESS_THAN}};
    return operator_type_map;
  }

  const std::unordered_map<std::string, condition_type> &MotionSpecificationActionServer::getConditionTypeMap()
  {
    static const std::unordered_map<std::string, condition_type> condition_type_map = {
        {"PRE_CONDITION", condition_type::PRE_CONDITION},
        {"PER_CONDITION", condition_type::PER_CONDITION},
        {"POST_CONDITION", condition_type::POST_CONDITION},
        {"PREVAIL_CONDITION", condition_type::PREVAIL_CONDITION}};
    return condition_type_map;
  }

  std::string MotionSpecificationActionServer::getTimestamp()
  {
    // Get current time
    std::time_t now = std::time(nullptr);
    std::tm *localTime = std::localtime(&now);

    // Format: YYYYMMDD_HHMMSS
    std::ostringstream oss;
    oss << (localTime->tm_year + 1900) // Year
        << (localTime->tm_mon + 1)     // Month
        << localTime->tm_mday << "_"   // Day
        << localTime->tm_hour          // Hour
        << localTime->tm_min           // Minute
        << localTime->tm_sec;          // Second

    return oss.str();
  }

  void MotionSpecificationActionServer::check_3D_vector_constraint_satisfaction(
      const double &measured_x_axis_data,
      const double &measured_y_axis_data,
      const double &measured_z_axis_data,
      bool &constraint_satisfied,
      const int &constraint_idx,
      const YAML::Node &motion_specification_params,
      const std::string &arm_name,
      const condition_type &condition_type_value)
  {
    auto operator_type_map = getOperatorTypeMap();
    std::string condition_type_str;
    double desired_data;
    operator_type operator_type_;
    std::string operator_type_str;

    if (condition_type_value == condition_type::PRE_CONDITION)
    {
      condition_type_str = "PRE_CONDITION";
    }
    else if (condition_type_value == condition_type::POST_CONDITION)
    {
      condition_type_str = "POST_CONDITION";
    }
    else if (condition_type_value == condition_type::PREVAIL_CONDITION)
    {
      condition_type_str = "PREVAIL_CONDITION";
    }
    else
    {
      std::cout << "[check_3D_vector_constraint_satisfaction] Condition type not found" << std::endl;
      flag = 1; // stop the execution
    }

    for (int j = 0; j < 3; j++)
    {
      if (!constraint_satisfied)
      {
        break;
      }
      auto operator_value = motion_specification_params[arm_name][condition_type_str]["constraints"][constraint_idx]["operator"][j];
      if (!operator_value.IsNull())
      {
        operator_type_str = motion_specification_params[arm_name][condition_type_str]["constraints"][constraint_idx]["operator"][j].as<std::string>();
        auto operator_iterator = operator_type_map.find(operator_type_str);
        if (operator_iterator != operator_type_map.end())
        {
          operator_type_ = operator_iterator->second;

          switch (operator_type_)
          {
          case GREATER_THAN:
            desired_data = motion_specification_params[arm_name][condition_type_str]["constraints"][constraint_idx]["value"][j].as<double>();
            if (j == 0)
            {
              greater_than_monitor(&measured_x_axis_data, &desired_data, &constraint_satisfied);
            }
            else if (j == 1)
            {
              greater_than_monitor(&measured_y_axis_data, &desired_data, &constraint_satisfied);
            }
            else if (j == 2)
            {
              greater_than_monitor(&measured_z_axis_data, &desired_data, &constraint_satisfied);
            }
            break;

          case LESS_THAN:
            desired_data = motion_specification_params[arm_name][condition_type_str]["constraints"][constraint_idx]["value"][j].as<double>();
            if (j == 0)
            {
              less_than_monitor(&measured_x_axis_data, &desired_data, &constraint_satisfied);
            }
            else if (j == 1)
            {
              less_than_monitor(&measured_y_axis_data, &desired_data, &constraint_satisfied);
            }
            else if (j == 2)
            {
              less_than_monitor(&measured_z_axis_data, &desired_data, &constraint_satisfied);
            }
            break;

          default:
            break;
          }
        }
        else
        {
          std::cout << "[check_3D_vector_constraint_satisfaction] Operator type not found" << std::endl;
          flag = 1; // stop the execution
        }
      }
    }
  }

  void MotionSpecificationActionServer::check_1D_vector_constraint_satisfaction(
      const double &measured_data,
      bool &constraint_satisfied,
      const int &constraint_idx,
      const YAML::Node &motion_specification_params,
      const std::string &arm_name,
      const condition_type &condition_type_value)
  {
    auto operator_type_map = getOperatorTypeMap();
    std::string condition_type_str;
    double desired_data;
    operator_type operator_type_;
    std::string operator_type_str;

    if (condition_type_value == condition_type::PRE_CONDITION)
    {
      condition_type_str = "PRE_CONDITION";
    }
    else if (condition_type_value == condition_type::POST_CONDITION)
    {
      condition_type_str = "POST_CONDITION";
    }
    else if (condition_type_value == condition_type::PREVAIL_CONDITION)
    {
      condition_type_str = "PREVAIL_CONDITION";
    }
    else
    {
      std::cout << "[check_1D_vector_constraint_satisfaction] Condition type not found" << std::endl;
      flag = 1; // stop the execution
    }

    auto operator_value = motion_specification_params[arm_name][condition_type_str]["constraints"][constraint_idx]["operator"];
    if (!operator_value.IsNull())
    {
      operator_type_str = motion_specification_params[arm_name][condition_type_str]["constraints"][constraint_idx]["operator"].as<std::string>();
      auto operator_iterator = operator_type_map.find(operator_type_str);

      if (operator_iterator != operator_type_map.end())
      {
        operator_type_ = operator_iterator->second;

        switch (operator_type_)
        {
        case GREATER_THAN:
          desired_data = motion_specification_params[arm_name][condition_type_str]["constraints"][constraint_idx]["value"].as<double>();
          greater_than_monitor(&measured_data, &desired_data, &constraint_satisfied);
          break;

        case LESS_THAN:
          desired_data = motion_specification_params[arm_name][condition_type_str]["constraints"][constraint_idx]["value"].as<double>();
          less_than_monitor(&measured_data, &desired_data, &constraint_satisfied);
          break;

        default:
          break;
        }
      }
      else
      {
        std::cout << "[check_1D_vector_constraint_satisfaction] Operator type not found" << std::endl;
        flag = 1; // stop the execution
      }
    }
    else
    {
      std::cout << "[check_1D_vector_constraint_satisfaction] Null operator in 1D constraint check is not meaningful" << std::endl;
    }
  }

  void MotionSpecificationActionServer::check_pre_or_post_or_prevail_condition_satisfaction(
      const double &measured_lin_pos_x_axis_data,
      const double &measured_lin_pos_y_axis_data,
      const double &measured_lin_pos_z_axis_data,
      const double &measured_roll_data,
      const double &measured_pitch_data,
      const double &measured_yaw_data,
      const double &measured_lin_vel_x_axis_data,
      const double &measured_lin_vel_y_axis_data,
      const double &measured_lin_vel_z_axis_data,
      KDL::Wrench &linkWrenches_EE,
      const int &condition_constraint_count,
      std::string &constraint_type_str,
      const std::string &arm_name,
      bool &condition_satisfied,
      const YAML::Node &motion_specification_params,
      const condition_type &condition_type_value)
  {
    auto constraint_type_map = getConstraintTypeMap();
    bool constraint_satisfied = true;
    std::string condition_type_str;
    constraint_type constraint_type_;

    // for every constraint in the pre-condition
    if (condition_constraint_count > 0)
    {
      for (int i = 1; i < condition_constraint_count + 1; i++)
      {
        if (condition_type_value == condition_type::PRE_CONDITION)
        {
          condition_type_str = "PRE_CONDITION";
        }
        else if (condition_type_value == condition_type::POST_CONDITION)
        {
          condition_type_str = "POST_CONDITION";
        }
        else if (condition_type_value == condition_type::PREVAIL_CONDITION)
        {
          condition_type_str = "PREVAIL_CONDITION";
        }
        else
        {
          std::cout << "[check_pre_or_post_or_prevail_condition_satisfaction] Condition type not found" << std::endl;
          flag = 1; // stop the execution
        }
        constraint_type_str = motion_specification_params[arm_name][condition_type_str]["constraints"][i]["type"].as<std::string>();

        auto constraint_iterator = constraint_type_map.find(constraint_type_str);

        if (constraint_iterator != constraint_type_map.end())
        {
          constraint_type_ = constraint_iterator->second; // selecting the second value stored in the iterator (the first value is the key)

          switch (constraint_type_)
          {
          case POSITION_XYZ:
            check_3D_vector_constraint_satisfaction(
                measured_lin_pos_x_axis_data,
                measured_lin_pos_y_axis_data,
                measured_lin_pos_z_axis_data,
                constraint_satisfied,
                i,
                motion_specification_params,
                arm_name,
                condition_type_value);
            break;

          case VELOCITY_XYZ:
            if (condition_type_value == condition_type::PRE_CONDITION)
            {
              std::cout << "[check_pre_or_post_or_prevail_condition_satisfaction] Velocity constraint not allowed in pre-condition" << std::endl;
              flag = 1; // stop the execution
              break;
            }
            check_3D_vector_constraint_satisfaction(
                measured_lin_vel_x_axis_data,
                measured_lin_vel_y_axis_data,
                measured_lin_vel_z_axis_data,
                constraint_satisfied,
                i,
                motion_specification_params,
                arm_name,
                condition_type_value);
            break;

          case ORIENTATION_ROLL:
            check_1D_vector_constraint_satisfaction(measured_roll_data, constraint_satisfied, i, motion_specification_params, arm_name, condition_type_value);
            break;

          case ORIENTATION_PITCH:
            check_1D_vector_constraint_satisfaction(measured_pitch_data, constraint_satisfied, i, motion_specification_params, arm_name, condition_type_value);
            break;

          case ORIENTATION_YAW:
            check_1D_vector_constraint_satisfaction(measured_yaw_data, constraint_satisfied, i, motion_specification_params, arm_name, condition_type_value);
            break;

          case FORCE_XYZ:
            if (condition_type_value == condition_type::PRE_CONDITION)
            {
              std::cout << "[check_pre_or_post_or_prevail_condition_satisfaction] Force constraint not allowed in pre-condition" << std::endl;
              flag = 1; // stop the execution
              break;
            }
            check_3D_vector_constraint_satisfaction(
                linkWrenches_EE.force(0),
                linkWrenches_EE.force(1),
                linkWrenches_EE.force(2),
                constraint_satisfied,
                i,
                motion_specification_params,
                arm_name,
                condition_type_value);
            break;

          case TORQUE_RPY:
            if (condition_type_value == condition_type::PRE_CONDITION)
            {
              std::cout << "[check_pre_or_post_or_prevail_condition_satisfaction] Torque constraint not allowed in pre-condition" << std::endl;
              flag = 1; // stop the execution
              break;
            }
            check_3D_vector_constraint_satisfaction(
                linkWrenches_EE.torque(0),
                linkWrenches_EE.torque(1),
                linkWrenches_EE.torque(2),
                constraint_satisfied,
                i,
                motion_specification_params,
                arm_name,
                condition_type_value);
            break;

          default:
            std::cout << "[check_pre_or_post_or_prevail_condition_satisfaction] Constraint checking not defined for given constraint" << std::endl;
            flag = 1; // stop the execution
            break;
          }
        }
        else
        {
          std::cout << "[check_pre_or_post_or_prevail_condition_satisfaction] Constraint type not found" << std::endl;
          flag = 1; // stop the execution
        }

        if (constraint_satisfied && i == condition_constraint_count)
        {
          condition_satisfied = true;
          return;
        }
        else if (!constraint_satisfied)
        {
          condition_satisfied = false;
          return;
        }
      }
    }
  }

  void MotionSpecificationActionServer::get_setpoints_from_motion_specification(
      double &lin_pos_sp_x_axis_data,
      double &lin_pos_sp_y_axis_data,
      double &lin_pos_sp_z_axis_data,
      double &lin_vel_sp_x_axis_data,
      double &lin_vel_sp_y_axis_data,
      double &lin_vel_sp_z_axis_data,
      double &force_to_apply_x_axis,
      double &force_to_apply_y_axis,
      double &force_to_apply_z_axis,
      const int &per_condition_constraint_count,
      std::array<double, 4> &desired_quat_GF,
      const YAML::Node &motion_specification_params,
      const std::string &arm_name)
  {
    auto constraint_type_map = getConstraintTypeMap();
    std::string condition_type_str = "PER_CONDITION";

    if (per_condition_constraint_count > 0)
    {
      for (int i = 1; i < per_condition_constraint_count + 1; i++)
      {
        std::string constraint_type_str = motion_specification_params[arm_name][condition_type_str]["constraints"][i]["type"].as<std::string>();
        auto constraint_iterator = constraint_type_map.find(constraint_type_str);

        if (constraint_iterator != constraint_type_map.end())
        {
          auto constraint_value_list = motion_specification_params[arm_name][condition_type_str]["constraints"][i]["value"];
          constraint_type constraint_type_ = constraint_iterator->second;
          switch (constraint_type_)
          {
          case POSITION_XYZ:
            for (int k = 0; k < 3; k++)
            {
              if (!constraint_value_list[k].IsNull())
              {
                if (k == 0)
                {
                  lin_pos_sp_x_axis_data = constraint_value_list[k].as<double>();
                }
                else if (k == 1)
                {
                  lin_pos_sp_y_axis_data = constraint_value_list[k].as<double>();
                }
                else if (k == 2)
                {
                  lin_pos_sp_z_axis_data = constraint_value_list[k].as<double>();
                }
              }
            }
            break;

          case VELOCITY_XYZ:
            for (int k = 0; k < 3; k++)
            {
              if (!constraint_value_list[k].IsNull())
              {
                if (k == 0)
                {
                  lin_vel_sp_x_axis_data = constraint_value_list[k].as<double>();
                }
                else if (k == 1)
                {
                  lin_vel_sp_y_axis_data = constraint_value_list[k].as<double>();
                }
                else if (k == 2)
                {
                  lin_vel_sp_z_axis_data = constraint_value_list[k].as<double>();
                }
              }
            }
            break;

          case FORCE_XYZ:
            for (int k = 0; k < 3; k++)
            {
              if (!constraint_value_list[k].IsNull())
              {
                if (k == 0)
                {
                  force_to_apply_x_axis = constraint_value_list[k].as<double>();
                }
                else if (k == 1)
                {
                  force_to_apply_y_axis = constraint_value_list[k].as<double>();
                }
                else if (k == 2)
                {
                  force_to_apply_z_axis = constraint_value_list[k].as<double>();
                }
              }
            }
            break;

          case ORIENTATION_QUATERNION:
            for (int k = 0; k < 4; k++)
            {
              if (!constraint_value_list[k].IsNull())
              {
                desired_quat_GF[k] = constraint_value_list[k].as<double>();
              }
            }
            break;

          default:
            break;
          }
        }
        else
        {
          std::cout << "[get_setpoints_from_motion_specification] Constraint type not found" << std::endl;
          flag = 1; // stop the execution
        }
      }
    }
  }

  void MotionSpecificationActionServer::get_force_and_torque_from_controller_described_in_GF_to_apply_at_EE(
      const double &stiffness_lin_x_axis_data,
      const double &stiffness_lin_y_axis_data,
      const double &stiffness_lin_z_axis_data,
      const double &damping_lin_x_axis_data,
      const double &damping_lin_y_axis_data,
      const double &damping_lin_z_axis_data,
      const double &stiffness_roll_axis_data,
      const double &stiffness_pitch_axis_data,
      const double &stiffness_yaw_axis_data,
      const double &measured_lin_pos_x_axis_data,
      const double &measured_lin_pos_y_axis_data,
      const double &measured_lin_pos_z_axis_data,
      const double &measured_lin_vel_x_axis_data,
      const double &measured_lin_vel_y_axis_data,
      const double &measured_lin_vel_z_axis_data,
      const double &lin_pos_sp_x_axis_data,
      const double &lin_pos_sp_y_axis_data,
      const double &lin_pos_sp_z_axis_data,
      const double &lin_vel_sp_x_axis_data,
      const double &lin_vel_sp_y_axis_data,
      const double &lin_vel_sp_z_axis_data,
      const double &force_to_apply_x_axis,
      const double &force_to_apply_y_axis,
      const double &force_to_apply_z_axis,
      const std::array<double, 4> &desired_quat_GF,
      double &apply_ee_force_x_axis_data,
      double &apply_ee_force_y_axis_data,
      double &apply_ee_force_z_axis_data,
      double &apply_ee_torque_x_axis_data,
      double &apply_ee_torque_y_axis_data,
      double &apply_ee_torque_z_axis_data,
      KDL::Frame &desired_endEffPose_GF_arm,
      const KDL::Frame &measured_endEffPose_GF_arm,
      const int &per_condition_constraint_count,
      KDL::Vector &angle_axis_diff_GF_arm,
      const YAML::Node &motion_specification_params,
      const std::string &arm_name)
  {
    auto constraint_type_map = getConstraintTypeMap();
    std::string condition_type_str = "PER_CONDITION";

    if (per_condition_constraint_count > 0)
    {
      for (int i = 1; i < per_condition_constraint_count + 1; i++)
      {
        std::string constraint_type_str = motion_specification_params[arm_name][condition_type_str]["constraints"][i]["type"].as<std::string>();
        auto constraint_iterator = constraint_type_map.find(constraint_type_str);

        if (constraint_iterator != constraint_type_map.end())
        {
          constraint_type constraint_type_ = constraint_iterator->second;
          auto constraint_value_list = motion_specification_params[arm_name][condition_type_str]["constraints"][i]["value"];

          switch (constraint_type_)
          {
          case POSITION_XYZ:
            for (int k = 0; k < 3; k++)
            {
              if (!constraint_value_list[k].IsNull())
              {
                if (k == 0)
                {
                  apply_ee_force_x_axis_data += stiffness_lin_x_axis_data * (lin_pos_sp_x_axis_data - measured_lin_pos_x_axis_data);
                }
                else if (k == 1)
                {
                  apply_ee_force_y_axis_data += stiffness_lin_y_axis_data * (lin_pos_sp_y_axis_data - measured_lin_pos_y_axis_data);
                }
                else if (k == 2)
                {
                  apply_ee_force_z_axis_data += stiffness_lin_z_axis_data * (lin_pos_sp_z_axis_data - measured_lin_pos_z_axis_data);
                }
              }
            }
            break;

          case VELOCITY_XYZ:
            for (int k = 0; k < 3; k++)
            {
              if (!constraint_value_list[k].IsNull())
              {
                if (k == 0)
                {
                  apply_ee_force_x_axis_data += damping_lin_x_axis_data * (lin_vel_sp_x_axis_data - measured_lin_vel_x_axis_data);
                }
                else if (k == 1)
                {
                  apply_ee_force_y_axis_data += damping_lin_y_axis_data * (lin_vel_sp_y_axis_data - measured_lin_vel_y_axis_data);
                }
                else if (k == 2)
                {
                  apply_ee_force_z_axis_data += damping_lin_z_axis_data * (lin_vel_sp_z_axis_data - measured_lin_vel_z_axis_data);
                }
              }
            }
            break;

          case FORCE_XYZ:
            for (int k = 0; k < 3; k++)
            {
              if (!constraint_value_list[k].IsNull())
              {
                if (k == 0)
                {
                  apply_ee_force_x_axis_data += force_to_apply_x_axis;
                }
                else if (k == 1)
                {
                  apply_ee_force_y_axis_data += force_to_apply_y_axis;
                }
                else if (k == 2)
                {
                  apply_ee_force_z_axis_data += force_to_apply_z_axis;
                }
              }
            }
            break;

          case ORIENTATION_QUATERNION:
            desired_endEffPose_GF_arm.M = KDL::Rotation::Quaternion(desired_quat_GF[0], desired_quat_GF[1], desired_quat_GF[2], desired_quat_GF[3]);
            angle_axis_diff_GF_arm = KDL::diff(measured_endEffPose_GF_arm.M, desired_endEffPose_GF_arm.M);
            apply_ee_torque_x_axis_data = stiffness_roll_axis_data * angle_axis_diff_GF_arm(0);
            apply_ee_torque_y_axis_data = stiffness_pitch_axis_data * angle_axis_diff_GF_arm(1);
            apply_ee_torque_z_axis_data = stiffness_yaw_axis_data * angle_axis_diff_GF_arm(2);

            break;

          default:

            break;
          }
        }
        else
        {
          std::cout << "[get_force_and_torque_from_controller_described_in_GF_to_apply_at_EE] Constraint type not found" << std::endl;
          flag = 1; // stop the execution
        }
      }
    }
  }

  void MotionSpecificationActionServer::read_motion_specification(const YAML::Node &motion_specification_params)
  {
    try
    {
      pre_condition_constraint_count = motion_specification_params[arm_name]["PRE_CONDITION"]["constraint_count"].as<int>();
      per_condition_constraint_count = motion_specification_params[arm_name]["PER_CONDITION"]["constraint_count"].as<int>();
      post_condition_constraint_count = motion_specification_params[arm_name]["POST_CONDITION"]["constraint_count"].as<int>();
      prevail_condition_constraint_count = motion_specification_params[arm_name]["PREVAIL_CONDITION"]["constraint_count"].as<int>();
    }
    catch (const YAML::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "Error reading motion specification parameters: %s", e.what());
      return;
    }
  }

  void MotionSpecificationActionServer::read_config_file(const YAML::Node &config_file_object)
  {
    try
    {
      arm_name = config_file_object["arm_name"].as<std::string>();

      STIFFNESS_GAIN_X = config_file_object[arm_name]["STIFFNESS_GAIN_X"].as<double>();
      STIFFNESS_GAIN_Y = config_file_object[arm_name]["STIFFNESS_GAIN_Y"].as<double>();
      STIFFNESS_GAIN_Z = config_file_object[arm_name]["STIFFNESS_GAIN_Z"].as<double>();
  
      DAMPING_GAIN_X = config_file_object[arm_name]["DAMPING_GAIN_X"].as<double>();
      DAMPING_GAIN_Y = config_file_object[arm_name]["DAMPING_GAIN_Y"].as<double>();
      DAMPING_GAIN_Z = config_file_object[arm_name]["DAMPING_GAIN_Z"].as<double>();
  
      STIFFNESS_GAIN_ROLL = config_file_object[arm_name]["STIFFNESS_GAIN_ROLL"].as<double>();
      STIFFNESS_GAIN_PITCH = config_file_object[arm_name]["STIFFNESS_GAIN_PITCH"].as<double>();
      STIFFNESS_GAIN_YAW = config_file_object[arm_name]["STIFFNESS_GAIN_YAW"].as<double>();
      STIFFNESS_GAIN_JOINT_IMPEDANCE_CTRL = config_file_object[arm_name]["STIFFNESS_GAIN_JOINT_IMPEDANCE_CTRL"].as<double>();
  
      std::vector<float> gravitational_acceleration = config_file_object[arm_name]["gravitational_acceleration"].as<std::vector<float>>();
      TIMEOUT_DURATION_TASK = config_file_object[arm_name]["TIMEOUT_DURATION_TASK"].as<double>(); // seconds
      WRENCH_THRESHOLD_LINEAR = config_file_object[arm_name]["WRENCH_THRESHOLD_LINEAR"].as<double>();
      WRENCH_THRESHOLD_ROTATIONAL = config_file_object[arm_name]["WRENCH_THRESHOLD_ROTATIONAL"].as<double>();
      JOINT_TORQUE_THRESHOLD = config_file_object[arm_name]["JOINT_TORQUE_THRESHOLD"].as<double>();
      DESIRED_TIME_STEP = config_file_object[arm_name]["DESIRED_TIME_STEP"].as<double>();
      SAVE_LOG_EVERY_NTH_STEP = config_file_object[arm_name]["SAVE_LOG_EVERY_NTH_STEP"].as<int>();
    }
    catch (const YAML::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "Error reading configuration parameters: %s", e.what());
      return;
    }
    
    configuration_file_read = true;
  }

  void MotionSpecificationActionServer::control_loop()
  {
    rclcpp::Rate loop_rate(1000); // Control loop at 1kHz
    while (rclcpp::ok() && control_loop_active_)
    {
      // This will execute continuously, performing necessary control logic.
      // You can check states, do calculations, or send periodic updates.

      // Example: Check system state
      if (!control_loop_active_)
      {
        break; // Exit the loop gracefully if the node is shutting down
      }

      // Control loop logic here (performing control actions or managing state)
      // Example: Print the current state (can replace with actual control logic)
      // RCLCPP_INFO(this->get_logger(), "Control loop is running at 1kHz");

      loop_rate.sleep(); // Maintain the loop at 1kHz
    }
  }

  void MotionSpecificationActionServer::execute(const std::shared_ptr<GoalHandleMotionSpecification> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal");
    rclcpp::Rate loop_rate(1);
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<MotionSpecification::Feedback>();
    auto &tcp_wrt_GF = feedback->tcp_position; // TODO: set the tcp position in feedback
    tcp_wrt_GF = {0.0, 0.0, 0.0};
    auto result = std::make_shared<MotionSpecification::Result>();

    try
    {
      motion_specification_params_object = YAML::Load(goal->motion_specification.c_str());
    }
    catch (const YAML::ParserException &e)
    {
      RCLCPP_ERROR(this->get_logger(), "YAML parsing error: %s", e.what());
      result->motion_successful = false;
      goal_handle->abort(result);
      return;
    }
    // Initialize the parameters
    motion_completed = false;
    goal_accepted_and_executing = true;
    
    // print string message on goal
    while (!motion_completed && rclcpp::ok())
    {
      RCLCPP_INFO(this->get_logger(), "Goal: %s", goal->motion_specification.c_str());

      if (goal_handle->is_canceling())
      {
        result->motion_successful = false;
        goal_handle->canceled(result);
        RCLCPP_INFO(this->get_logger(), "Goal canceled");
        goal_accepted_and_executing = false;
        return;
      }
      tcp_wrt_GF = {measured_lin_pos_x_axis_data, measured_lin_pos_y_axis_data, measured_lin_pos_z_axis_data};
      goal_handle->publish_feedback(feedback);
      RCLCPP_INFO(this->get_logger(), "Publish feedback");

      loop_rate.sleep();
    }

    // Check if goal is done. i.e., if it runs until the end of the fibonacci order, then rclcpp will be ok, so the goal is successful
    if (rclcpp::ok())
    {
      goal_accepted_and_executing = false;
      result->motion_successful = true;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Goal succeeded");
    }
  }
} // namespace motion_specification_action

RCLCPP_COMPONENTS_REGISTER_NODE(motion_specification_action::MotionSpecificationActionServer)
#include "motion_specification_action/motion_specification_server.hpp"

namespace motion_specification_action
{
  MotionSpecificationActionServer::MotionSpecificationActionServer(const rclcpp::NodeOptions &options)
      : Node("motion_specification_action_server", options),
        flag(0),
        goal_accepted_and_executing(false),
        jnt_positions(kinova_constants::NUMBER_OF_JOINTS),
        torques_gravity_compensation(kinova_constants::NUMBER_OF_JOINTS),
        jnt_velocities(kinova_constants::NUMBER_OF_JOINTS),
        jnt_torques_read(kinova_constants::NUMBER_OF_JOINTS), // to read from the robot
        jnt_torques_cmd(kinova_constants::NUMBER_OF_JOINTS),  // to send to the robot
        jnt_velocity(kinova_constants::NUMBER_OF_JOINTS),     // has both joint position and joint velocity of all joints
        jnt_accelerations(kinova_constants::NUMBER_OF_JOINTS),
        zero_jnt_velocities(kinova_constants::NUMBER_OF_JOINTS),
        switch_to_joint_impendance_control(false),
        jnt_angle_diff(0.0),
        pre_condition_constraint_count(0),
        per_condition_constraint_count(0),
        post_condition_constraint_count(0),
        iterationCount(0),
        frequency_of_state_publish(10),
        gravitational_acceleration{0.0f, 0.0f, -9.81f},
        time_period_of_complete_controller_cycle_data(0.0),
        desired_quat_FrameName{0.0, 0.0, 0.0, 1.0},
        measured_quat_FrameName{0.0, 0.0, 0.0, 1.0},
        measured_lin_pos_x_axis_data(0.0),
        measured_lin_pos_y_axis_data(0.0),
        measured_lin_pos_z_axis_data(0.0),
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
        torque_control_mode_set(false),
        configuration_file_read(false),
        pre_condition_satisfied(false),
        post_condition_satisfied(false),
        jnt_impedance_setpoint_is_set(false),
        communication_setup_done(false),
        state_publish_time_step(0.1),
        rne_output_jnt_torques_vector_to_set_control_mode(kinova_constants::NUMBER_OF_JOINTS, 0.0),
        arm_name("kinova_gen3_2_right"),
        frame_name("eddie_base_link"),
        transform_available(false),
        transform_timeout_duration(std::chrono::seconds(10))
  {
    using namespace std::placeholders;
    package_share_directory = ament_index_cpp::get_package_share_directory("motion_specification_action");
    config_file_path = package_share_directory + "/config/ms_config.yaml";
    urdf_file_path = package_share_directory + "/urdf/Kinova_1.urdf";
    config_file_object = YAML::LoadFile(config_file_path);
    parse_urdf_file(urdf_file_path, kinematic_tree, chain_urdf, NUM_LINKS);
    zero_jnt_velocities.data.setZero();

    read_config_file(config_file_object);
    initialise_solvers(jacobDotSolver, fkSolverPos, fkSolverVel, ikSolverAcc, idSolver, gravitational_acceleration, chain_urdf);

    BL_x_axis_wrt_GF = KDL::Vector(BL_x_axis_wrt_GF_vector[0], BL_x_axis_wrt_GF_vector[1], BL_x_axis_wrt_GF_vector[2]);
    BL_y_axis_wrt_GF = KDL::Vector(BL_y_axis_wrt_GF_vector[0], BL_y_axis_wrt_GF_vector[1], BL_y_axis_wrt_GF_vector[2]);
    BL_z_axis_wrt_GF = KDL::Vector(BL_z_axis_wrt_GF_vector[0], BL_z_axis_wrt_GF_vector[1], BL_z_axis_wrt_GF_vector[2]);
    BL_position_wrt_GF = KDL::Vector(BL_position_wrt_GF_vector[0], BL_position_wrt_GF_vector[1], BL_position_wrt_GF_vector[2]);

    // Initialize the KDL frame
    BL_wrt_GF = KDL::Rotation(BL_x_axis_wrt_GF, BL_y_axis_wrt_GF, BL_z_axis_wrt_GF);

    BL_wrt_GF_frame = KDL::Frame(
        BL_wrt_GF,           // rotation
        BL_position_wrt_GF); 

    BL_wrt_FrameName_frame = BL_wrt_GF_frame;

    linkWrenches_FrameName = KDL::Wrenches(NUM_LINKS, KDL::Wrench::Zero());
    linkWrenches_EE = KDL::Wrenches(NUM_LINKS, KDL::Wrench::Zero());
    linkWrenches_zero = KDL::Wrenches(NUM_LINKS, KDL::Wrench::Zero());

    auto handle_goal = [this](
                           const rclcpp_action::GoalUUID &uuid,
                           std::shared_ptr<const MotionSpecification::Goal> goal)
    {
      if (goal_accepted_and_executing)
      {
        RCLCPP_WARN(this->get_logger(), "A goal is already in progress. Rejecting new goal.");
        return rclcpp_action::GoalResponse::REJECT;
      }
      else
      {
        RCLCPP_INFO(this->get_logger(), "Received motion_specification as goal");
        (void)uuid;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      }
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

    joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("ee_pose", 10);

    joint_names_ = {"Actuator1", "Actuator2", "Actuator3", "Actuator4", "Actuator5", "Actuator6", "Actuator7"};

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  }

  MotionSpecificationActionServer::~MotionSpecificationActionServer()
  {
  }

  // Define all the functions related to the control loop

  // void MotionSpecificationActionServer::handle_signal(int sig)
  // {
  //     flag = 1;
  //     std::cout << "Received signal: " << sig << std::endl;
  // }


  void MotionSpecificationActionServer::publish_ee_pose(const double &measured_lin_pos_x_axis_data, const double &measured_lin_pos_y_axis_data, const double &measured_lin_pos_z_axis_data, const std::array<double, 4> &measured_quat_FrameName, const std::string &frame_name) 
  {
    auto pose_msg = geometry_msgs::msg::PoseStamped();

    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = frame_name;

    // Example EE position (replace with real FK values)
    pose_msg.pose.position.x = measured_lin_pos_x_axis_data;
    pose_msg.pose.position.y = measured_lin_pos_y_axis_data;
    pose_msg.pose.position.z = measured_lin_pos_z_axis_data;

    // Example orientation (identity quaternion)
    pose_msg.pose.orientation.x = measured_quat_FrameName[0];
    pose_msg.pose.orientation.y = measured_quat_FrameName[1];
    pose_msg.pose.orientation.z = measured_quat_FrameName[2];
    pose_msg.pose.orientation.w = measured_quat_FrameName[3];

    pose_publisher_->publish(pose_msg);
  }

  void MotionSpecificationActionServer::publish_joint_states(KDL::JntArray& jnt_positions) 
  {
    auto message = sensor_msgs::msg::JointState();
    message.header.stamp = this->now();
    message.name = joint_names_;

    double time_now = this->now().seconds();
    message.position = {
        jnt_positions(0),
        jnt_positions(1),
        jnt_positions(2),
        jnt_positions(3),
        jnt_positions(4),
        jnt_positions(5),
        jnt_positions(6)
    };

    joint_state_pub_->publish(message);
  }

  void MotionSpecificationActionServer::reset_flags()
  {
    flag = 0;
    torque_control_mode_set = false;
    switch_to_joint_impendance_control = false;
    jnt_impedance_setpoint_is_set = false;
    pre_condition_satisfied = false;
    post_condition_satisfied = false;
  }

  void MotionSpecificationActionServer::kinova_setup_communication(
      const robot_controlled &robot_to_control,
      kinova_mediator &kinova_arm_mediator,
      bool &communication_setup_done)
  {
    // robot communication
    if (robot_to_control == robot_controlled::KINOVA_GEN3_1_LEFT)
    {
      kinova_arm_mediator.kinova_id = robot_id::KINOVA_GEN3_1;
      kinova_arm_mediator.initialize(kinova_environment::REAL, robot_id::KINOVA_GEN3_1,
                                     0.0);
    }
    else if (robot_to_control == robot_controlled::KINOVA_GEN3_2_RIGHT)
    {
      kinova_arm_mediator.kinova_id = robot_id::KINOVA_GEN3_2;
      kinova_arm_mediator.initialize(kinova_environment::REAL, robot_id::KINOVA_GEN3_2,
                                     0.0);
    }
    else
    {
      std::cout << "Invalid robot to control" << std::endl;
      flag = 1; // stop the execution
      communication_setup_done = false;
    }
    std::cout << "Kinova arm mediator initialized" << std::endl;
    communication_setup_done = true;
  }

  void MotionSpecificationActionServer::initialise_solvers(
      std::shared_ptr<KDL::ChainJntToJacDotSolver> &jacobDotSolver,
      std::shared_ptr<KDL::ChainFkSolverPos_recursive> &fkSolverPos,
      std::shared_ptr<KDL::ChainFkSolverVel_recursive> &fkSolverVel,
      std::shared_ptr<KDL::ChainIkSolverVel_pinv> &ikSolverAcc,
      std::shared_ptr<KDL::ChainIdSolver_RNE> &idSolver,
      const std::vector<float> &gravitational_acceleration,
      const KDL::Chain &chain_urdf)
  {
    jacobDotSolver = std::make_shared<KDL::ChainJntToJacDotSolver>(chain_urdf);
    fkSolverPos = std::make_shared<KDL::ChainFkSolverPos_recursive>(chain_urdf);
    fkSolverVel = std::make_shared<KDL::ChainFkSolverVel_recursive>(chain_urdf);
    ikSolverAcc = std::make_shared<KDL::ChainIkSolverVel_pinv>(chain_urdf);

    KDL::Vector gravity(gravitational_acceleration[0],
                        gravitational_acceleration[1],
                        gravitational_acceleration[2]);

    idSolver = std::make_shared<KDL::ChainIdSolver_RNE>(chain_urdf, gravity);
  }

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
                                                                        KDL::Frame &measured_endEffPose_FrameName,
                                                                        KDL::FrameVel &measured_endEffTwist_FrameName,
                                                                        std::shared_ptr<KDL::ChainFkSolverPos_recursive> &fkSolverPos,
                                                                        std::shared_ptr<KDL::ChainFkSolverVel_recursive> &fkSolverVel,
                                                                        const KDL::Frame &BL_wrt_FrameName_frame)
  {
    jnt_velocity.q = jnt_positions;
    jnt_velocity.qdot = jnt_velocities;

    fkSolverPos->JntToCart(jnt_positions, measured_endEffPose_BL);
    fkSolverVel->JntToCart(jnt_velocity, measured_endEffTwist_BL);

    // // if (frame_name=="marker_frame_0")
    // {
    //   std::cout << "BL_wrt_FrameName_frame" << BL_wrt_FrameName_frame.p.x() << ", " << BL_wrt_FrameName_frame.p.y() << ", " << BL_wrt_FrameName_frame.p.z() << std::endl;
    //   std::cout << "measured_endEffPose_BL" << measured_endEffPose_BL.p.x() << ", " << measured_endEffPose_BL.p.y() << ", " << measured_endEffPose_BL.p.z() << std::endl;
    // }

    measured_endEffPose_FrameName = BL_wrt_FrameName_frame * measured_endEffPose_BL;
    measured_endEffTwist_FrameName = BL_wrt_FrameName_frame * measured_endEffTwist_BL;
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
        {"LESS_THAN", operator_type::LESS_THAN},
        {"EQUAL", operator_type::EQUAL}};
    return operator_type_map;
  }

  const std::unordered_map<std::string, condition_type> &MotionSpecificationActionServer::getConditionTypeMap()
  {
    static const std::unordered_map<std::string, condition_type> condition_type_map = {
        {"PRE_CONDITION", condition_type::PRE_CONDITION},
        {"PER_CONDITION", condition_type::PER_CONDITION},
        {"POST_CONDITION", condition_type::POST_CONDITION}};
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
      const YAML::Node &motion_specification_params_object,
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
      auto operator_value = motion_specification_params_object[arm_name][condition_type_str]["constraints"][constraint_idx]["operator"][j];
      std::string op_str = operator_value.as<std::string>("");
      if (!(op_str == "None"))
      {
        operator_type_str = motion_specification_params_object[arm_name][condition_type_str]["constraints"][constraint_idx]["operator"][j].as<std::string>();
        auto operator_iterator = operator_type_map.find(operator_type_str);
        if (operator_iterator != operator_type_map.end())
        {
          operator_type_ = operator_iterator->second;

          switch (operator_type_)
          {
          case GREATER_THAN:
            desired_data = motion_specification_params_object[arm_name][condition_type_str]["constraints"][constraint_idx]["value"][j].as<double>();
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
            desired_data = motion_specification_params_object[arm_name][condition_type_str]["constraints"][constraint_idx]["value"][j].as<double>();
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
          std::cout << "[check_3D_vector_constraint_satisfaction] Operator type not found: " << operator_type_str << std::endl;
          flag = 1; // stop the execution
        }
      }
    }
  }

  void MotionSpecificationActionServer::check_1D_vector_constraint_satisfaction(
      const double &measured_data,
      bool &constraint_satisfied,
      const int &constraint_idx,
      const YAML::Node &motion_specification_params_object,
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
    else
    {
      std::cout << "[check_1D_vector_constraint_satisfaction] Condition type not found" << std::endl;
      flag = 1; // stop the execution
    }

    auto operator_value = motion_specification_params_object[arm_name][condition_type_str]["constraints"][constraint_idx]["operator"];
    std::string op_str = operator_value.as<std::string>("");
    if (!(op_str == "None"))
    {
      operator_type_str = motion_specification_params_object[arm_name][condition_type_str]["constraints"][constraint_idx]["operator"].as<std::string>();
      auto operator_iterator = operator_type_map.find(operator_type_str);

      if (operator_iterator != operator_type_map.end())
      {
        operator_type_ = operator_iterator->second;

        switch (operator_type_)
        {
        case GREATER_THAN:
          desired_data = motion_specification_params_object[arm_name][condition_type_str]["constraints"][constraint_idx]["value"].as<double>();
          greater_than_monitor(&measured_data, &desired_data, &constraint_satisfied);
          break;

        case LESS_THAN:
          desired_data = motion_specification_params_object[arm_name][condition_type_str]["constraints"][constraint_idx]["value"].as<double>();
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

  void MotionSpecificationActionServer::check_pre_or_post_condition_satisfaction(
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
      const YAML::Node &motion_specification_params_object,
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
        else
        {
          std::cout << "[check_pre_or_post_condition_satisfaction] Condition type not found" << std::endl;
          flag = 1; // stop the execution
        }
        constraint_type_str = motion_specification_params_object[arm_name][condition_type_str]["constraints"][i]["type"].as<std::string>();

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
                motion_specification_params_object,
                arm_name,
                condition_type_value);
            break;

          case VELOCITY_XYZ:
            if (condition_type_value == condition_type::PRE_CONDITION)
            {
              std::cout << "[check_pre_or_post_condition_satisfaction] Velocity constraint not allowed in pre-condition" << std::endl;
              flag = 1; // stop the execution
              break;
            }
            check_3D_vector_constraint_satisfaction(
                measured_lin_vel_x_axis_data,
                measured_lin_vel_y_axis_data,
                measured_lin_vel_z_axis_data,
                constraint_satisfied,
                i,
                motion_specification_params_object,
                arm_name,
                condition_type_value);
            break;

          case ORIENTATION_ROLL:
            check_1D_vector_constraint_satisfaction(measured_roll_data, constraint_satisfied, i, motion_specification_params_object, arm_name, condition_type_value);
            break;

          case ORIENTATION_PITCH:
            check_1D_vector_constraint_satisfaction(measured_pitch_data, constraint_satisfied, i, motion_specification_params_object, arm_name, condition_type_value);
            break;

          case ORIENTATION_YAW:
            check_1D_vector_constraint_satisfaction(measured_yaw_data, constraint_satisfied, i, motion_specification_params_object, arm_name, condition_type_value);
            break;

          case FORCE_XYZ:
            if (condition_type_value == condition_type::PRE_CONDITION)
            {
              std::cout << "[check_pre_or_post_condition_satisfaction] Force constraint not allowed in pre-condition" << std::endl;
              flag = 1; // stop the execution
              break;
            }
            check_3D_vector_constraint_satisfaction(
                linkWrenches_EE.force(0),
                linkWrenches_EE.force(1),
                linkWrenches_EE.force(2),
                constraint_satisfied,
                i,
                motion_specification_params_object,
                arm_name,
                condition_type_value);
            break;

          case TORQUE_RPY:
            if (condition_type_value == condition_type::PRE_CONDITION)
            {
              std::cout << "[check_pre_or_post_condition_satisfaction] Torque constraint not allowed in pre-condition" << std::endl;
              flag = 1; // stop the execution
              break;
            }
            check_3D_vector_constraint_satisfaction(
                linkWrenches_EE.torque(0),
                linkWrenches_EE.torque(1),
                linkWrenches_EE.torque(2),
                constraint_satisfied,
                i,
                motion_specification_params_object,
                arm_name,
                condition_type_value);
            break;

          default:
            std::cout << "[check_pre_or_post_condition_satisfaction] Constraint checking not defined for given constraint" << std::endl;
            flag = 1; // stop the execution
            break;
          }
        }
        else
        {
          std::cout << "[check_pre_or_post_condition_satisfaction] Constraint type not found" << std::endl;
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
    std::array<double, 4> &desired_quat_FrameName,
    const YAML::Node &motion_specification_params_object,
    const std::string &arm_name)
  {
    auto constraint_type_map = getConstraintTypeMap();
    std::string condition_type_str = "PER_CONDITION";

    if (per_condition_constraint_count > 0)
    {
      for (int i = 1; i < per_condition_constraint_count + 1; i++)
      {
        std::string constraint_type_str = motion_specification_params_object[arm_name][condition_type_str]["constraints"][i]["type"].as<std::string>();
        auto constraint_iterator = constraint_type_map.find(constraint_type_str);

        if (constraint_iterator != constraint_type_map.end())
        {
          auto constraint_value_list = motion_specification_params_object[arm_name][condition_type_str]["constraints"][i]["value"];
          constraint_type constraint_type_ = constraint_iterator->second;
          switch (constraint_type_)
          {
          case POSITION_XYZ:
            for (int k = 0; k < 3; k++)
            {
              std::string constraint_str = constraint_value_list[k].as<std::string>("");
              if (!(constraint_str == "None"))
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
              std::string constraint_str = constraint_value_list[k].as<std::string>("");
              if (!(constraint_str == "None"))
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
              std::string constraint_str = constraint_value_list[k].as<std::string>("");
              if (!(constraint_str == "None"))
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
              std::string constraint_str = constraint_value_list[k].as<std::string>("");
              if (!(constraint_str == "None"))

              {
                desired_quat_FrameName[k] = constraint_value_list[k].as<double>();
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

  void MotionSpecificationActionServer::get_force_and_torque_from_controller_described_in_FrameName_to_apply_at_EE(
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
      const std::array<double, 4> &desired_quat_FrameName,
      double &apply_ee_force_x_axis_data,
      double &apply_ee_force_y_axis_data,
      double &apply_ee_force_z_axis_data,
      double &apply_ee_torque_x_axis_data,
      double &apply_ee_torque_y_axis_data,
      double &apply_ee_torque_z_axis_data,
      KDL::Frame &desired_endEffPose_FrameName_arm,
      const KDL::Frame &measured_endEffPose_FrameName,
      const int &per_condition_constraint_count,
      KDL::Vector &angle_axis_diff_FrameName_arm,
      const YAML::Node &motion_specification_params_object,
      const std::string &arm_name)
  {
    auto constraint_type_map = getConstraintTypeMap();
    std::string condition_type_str = "PER_CONDITION";

    if (per_condition_constraint_count > 0)
    {
      for (int i = 1; i < per_condition_constraint_count + 1; i++)
      {
        std::string constraint_type_str = motion_specification_params_object[arm_name][condition_type_str]["constraints"][i]["type"].as<std::string>();
        auto constraint_iterator = constraint_type_map.find(constraint_type_str);

        if (constraint_iterator != constraint_type_map.end())
        {
          constraint_type constraint_type_ = constraint_iterator->second;
          auto constraint_value_list = motion_specification_params_object[arm_name][condition_type_str]["constraints"][i]["value"];

          switch (constraint_type_)
          {
          case POSITION_XYZ:
            for (int k = 0; k < 3; k++)
            {
              std::string constraint_str = constraint_value_list[k].as<std::string>("");
              if (!(constraint_str == "None"))
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
              std::string constraint_str = constraint_value_list[k].as<std::string>("");
              if (!(constraint_str == "None"))
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
              std::string constraint_str = constraint_value_list[k].as<std::string>("");
              if (!(constraint_str == "None"))
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
            desired_endEffPose_FrameName_arm.M = KDL::Rotation::Quaternion(desired_quat_FrameName[0], desired_quat_FrameName[1], desired_quat_FrameName[2], desired_quat_FrameName[3]);
            angle_axis_diff_FrameName_arm = KDL::diff(measured_endEffPose_FrameName.M, desired_endEffPose_FrameName_arm.M);
            apply_ee_torque_x_axis_data = stiffness_roll_axis_data * angle_axis_diff_FrameName_arm(0);
            apply_ee_torque_y_axis_data = stiffness_pitch_axis_data * angle_axis_diff_FrameName_arm(1);
            apply_ee_torque_z_axis_data = stiffness_yaw_axis_data * angle_axis_diff_FrameName_arm(2);

            break;

          default:

            break;
          }
        }
        else
        {
          std::cout << "[get_force_and_torque_from_controller_described_in_FrameName_to_apply_at_EE] Constraint type not found" << std::endl;
          flag = 1; // stop the execution
        }
      }
    }
  }

  void MotionSpecificationActionServer::read_ms_conditions_count(const YAML::Node &motion_specification_params_object)
  {
    pre_condition_constraint_count = motion_specification_params_object[arm_name]["PRE_CONDITION"]["constraint_count"].as<int>();
    per_condition_constraint_count = motion_specification_params_object[arm_name]["PER_CONDITION"]["constraint_count"].as<int>();
    post_condition_constraint_count = motion_specification_params_object[arm_name]["POST_CONDITION"]["constraint_count"].as<int>();
  }

  void MotionSpecificationActionServer::read_frame_name(const YAML::Node &motion_specification_params_object)
  {
    frame_name = motion_specification_params_object[arm_name]["frame_name"].as<std::string>();
  }

  void MotionSpecificationActionServer::parse_urdf_file(const std::string &urdf_file_path, KDL::Tree &kinematic_tree, KDL::Chain &chain_urdf, unsigned int &NUM_LINKS)
  {
    RCLCPP_INFO(this->get_logger(), "Parsing URDF file");
    if (urdf_file_path.empty())
    {
      RCLCPP_ERROR(this->get_logger(), "URDF file path is empty");
      return;
    }
    try
    {
      kdl_parser::treeFromFile(urdf_file_path, kinematic_tree);
      kinematic_tree.getChain("base_link", "EndEffector_Link", chain_urdf);
      NUM_LINKS = chain_urdf.getNrOfSegments();
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "Error parsing URDF file: %s", e.what());
      return;
    }
  }

  void MotionSpecificationActionServer::read_config_file(const YAML::Node &config_file_object)
  {
    try
    {
      arm_name = config_file_object["arm_name"].as<std::string>();

      if (arm_name == "KINOVA_GEN3_1_LEFT")
      {
        robot_to_control = robot_controlled::KINOVA_GEN3_1_LEFT;
      }
      else if (arm_name == "KINOVA_GEN3_2_RIGHT")
      {
        robot_to_control = robot_controlled::KINOVA_GEN3_2_RIGHT;
      }
      else
      {
        RCLCPP_ERROR(this->get_logger(), "Invalid arm name in configuration file");
        return;
      };

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

      gravitational_acceleration = config_file_object[arm_name]["gravitational_acceleration"].as<std::vector<float>>();
      WRENCH_THRESHOLD_LINEAR = config_file_object[arm_name]["WRENCH_THRESHOLD_LINEAR"].as<double>();
      WRENCH_THRESHOLD_ROTATIONAL = config_file_object[arm_name]["WRENCH_THRESHOLD_ROTATIONAL"].as<double>();
      JOINT_TORQUE_THRESHOLD = config_file_object[arm_name]["JOINT_TORQUE_THRESHOLD"].as<double>();

      stiffness_lin_x_axis_data = STIFFNESS_GAIN_X;
      stiffness_lin_y_axis_data = STIFFNESS_GAIN_Y;
      stiffness_lin_z_axis_data = STIFFNESS_GAIN_Z;

      damping_lin_x_axis_data = DAMPING_GAIN_X;
      damping_lin_y_axis_data = DAMPING_GAIN_Y;
      damping_lin_z_axis_data = DAMPING_GAIN_Z;

      stiffness_roll_axis_data = STIFFNESS_GAIN_ROLL;
      stiffness_pitch_axis_data = STIFFNESS_GAIN_PITCH;
      stiffness_yaw_axis_data = STIFFNESS_GAIN_YAW;
      stiffness_joint_impedance_ctrl = STIFFNESS_GAIN_JOINT_IMPEDANCE_CTRL;

      BL_x_axis_wrt_GF_vector = config_file_object[arm_name]["BL_x_axis_wrt_GF"].as<std::vector<double>>();
      BL_y_axis_wrt_GF_vector = config_file_object[arm_name]["BL_y_axis_wrt_GF"].as<std::vector<double>>();
      BL_z_axis_wrt_GF_vector = config_file_object[arm_name]["BL_z_axis_wrt_GF"].as<std::vector<double>>();
      BL_position_wrt_GF_vector = config_file_object[arm_name]["BL_position_wrt_GF"].as<std::vector<double>>();
    }
    catch (const YAML::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "Error reading configuration parameters: %s", e.what());
      configuration_file_read = false;
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Configuration file read successfully");
    configuration_file_read = true;
  }


  void MotionSpecificationActionServer::get_transform_BL_wrt_desired_frame(
      const std::string &frame_name,
      KDL::Frame &BL_wrt_FrameName_frame,
      geometry_msgs::msg::TransformStamped &transform_stamped,
      std::chrono::duration<double> &transform_timeout_duration,
      bool &transform_available)
  {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();

    while (current_time - start_time < transform_timeout_duration)
    {
      current_time = std::chrono::high_resolution_clock::now();
      // Check if the transform is available
      if (tf_buffer_->canTransform(frame_name, "base_link", tf2::TimePointZero))
      {
        try {
          transform_stamped = tf_buffer_->lookupTransform(
            frame_name,
            "base_link",
            tf2::TimePointZero);
          transform_available = true;
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
            transform_available = false;
        }
        if (transform_available)
        {
          break; // Exit the loop if the transform is available
        }
        // sleep for a short duration to allow the transform to be available
        RCLCPP_INFO(this->get_logger(), "Waiting for transform from %s to base_link", frame_name.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Spin the node to process incoming messages
        rclcpp::spin_some(this->get_node_base_interface());
      }
    }
    BL_wrt_FrameName_frame = tf2::transformToKDL(transform_stamped);

    // print transform_stamped
    std::cout << "Frame name: " << frame_name << std::endl;

    const auto &t = transform_stamped.transform.translation;
    // std::cout << "Translation: x=" << BL_wrt_FrameName_frame.p.x() << ", y=" << BL_wrt_FrameName_frame.p.y() << ", z=" << BL_wrt_FrameName_frame.p.z() << std::endl;
    // std::cout << "Translation: x=" << t.x << ", y=" << t.y << ", z=" << t.z << std::endl;

    const auto &r = transform_stamped.transform.rotation;
    // std::cout << "Rotation: x=" << r.x << ", y=" << r.y << ", z=" << r.z << ", w=" << r.w << std::endl;
  }

  void MotionSpecificationActionServer::control_loop()
  {
    if (!torque_control_mode_set)
    {
      RCLCPP_INFO(this->get_logger(), "Setting torque control mode");
      kinova_feedback(kinova_arm_mediator, jnt_positions, jnt_velocities,
        jnt_torques_read);
        
      get_end_effector_pose_and_twist(
          jnt_velocity, jnt_positions, jnt_velocities,
          measured_endEffPose_BL_arm, measured_endEffTwist_BL_arm,
          measured_endEffPose_FrameName, measured_endEffTwist_FrameName_arm,
          fkSolverPos, fkSolverVel, BL_wrt_FrameName_frame);

      calculate_joint_torques_RNEA(jacobDotSolver, ikSolverAcc, idSolver,
                                    jnt_velocity, jd_qd, xdd,
                                    xdd_minus_jd_qd, jnt_accelerations,
                                    jnt_positions, jnt_velocities,
                                    linkWrenches_zero, jnt_torques_cmd);

      // convert JntArray to double array
      for (int i = 0; i < kinova_constants::NUMBER_OF_JOINTS; i++)
      {
          rne_output_jnt_torques_vector_to_set_control_mode[i] =
              jnt_torques_cmd(i);
      }
      kinova_arm_mediator.set_control_mode(control_mode::TORQUE, rne_output_jnt_torques_vector_to_set_control_mode.data());
      torque_control_mode_set = true;
    }
    else
    {
      kinova_feedback(kinova_arm_mediator, jnt_positions, jnt_velocities,
        jnt_torques_read);
        
      get_end_effector_pose_and_twist(
          jnt_velocity, jnt_positions, jnt_velocities,
          measured_endEffPose_BL_arm, measured_endEffTwist_BL_arm,
          measured_endEffPose_FrameName, measured_endEffTwist_FrameName_arm,
          fkSolverPos, fkSolverVel, BL_wrt_FrameName_frame);

      measured_lin_pos_x_axis_data = measured_endEffPose_FrameName.p.x();
      measured_lin_vel_x_axis_data = measured_endEffTwist_FrameName_arm.GetTwist().vel.x();
      measured_lin_pos_y_axis_data = measured_endEffPose_FrameName.p.y();
      measured_lin_vel_y_axis_data = measured_endEffTwist_FrameName_arm.GetTwist().vel.y();
      measured_lin_pos_z_axis_data = measured_endEffPose_FrameName.p.z();
      measured_lin_vel_z_axis_data = measured_endEffTwist_FrameName_arm.GetTwist().vel.z();
      measured_endEffPose_FrameName.M.GetQuaternion(measured_quat_FrameName[0], measured_quat_FrameName[1], measured_quat_FrameName[2], measured_quat_FrameName[3]);
      measured_endEffPose_FrameName.M.GetRPY(measured_roll_data, measured_pitch_data, measured_yaw_data);

      // check if any motion specification satisfies pre condition
      if (!pre_condition_satisfied)
      {
        check_pre_or_post_condition_satisfaction(
            measured_lin_pos_x_axis_data,
            measured_lin_pos_y_axis_data,
            measured_lin_pos_z_axis_data,
            measured_roll_data,
            measured_pitch_data,
            measured_yaw_data,
            measured_lin_vel_x_axis_data,
            measured_lin_vel_y_axis_data,
            measured_lin_vel_z_axis_data,
            linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS],
            pre_condition_constraint_count,
            constraint_type_str,
            arm_name,
            pre_condition_satisfied,
            motion_specification_params_object,
            condition_type::PRE_CONDITION);

        if (pre_condition_satisfied)
        {
          std::cout << "Pre condition satisfied. Now running controller to achieve per-condition until post-condition is satisfied." << std::endl;
        }
      }

      if (pre_condition_satisfied)
      {
        // check if the motion specification satisfies post condition
        if (!post_condition_satisfied)
        {
          check_pre_or_post_condition_satisfaction(
              measured_lin_pos_x_axis_data,
              measured_lin_pos_y_axis_data,
              measured_lin_pos_z_axis_data,
              measured_roll_data,
              measured_pitch_data,
              measured_yaw_data,
              measured_lin_vel_x_axis_data,
              measured_lin_vel_y_axis_data,
              measured_lin_vel_z_axis_data,
              linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS],
              post_condition_constraint_count,
              constraint_type_str,
              arm_name,
              post_condition_satisfied,
              motion_specification_params_object,
              condition_type::POST_CONDITION);
        }

        if (post_condition_satisfied)
        {
          if (!switch_to_joint_impendance_control)
          {
            std::cout << "Post condition satisfied. Switching to impedance control mode." << std::endl;
            switch_to_joint_impendance_control = true;
          }
        }
        else
        {
          get_setpoints_from_motion_specification(
              // measured_lin_pos_x_axis_data,
              // measured_lin_pos_y_axis_data,
              // measured_lin_pos_z_axis_data,
              // measured_lin_vel_x_axis_data,
              // measured_lin_vel_y_axis_data,
              // measured_lin_vel_z_axis_data,
              // measured_roll_data,
              // measured_pitch_data,
              // measured_yaw_data,
              lin_pos_sp_x_axis_data,
              lin_pos_sp_y_axis_data,
              lin_pos_sp_z_axis_data,
              lin_vel_sp_x_axis_data,
              lin_vel_sp_y_axis_data,
              lin_vel_sp_z_axis_data,
              force_to_apply_x_axis,
              force_to_apply_y_axis,
              force_to_apply_z_axis,
              per_condition_constraint_count,
              desired_quat_FrameName,
              motion_specification_params_object,
              arm_name);

          get_force_and_torque_from_controller_described_in_FrameName_to_apply_at_EE(
              stiffness_lin_x_axis_data,
              stiffness_lin_y_axis_data,
              stiffness_lin_z_axis_data,
              damping_lin_x_axis_data,
              damping_lin_y_axis_data,
              damping_lin_z_axis_data,
              stiffness_roll_axis_data,
              stiffness_pitch_axis_data,
              stiffness_yaw_axis_data,
              measured_lin_pos_x_axis_data,
              measured_lin_pos_y_axis_data,
              measured_lin_pos_z_axis_data,
              measured_lin_vel_x_axis_data,
              measured_lin_vel_y_axis_data,
              measured_lin_vel_z_axis_data,
              lin_pos_sp_x_axis_data,
              lin_pos_sp_y_axis_data,
              lin_pos_sp_z_axis_data,
              lin_vel_sp_x_axis_data,
              lin_vel_sp_y_axis_data,
              lin_vel_sp_z_axis_data,
              force_to_apply_x_axis,
              force_to_apply_y_axis,
              force_to_apply_z_axis,
              desired_quat_FrameName,
              apply_ee_force_x_axis_data,
              apply_ee_force_y_axis_data,
              apply_ee_force_z_axis_data,
              apply_ee_torque_x_axis_data,
              apply_ee_torque_y_axis_data,
              apply_ee_torque_z_axis_data,
              desired_endEffPose_FrameName_arm,
              measured_endEffPose_FrameName,
              per_condition_constraint_count,
              angle_axis_diff_FrameName_arm,
              motion_specification_params_object,
              arm_name);
        }
      }

      if (switch_to_joint_impendance_control)
      {
        if (!jnt_impedance_setpoint_is_set)
        {
          std::cout << "In joint impedance mode" << std::endl;
          jnt_positions_setpoint = jnt_positions;
          jnt_impedance_setpoint_is_set = true;
        }

        calculate_joint_torques_RNEA(jacobDotSolver, ikSolverAcc, idSolver,
                                     jnt_velocity, jd_qd, xdd,
                                     xdd_minus_jd_qd, jnt_accelerations,
                                     jnt_positions, jnt_velocities,
                                     linkWrenches_zero, torques_gravity_compensation);

        for (int i = 0; i < kinova_constants::NUMBER_OF_JOINTS; i++)
        {
          jnt_angle_diff = jnt_positions_setpoint(i) - jnt_positions(i);
          if (jnt_angle_diff > abs(jnt_angle_diff - 2 * M_PI))
          {
            jnt_angle_diff = jnt_angle_diff - 2 * M_PI;
          }
          jnt_torques_cmd(i) = stiffness_joint_impedance_ctrl * jnt_angle_diff + torques_gravity_compensation(i);
        }
      }
      else
      {
        // write the ee torques to linkWrenches_FrameName
        linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].force(0) = -apply_ee_force_x_axis_data;
        linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].force(1) = -apply_ee_force_y_axis_data;
        linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].force(2) = -apply_ee_force_z_axis_data;
        // TODO: apply only when a per condition constraint exists for orientation
        linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].torque(0) = -apply_ee_torque_x_axis_data;
        linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].torque(1) = -apply_ee_torque_y_axis_data;
        linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].torque(2) = -apply_ee_torque_z_axis_data;

        // thresholding in cartesian space of the end effector
        for (int i = 0; i < 3; i++)
        {
          if (linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].force(i) > 0.0)
          {
            linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].force(i) = std::min(WRENCH_THRESHOLD_LINEAR, linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].force(i));
          }
          else
          {
            linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].force(i) = std::max(-WRENCH_THRESHOLD_LINEAR, linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].force(i));
          }

          if (linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].torque(i) > 0.0)
          {
            linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].torque(i) = std::min(WRENCH_THRESHOLD_ROTATIONAL, linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].torque(i));
          }
          else
          {
            linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].torque(i) = std::max(-WRENCH_THRESHOLD_ROTATIONAL, linkWrenches_FrameName[kinova_constants::NUMBER_OF_JOINTS].torque(i));
          }
        };

        // LinkWrenches are calculated in BL frame. As RNE solver requires them in EE frame, the wrenches are transformed from BL to EE frame
        linkWrenches_EE[NUM_LINKS - 1].force = measured_endEffPose_FrameName.M.Inverse() * linkWrenches_FrameName[NUM_LINKS - 1].force;
        linkWrenches_EE[NUM_LINKS - 1].torque = measured_endEffPose_FrameName.M.Inverse() * linkWrenches_FrameName[NUM_LINKS - 1].torque;

        calculate_joint_torques_RNEA(jacobDotSolver, ikSolverAcc, idSolver,
                                     jnt_velocity, jd_qd, xdd,
                                     xdd_minus_jd_qd, jnt_accelerations,
                                     jnt_positions, jnt_velocities,
                                     linkWrenches_EE, jnt_torques_cmd);
      }

      // thresholding the jnt_torques_cmd before sending to the robot
      for (int i = 0; i < kinova_constants::NUMBER_OF_JOINTS; i++)
      {
        if (jnt_torques_cmd(i) > 0.0)
        {
          jnt_torques_cmd(i) = std::min(JOINT_TORQUE_THRESHOLD, jnt_torques_cmd(i));
        }
        else
        {
          jnt_torques_cmd(i) = std::max(-JOINT_TORQUE_THRESHOLD, jnt_torques_cmd(i));
        }
      }

      kinova_arm_mediator.set_joint_torques(jnt_torques_cmd);
      apply_ee_force_x_axis_data = 0.0;
      apply_ee_force_y_axis_data = 0.0;
      apply_ee_force_z_axis_data = 0.0;
      apply_ee_torque_x_axis_data = 0.0;
      apply_ee_torque_y_axis_data = 0.0;
      apply_ee_torque_z_axis_data = 0.0;
    }
  }

  void MotionSpecificationActionServer::execute(const std::shared_ptr<GoalHandleMotionSpecification> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal");
    rclcpp::Rate loop_rate(1000);
    goal_accepted_and_executing = false;
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<MotionSpecification::Feedback>();
    auto &tcp_wrt_FrameName = feedback->tcp_position;
    tcp_wrt_FrameName = {0.0, 0.0, 0.0};
    auto result = std::make_shared<MotionSpecification::Result>();
    auto previous_time = std::chrono::high_resolution_clock::now();

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
    try
    {
      read_ms_conditions_count(motion_specification_params_object);
      read_frame_name(motion_specification_params_object);
    }
    catch (const YAML::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "Error reading motion specification parameters: %s", e.what());
      result->motion_successful = false;
      goal_handle->abort(result);
      return;
    } // if there is an error while reading the motion specification, abort the goal


    get_transform_BL_wrt_desired_frame(
        frame_name,
        BL_wrt_FrameName_frame,
        transform_stamped,
        transform_timeout_duration,
        transform_available);

    // Initialize the parameters
    reset_flags();
    goal_accepted_and_executing = true;

    // print string message on goal
    while (goal_accepted_and_executing && rclcpp::ok())
    {
      if (!communication_setup_done)
      {
        RCLCPP_INFO(this->get_logger(), "Setting up communication with robot");
        kinova_setup_communication(robot_to_control, kinova_arm_mediator, communication_setup_done);
      }

      if (!configuration_file_read) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Small delay
        RCLCPP_INFO(this->get_logger(), "Waiting for configuration file to be read...");
      }
      
      if (communication_setup_done && configuration_file_read)
      {
        if (flag == 0)
        {
          control_loop();
        }
        else
        {
          RCLCPP_ERROR(this->get_logger(), "Error in motion specification. Setting to position control.");
          kinova_arm_mediator.set_control_mode(control_mode::POSITION, nullptr);
          torque_control_mode_set = false;
          result->motion_successful = false;
          goal_handle->abort(result);
          return;
        }

        auto current_time = std::chrono::high_resolution_clock::now();
        auto time_since_last_publish = std::chrono::duration<double>(current_time-previous_time);
        if (time_since_last_publish.count() > state_publish_time_step)
        {
          publish_joint_states(jnt_positions);
          publish_ee_pose(measured_lin_pos_x_axis_data, measured_lin_pos_y_axis_data, measured_lin_pos_z_axis_data, measured_quat_FrameName, frame_name);

          tcp_wrt_FrameName = {measured_lin_pos_x_axis_data, measured_lin_pos_y_axis_data, measured_lin_pos_z_axis_data};
          goal_handle->publish_feedback(feedback);
          previous_time = current_time;
        };      
      };


      if (goal_handle->is_canceling())
      {
        RCLCPP_INFO(this->get_logger(), "Goal canceled. Setting to position control.");
        kinova_arm_mediator.set_control_mode(control_mode::POSITION, nullptr);
        torque_control_mode_set = false;
        result->motion_successful = false;
        goal_handle->canceled(result);
        RCLCPP_INFO(this->get_logger(), "Goal canceled");
        goal_accepted_and_executing = false;
        return;
      };
      if (post_condition_satisfied)
      {
        RCLCPP_INFO(this->get_logger(), "Post condition satisfied. Setting to position control.");
        kinova_arm_mediator.set_control_mode(control_mode::POSITION, nullptr);
        torque_control_mode_set = false;
        goal_accepted_and_executing = false;
        result->motion_successful = true;
        goal_handle->succeed(result);
        RCLCPP_INFO(this->get_logger(), "Goal succeeded");
        return;
      };
      loop_rate.sleep();
    }

    if (!rclcpp::ok())
    {
      RCLCPP_ERROR(this->get_logger(), "ROS node terminated. Setting to position control.");
      kinova_arm_mediator.set_control_mode(control_mode::POSITION, nullptr);
      result->motion_successful = false;
      goal_handle->canceled(result);
      RCLCPP_INFO(this->get_logger(), "Goal canceled as ros node is terminated");
      goal_accepted_and_executing = false;
      return;
    }
  }
} // namespace motion_specification_action

RCLCPP_COMPONENTS_REGISTER_NODE(motion_specification_action::MotionSpecificationActionServer)
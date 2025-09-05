#include "motion_specification_action/motion_specification_server.hpp"

namespace motion_specification_action
{
  MotionSpecificationActionServer::MotionSpecificationActionServer(const rclcpp::NodeOptions &options)
      : Node("ms_action_server", options),
        control_loop_active_(true),
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
        iteration_count(0),
        frequency_of_state_publish(10),
        gravitational_acceleration{0.0f, 0.0f, -9.81f},
        time_period_of_complete_controller_cycle_data(0.0),
        time_since_start_per_condition_seconds(0.0),
        ms_start_time_set(false),
        stiffness_pos_x_axis_data(0.0),
        stiffness_pos_y_axis_data(0.0),
        stiffness_pos_z_axis_data(0.0),
        integral_pos_x_axis_data(0.0),
        integral_pos_y_axis_data(0.0),
        integral_pos_z_axis_data(0.0),
        damping_pos_x_axis_data(0.0),
        damping_pos_y_axis_data(0.0),
        damping_pos_z_axis_data(0.0),
        previous_error_x_pos(0.0),
        previous_error_y_pos(0.0),
        previous_error_z_pos(0.0),
        previous_d_signal_x_pos(0.0),
        previous_d_signal_y_pos(0.0),
        previous_d_signal_z_pos(0.0),
        error_sum_pos_x_axis_data(0.0),
        error_sum_pos_y_axis_data(0.0),
        error_sum_pos_z_axis_data(0.0),
        integral_clamping_limit_pos(0.0),
        integral_decay_rate_pos(0.0),
        dead_zone_limit_pos(0.0),
        lp_filter_alpha_pos(0.0),
        measured_pos_x_axis_data(0.0),
        measured_pos_y_axis_data(0.0),
        measured_pos_z_axis_data(0.0),
        pos_sp_x_axis_data(0.0),
        pos_sp_y_axis_data(0.0),
        pos_sp_z_axis_data(0.0),
        p_signal_x_pos(0.0),
        i_signal_x_pos(0.0),
        d_signal_x_pos(0.0),
        p_signal_y_pos(0.0),
        i_signal_y_pos(0.0),
        d_signal_y_pos(0.0),
        p_signal_z_pos(0.0),
        i_signal_z_pos(0.0),
        d_signal_z_pos(0.0),
        log_pid_pos(false),
        stiffness_vel_x_axis_data(0.0),
        stiffness_vel_y_axis_data(0.0),
        stiffness_vel_z_axis_data(0.0),
        integral_vel_x_axis_data(0.0),
        integral_vel_y_axis_data(0.0),
        integral_vel_z_axis_data(0.0),
        damping_vel_x_axis_data(0.0),
        damping_vel_y_axis_data(0.0),
        damping_vel_z_axis_data(0.0),
        previous_error_x_vel(0.0),
        previous_error_y_vel(0.0),
        previous_error_z_vel(0.0),
        previous_d_signal_x_vel(0.0),
        previous_d_signal_y_vel(0.0),
        previous_d_signal_z_vel(0.0),
        error_sum_vel_x_axis_data(0.0),
        error_sum_vel_y_axis_data(0.0),
        error_sum_vel_z_axis_data(0.0),
        integral_clamping_limit_vel(0.0),
        integral_decay_rate_vel(0.0),
        dead_zone_limit_vel(0.0),
        lp_filter_alpha_vel(0.0),
        measured_vel_x_axis_data(0.0),
        measured_vel_y_axis_data(0.0),
        measured_vel_z_axis_data(0.0),
        filtered_measured_vel_x_axis_data(0.0),
        filtered_measured_vel_y_axis_data(0.0),
        filtered_measured_vel_z_axis_data(0.0),
        vel_sp_x_axis_data(0.0),
        vel_sp_y_axis_data(0.0),
        vel_sp_z_axis_data(0.0),
        p_signal_x_vel(0.0),
        i_signal_x_vel(0.0),
        d_signal_x_vel(0.0),
        p_signal_y_vel(0.0),
        i_signal_y_vel(0.0),
        d_signal_y_vel(0.0),
        p_signal_z_vel(0.0),
        i_signal_z_vel(0.0),
        d_signal_z_vel(0.0),
        log_pid_vel(false),
        desired_quat_desired_frame{0.0, 0.0, 0.0, 1.0},
        measured_quat_desired_frame{0.0, 0.0, 0.0, 1.0},
        stiffness_roll_axis_data(0.0),
        stiffness_pitch_axis_data(0.0),
        stiffness_yaw_axis_data(0.0),
        force_to_apply_x_axis(0.0),
        force_to_apply_y_axis(0.0),
        force_to_apply_z_axis(0.0),
        apply_ee_force_x_axis_data(0.0),
        apply_ee_force_y_axis_data(0.0),
        apply_ee_force_z_axis_data(0.0),
        apply_ee_torque_x_axis_data(0.0),
        apply_ee_torque_y_axis_data(0.0),
        apply_ee_torque_z_axis_data(0.0),
        control_dt(0.001),
        forearm_link_y_axis_angle_sp(0.0),
        deadband_forearm_y_axis_angle(0.0),
        torque_limit_forearm_link(0.0),
        apply_forearm_z_axis_torque(0.0),
        apply_forearm_y_axis_torque(0.0),
        apply_forearm_x_axis_torque(0.0),
        measured_roll_data(0.0),
        measured_pitch_data(0.0),
        measured_yaw_data(0.0),
        configuration_file_read(false),
        pre_condition_satisfied(false),
        post_condition_satisfied(false),
        jnt_impedance_setpoint_is_set(false),
        reach_pre_configuration_joint_angles(false),
        pre_configuration_joint_angles_reached(false),
        pre_condition_exists(false),
        post_condition_exists(false),
        abort_motion_execution(false),
        pre_configuration_joint_angles_tolerance_radians(0.1),
        pre_configuration_max_deviation_radians(0.0),
        pre_configuration_joint_angles_radians{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        pre_configuration_joint_angle_0_rad(0.0),
        pre_configuration_joint_angle_1_rad(0.0),
        pre_configuration_joint_angle_2_rad(0.0),
        pre_configuration_joint_angle_3_rad(0.0),
        pre_configuration_joint_angle_4_rad(0.0),
        pre_configuration_joint_angle_5_rad(0.0),
        pre_configuration_joint_angle_6_rad(0.0),
        state_publish_time_step(0.1),
        lp_filter_alpha_measured_vel(0.0),
        rne_output_jnt_torques_vector_to_set_control_mode(kinova_constants::NUMBER_OF_JOINTS, 0.0),
        arm_name("kinova_gen3_2_right"),
        arm_base_link_name("base_link"),
        robot_base_link_name("eddie_base_link"),
        transform_available(false),
        goal_handle_result_published(true),
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
    
    ss_pos_pid << package_share_directory 
      << "/log_files/pos_pid_controller_"
      << getTimestamp()
      << "_P" << STIFFNESS_GAIN_X_POS
      << "_I" << INTEGRAL_GAIN_X_POS
      << "_D" << DAMPING_GAIN_X_POS
      << ".csv";
    pos_pid_log_file_name = ss_pos_pid.str();
    if (log_pid_pos)
    {
      pos_pid_data_stream_log.open(pos_pid_log_file_name);
      if (!pos_pid_data_stream_log.is_open()) {
          RCLCPP_ERROR(this->get_logger(), "Failed to open log file: %s", pos_pid_log_file_name.c_str());
          throw std::runtime_error("Failed to open log file");
      }
      std::cout << "Opened log file successfully" << std::endl;
      pos_pid_data_stream_log << "e_pos_x,e_pos_y,e_pos_z,pos_sp_x_axis_data,pos_sp_y_axis_data,pos_sp_z_axis_data,measured_pos_x_axis_data,measured_pos_y_axis_data,measured_pos_z_axis_data,stiffness_pos_x_axis_data,stiffness_pos_y_axis_data,stiffness_pos_z_axis_data,integral_pos_x_axis_data,integral_pos_y_axis_data,integral_pos_z_axis_data,error_sum_pos_x_axis_data,error_sum_pos_y_axis_data,error_sum_pos_z_axis_data,damping_pos_x_axis_data,damping_pos_y_axis_data,damping_pos_z_axis_data,p_signal_x_pos,p_signal_y_pos,p_signal_z_pos,i_signal_x_pos,i_signal_y_pos,i_signal_z_pos,d_signal_x_pos,d_signal_y_pos,d_signal_z_pos,apply_ee_force_x_axis_data,apply_ee_force_y_axis_data,apply_ee_force_z_axis_data\n";
    }

    ss_vel_pid << package_share_directory 
      << "/log_files/vel_pid_controller_"
      << getTimestamp()
      << "_P" << STIFFNESS_GAIN_X_VEL
      << "_I" << INTEGRAL_GAIN_X_VEL
      << "_D" << DAMPING_GAIN_X_VEL
      << ".csv";
    vel_pid_log_file_name = ss_vel_pid.str();
    if (log_pid_vel)
    {
      vel_pid_data_stream_log.open(vel_pid_log_file_name);
      if (!vel_pid_data_stream_log.is_open()) {
          RCLCPP_ERROR(this->get_logger(), "Failed to open log file: %s", vel_pid_log_file_name.c_str());
          throw std::runtime_error("Failed to open log file");
      }
      std::cout << "Opened log file successfully" << std::endl;
      vel_pid_data_stream_log << "e_vel_x,e_vel_y,e_vel_z,vel_sp_x_axis_data,vel_sp_y_axis_data,vel_sp_z_axis_data,measured_vel_x_axis_data,measured_vel_y_axis_data,measured_vel_z_axis_data,stiffness_vel_x_axis_data,stiffness_vel_y_axis_data,stiffness_vel_z_axis_data,integral_vel_x_axis_data,integral_vel_y_axis_data,integral_vel_z_axis_data,error_sum_vel_x_axis_data,error_sum_vel_y_axis_data,error_sum_vel_z_axis_data,damping_vel_x_axis_data,damping_vel_y_axis_data,damping_vel_z_axis_data,p_signal_x_vel,p_signal_y_vel,p_signal_z_vel,i_signal_x_vel,i_signal_y_vel,i_signal_z_vel,d_signal_x_vel,d_signal_y_vel,d_signal_z_vel,apply_ee_force_x_axis_data,apply_ee_force_y_axis_data,apply_ee_force_z_axis_data\n";
    }

    sa.sa_handler = &MotionSpecificationActionServer::handle_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    if (sigaction(SIGINT, &sa, NULL) == -1) {
        perror("sigaction SIGINT");
    }
    if (sigaction(SIGTERM, &sa, NULL) == -1) {
        perror("sigaction SIGTERM");
    }

    // Read frame axes and position from the configuration file
    BL_x_axis_wrt_GF = KDL::Vector(BL_x_axis_wrt_GF_vector[0], BL_x_axis_wrt_GF_vector[1], BL_x_axis_wrt_GF_vector[2]);
    BL_y_axis_wrt_GF = KDL::Vector(BL_y_axis_wrt_GF_vector[0], BL_y_axis_wrt_GF_vector[1], BL_y_axis_wrt_GF_vector[2]);
    BL_z_axis_wrt_GF = KDL::Vector(BL_z_axis_wrt_GF_vector[0], BL_z_axis_wrt_GF_vector[1], BL_z_axis_wrt_GF_vector[2]);
    BL_position_wrt_GF = KDL::Vector(BL_position_wrt_GF_vector[0], BL_position_wrt_GF_vector[1], BL_position_wrt_GF_vector[2]);
    frame_name = robot_base_link_name;

    // Initialize the KDL frame
    BL_wrt_GF = KDL::Rotation(BL_x_axis_wrt_GF, BL_y_axis_wrt_GF, BL_z_axis_wrt_GF);
    BL_wrt_GF_frame = KDL::Frame(BL_wrt_GF, BL_position_wrt_GF); 
    BL_wrt_desired_frame = BL_wrt_GF_frame;

    linkWrenches = KDL::Wrenches(NUM_LINKS, KDL::Wrench::Zero());
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
        "motion_specification", // Action name
        handle_goal,
        handle_cancel,
        handle_accepted);

    joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("ee_pose", 10);

    joint_names_ = {"Actuator1", "Actuator2", "Actuator3", "Actuator4", "Actuator5", "Actuator6", "Actuator7"};

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    publish_static_transform_from_GF_to_BL(robot_base_link_name, arm_base_link_name, BL_wrt_GF_frame);

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
  
  volatile sig_atomic_t MotionSpecificationActionServer::flag = 0;

  void MotionSpecificationActionServer::handle_signal(int sig)
  {
    flag = 1;
    std::cout << "Received signal " << sig << ", shutting down..." << std::endl;
    rclcpp::shutdown();
  }

  void MotionSpecificationActionServer::publish_static_transform_from_GF_to_BL(const std::string &robot_base_link_name, const std::string &arm_base_link_name, KDL::Frame &BL_wrt_GF_frame)
  {
    geometry_msgs::msg::TransformStamped static_transform;
    std::array<double, 4> quat_BL_wrt_GF;
    BL_wrt_GF_frame.M.GetQuaternion(quat_BL_wrt_GF[0], quat_BL_wrt_GF[1], quat_BL_wrt_GF[2], quat_BL_wrt_GF[3]);

    static_transform.header.stamp = this->get_clock()->now();
    static_transform.header.frame_id = robot_base_link_name;
    static_transform.child_frame_id = arm_base_link_name;

    static_transform.transform.translation.x = BL_wrt_GF_frame.p.x();
    static_transform.transform.translation.y = BL_wrt_GF_frame.p.y();
    static_transform.transform.translation.z = BL_wrt_GF_frame.p.z();

    static_transform.transform.rotation.x = quat_BL_wrt_GF[0];
    static_transform.transform.rotation.y = quat_BL_wrt_GF[1];
    static_transform.transform.rotation.z = quat_BL_wrt_GF[2];
    static_transform.transform.rotation.w = quat_BL_wrt_GF[3];

    static_broadcaster_->sendTransform(static_transform);
  }

  void MotionSpecificationActionServer::publish_ee_pose(const double &measured_pos_x_axis_data, const double &measured_pos_y_axis_data, const double &measured_pos_z_axis_data, const std::array<double, 4> &measured_quat_desired_frame, const std::string &frame_name) {
    auto pose_msg = geometry_msgs::msg::PoseStamped();

    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = frame_name;

    // Example EE position (replace with real FK values)
    pose_msg.pose.position.x = measured_pos_x_axis_data;
    pose_msg.pose.position.y = measured_pos_y_axis_data;
    pose_msg.pose.position.z = measured_pos_z_axis_data;

    // Example orientation (identity quaternion)
    pose_msg.pose.orientation.x = measured_quat_desired_frame[0];
    pose_msg.pose.orientation.y = measured_quat_desired_frame[1];
    pose_msg.pose.orientation.z = measured_quat_desired_frame[2];
    pose_msg.pose.orientation.w = measured_quat_desired_frame[3];

    pose_publisher_->publish(pose_msg);
  }

  void MotionSpecificationActionServer::publish_joint_states(KDL::JntArray& jnt_positions) {
    auto message = sensor_msgs::msg::JointState();
    message.header.stamp = this->now();
    message.name = joint_names_;
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
    time_since_start_per_condition_seconds = 0.0;
    ms_start_time_set = false;
    switch_to_joint_impendance_control = false;
    jnt_impedance_setpoint_is_set = false;
    pre_condition_satisfied = false;
    post_condition_satisfied = false;
    pre_configuration_joint_angles_reached = false;
    previous_error_x_pos = 0.0;
    previous_d_signal_x_pos = 0.0;
    error_sum_pos_x_axis_data = 0.0;
    previous_error_y_pos = 0.0;
    previous_d_signal_y_pos = 0.0;
    error_sum_pos_y_axis_data = 0.0;
    previous_error_z_pos = 0.0;
    previous_d_signal_z_pos = 0.0;
    error_sum_pos_z_axis_data = 0.0;

    previous_error_x_vel = 0.0;
    previous_d_signal_x_vel = 0.0;
    error_sum_vel_x_axis_data = 0.0;
    previous_error_y_vel = 0.0;
    previous_d_signal_y_vel = 0.0;
    error_sum_vel_y_axis_data = 0.0;
    previous_error_z_vel = 0.0;
    previous_d_signal_z_vel = 0.0;
    error_sum_vel_z_axis_data = 0.0;
  }

  void MotionSpecificationActionServer::kinova_setup_communication(
      const robot_controlled &robot_to_control,
      kinova_mediator &kinova_arm_mediator)
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
    }
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
  
  void MotionSpecificationActionServer::get_ForeArm_Link_wrench(const KDL::JntArray &jnt_positions,
                                                                KDL::Frame &measured_ForeArm_Link_Pose_BL,
                                                                std::shared_ptr<KDL::ChainFkSolverPos_recursive> &fkSolverPos,
                                                                double &apply_forearm_x_axis_torque,
                                                                double &apply_forearm_y_axis_torque,
                                                                double &apply_forearm_z_axis_torque,
                                                                double &stiffness_forearm_y_axis_angle,
                                                                const double &forearm_link_y_axis_angle_sp,
                                                                const double &deadband_forearm_y_axis_angle,
                                                                const double &torque_limit_forearm_link)
  {
    KDL::Vector torque_vector = KDL::Vector::Zero();
    fkSolverPos->JntToCart(jnt_positions, measured_ForeArm_Link_Pose_BL, 3);

    // get y-axis of link3 (forearm) in arm base_link frame
    KDL::Vector measured_ForeArm_Link_y_axis_BL = measured_ForeArm_Link_Pose_BL.M.UnitY();

    // project y-axis onto yz-plane of arm base_link to get angle made with the plane
    KDL::Vector measured_ForeArm_Link_y_axis_BL_projection_to_YZ_plane = KDL::Vector(0.0, measured_ForeArm_Link_y_axis_BL.y(), measured_ForeArm_Link_y_axis_BL.z());
    measured_ForeArm_Link_y_axis_BL_projection_to_YZ_plane = measured_ForeArm_Link_y_axis_BL_projection_to_YZ_plane / measured_ForeArm_Link_y_axis_BL_projection_to_YZ_plane.Norm();
    double angle = std::acos(KDL::dot(measured_ForeArm_Link_y_axis_BL, measured_ForeArm_Link_y_axis_BL_projection_to_YZ_plane));

    // get torque axis as the cross product between y-axis and its projection vector
    KDL::Vector torque_axis = measured_ForeArm_Link_y_axis_BL * measured_ForeArm_Link_y_axis_BL_projection_to_YZ_plane;
    torque_axis = torque_axis / torque_axis.Norm();

    double error = 0;
    if (angle < (forearm_link_y_axis_angle_sp - deadband_forearm_y_axis_angle)) {
      error = (forearm_link_y_axis_angle_sp - deadband_forearm_y_axis_angle) - angle;
    } else if (angle > (forearm_link_y_axis_angle_sp + deadband_forearm_y_axis_angle)) {
      error = (forearm_link_y_axis_angle_sp + deadband_forearm_y_axis_angle) - angle;
    } else {
      error = 0;
    }
    double torque_magnitude = std::abs(stiffness_forearm_y_axis_angle * error);
    if (torque_magnitude > torque_limit_forearm_link) {
      torque_magnitude = torque_limit_forearm_link;
    } else if (torque_magnitude < -torque_limit_forearm_link) {
      torque_magnitude = -torque_limit_forearm_link;
    }

    // transform torque axis to forearm link frame
    KDL::Vector torque_axis_forearm_link = measured_ForeArm_Link_Pose_BL.M.Inverse() * torque_axis;
    torque_vector = torque_magnitude * torque_axis_forearm_link;

    apply_forearm_x_axis_torque = torque_vector.x();
    apply_forearm_y_axis_torque = torque_vector.y();
    apply_forearm_z_axis_torque = torque_vector.z();
  }

  void MotionSpecificationActionServer::get_end_effector_pose_and_twist(KDL::JntArrayVel &jnt_velocity,
                                                                        const KDL::JntArray &jnt_positions,
                                                                        const KDL::JntArray &jnt_velocities,
                                                                        KDL::Frame &measured_endEffPose_BL,
                                                                        KDL::FrameVel &measured_endEffTwist_BL,
                                                                        KDL::Frame &measured_endEffPose_desired_frame,
                                                                        KDL::FrameVel &measured_endEffTwist_desired_frame,
                                                                        std::shared_ptr<KDL::ChainFkSolverPos_recursive> &fkSolverPos,
                                                                        std::shared_ptr<KDL::ChainFkSolverVel_recursive> &fkSolverVel,
                                                                        const KDL::Frame &BL_wrt_desired_frame)
  {
    jnt_velocity.q = jnt_positions;
    jnt_velocity.qdot = jnt_velocities;

    fkSolverPos->JntToCart(jnt_positions, measured_endEffPose_BL);
    fkSolverVel->JntToCart(jnt_velocity, measured_endEffTwist_BL);

    measured_endEffPose_desired_frame = BL_wrt_desired_frame * measured_endEffPose_BL;
    measured_endEffTwist_desired_frame = BL_wrt_desired_frame * measured_endEffTwist_BL;
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
      KDL::Wrenches &linkWrenches,
      KDL::JntArray &jnt_torques)
  {
    jacobDotSolver->JntToJacDot(jnt_velocity, jd_qd);
    xdd_minus_jd_qd = xdd - jd_qd;
    ikSolverAcc->CartToJnt(jnt_positions, xdd_minus_jd_qd, jnt_accelerations);
    idSolver->CartToJnt(jnt_positions, jnt_velocities, jnt_accelerations, linkWrenches, jnt_torques);
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
        {"ORIENTATION_YAW", constraint_type::ORIENTATION_YAW},
        {"TIME_LIMIT", constraint_type::TIME_LIMIT}};
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
      const double &measured_pos_x_axis_data,
      const double &measured_pos_y_axis_data,
      const double &measured_pos_z_axis_data,
      const double &measured_roll_data,
      const double &measured_pitch_data,
      const double &measured_yaw_data,
      const double &measured_vel_x_axis_data,
      const double &measured_vel_y_axis_data,
      const double &measured_vel_z_axis_data,
      const double &time_since_start_per_condition_seconds,
      KDL::Wrench &linkWrench_EE,
      const int &condition_constraint_count,
      std::string &constraint_type_str,
      const std::string &arm_name,
      std::atomic<bool> &condition_satisfied,
      std::vector<int> &post_condition_indices,
      const YAML::Node &motion_specification_params_object,
      const condition_type &condition_type_value)
  {
    auto constraint_type_map = getConstraintTypeMap();
    bool constraint_satisfied;
    std::string condition_type_str;
    constraint_type constraint_type_;
    int number_of_disjunctions_post_condition = 0;
    std::vector<bool> disjunction_satisfaction_vector;

    if (condition_type_value == condition_type::PRE_CONDITION)
    {
      condition_type_str = "PRE_CONDITION";
    }
    else if (condition_type_value == condition_type::POST_CONDITION)
    {
      condition_type_str = "POST_CONDITION";
      number_of_disjunctions_post_condition = motion_specification_params_object[arm_name][condition_type_str]["number_of_disjunctions"].as<int>();
      disjunction_satisfaction_vector = std::vector<bool>(number_of_disjunctions_post_condition, true);
    }
    else
    {
      std::cout << "[check_pre_or_post_condition_satisfaction] Condition type not found" << std::endl;
      flag = 1; // stop the execution
      return;
    }

    // for every constraint in the pre or post condition
    if (condition_constraint_count > 0)
    {
      for (int i = 1; i < condition_constraint_count + 1; i++)
      {
        constraint_satisfied = true; // reset the constraint satisfaction for every constraint
        constraint_type_str = motion_specification_params_object[arm_name][condition_type_str]["constraints"][i]["type"].as<std::string>();

        auto constraint_iterator = constraint_type_map.find(constraint_type_str);

        if (constraint_iterator != constraint_type_map.end())
        {
          constraint_type_ = constraint_iterator->second; // selecting the second value stored in the iterator (the first value is the key)

          switch (constraint_type_)
          {
          case POSITION_XYZ:
            check_3D_vector_constraint_satisfaction(
                measured_pos_x_axis_data,
                measured_pos_y_axis_data,
                measured_pos_z_axis_data,
                constraint_satisfied,
                i,
                motion_specification_params_object,
                arm_name,
                condition_type_value);
            break;

          case VELOCITY_XYZ:
            if (condition_type_value == condition_type::PRE_CONDITION)
            {
              std::cout << "[check_pre_or_post_condition_satisfaction] Velocity constraint not implemented in pre-condition" << std::endl;
              flag = 1; // stop the execution
              break;
            }
            check_3D_vector_constraint_satisfaction(
                measured_vel_x_axis_data,
                measured_vel_y_axis_data,
                measured_vel_z_axis_data,
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

          case TIME_LIMIT: // in seconds
            if (condition_type_value == condition_type::PRE_CONDITION)
            {
              std::cout << "[check_pre_or_post_condition_satisfaction] Time limit constraint not implemented in pre-condition" << std::endl;
              flag = 1; // stop the execution
              break;
            }
            check_1D_vector_constraint_satisfaction(
                time_since_start_per_condition_seconds,
                constraint_satisfied,
                i,
                motion_specification_params_object,
                arm_name,
                condition_type_value);
            break;

          case FORCE_XYZ:
            if (condition_type_value == condition_type::PRE_CONDITION)
            {
              std::cout << "[check_pre_or_post_condition_satisfaction] Force constraint not implemented in pre-condition" << std::endl;
              flag = 1; // stop the execution
              break;
            }
            check_3D_vector_constraint_satisfaction(
                linkWrench_EE.force(0),
                linkWrench_EE.force(1),
                linkWrench_EE.force(2),
                constraint_satisfied,
                i,
                motion_specification_params_object,
                arm_name,
                condition_type_value);
            break;

          case TORQUE_RPY:
            if (condition_type_value == condition_type::PRE_CONDITION)
            {
              std::cout << "[check_pre_or_post_condition_satisfaction] Torque constraint not implemented in pre-condition" << std::endl;
              flag = 1; // stop the execution
              break;
            }
            check_3D_vector_constraint_satisfaction(
                linkWrench_EE.torque(0),
                linkWrench_EE.torque(1),
                linkWrench_EE.torque(2),
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

        if (condition_type_value == condition_type::PRE_CONDITION)
        {
          if (constraint_satisfied && 
              i == condition_constraint_count)
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
        if (condition_type_value == condition_type::POST_CONDITION)
        {
          if (!constraint_satisfied)
          {
            int disjunction_id = motion_specification_params_object[arm_name][condition_type_str]["constraints"][i]["disjunction_id"].as<int>();
            disjunction_satisfaction_vector[disjunction_id-1] = false;
          }
          if (i == condition_constraint_count)
          {
            // If any of the disjunctions is satisfied, the post condition is satisfied
            for (int j = 0; j < number_of_disjunctions_post_condition; j++)
            {
              if (disjunction_satisfaction_vector[j])
              {
                condition_satisfied = true;
                post_condition_indices.push_back(j+1);
              }
            }
            if (condition_satisfied)
            {
              return;
            }
            condition_satisfied = false;
            return;
          }
        }
      }
    }
  }

  void MotionSpecificationActionServer::get_setpoints_from_motion_specification(
    double &pos_sp_x_axis_data,
    double &pos_sp_y_axis_data,
    double &pos_sp_z_axis_data,
    double &vel_sp_x_axis_data,
    double &vel_sp_y_axis_data,
    double &vel_sp_z_axis_data,
    double &force_to_apply_x_axis,
    double &force_to_apply_y_axis,
    double &force_to_apply_z_axis,
    const int &per_condition_constraint_count,
    std::array<double, 4> &desired_quat_desired_frame,
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
                pos_sp_x_axis_data = constraint_value_list[k].as<double>();
              }
              else if (k == 1)
              {
                pos_sp_y_axis_data = constraint_value_list[k].as<double>();
              }
              else if (k == 2)
              {
                pos_sp_z_axis_data = constraint_value_list[k].as<double>();
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
                vel_sp_x_axis_data = constraint_value_list[k].as<double>();
              }
              else if (k == 1)
              {
                vel_sp_y_axis_data = constraint_value_list[k].as<double>();
              }
              else if (k == 2)
              {
                vel_sp_z_axis_data = constraint_value_list[k].as<double>();
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
              desired_quat_desired_frame[k] = constraint_value_list[k].as<double>();
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

  void MotionSpecificationActionServer::saturate_integral_error_sum(double *value, const double *integral_clamping_limit)
  {
    if (*value > *integral_clamping_limit)
    {
      *value = *integral_clamping_limit;
    }
    else if (*value < -*integral_clamping_limit)
    {
      *value = -*integral_clamping_limit;
    }
  }

  void MotionSpecificationActionServer::pid_controller(
      const double &stiffness_gain,
      const double &integral_gain,
      const double &damping_gain,
      double &previous_error,
      double &previous_d_signal,
      double &lp_filter_alpha,
      const double &control_dt,
      double &error_sum,
      const double &dead_zone_limit,
      const double &integral_decay_rate,
      const double &integral_clamping_limit,
      const double &measured_data,
      double &p_signal,
      double &i_signal,
      double &d_signal,
      const double &setpoint,
      double &pid_signal)
  {
    auto error = (setpoint - measured_data);
    if (std::abs(error) < dead_zone_limit)
    {
      error = 0.0;
    }
    p_signal = stiffness_gain * error;
    pid_signal += p_signal;
    error_sum += error * control_dt;
    if (previous_error == 0) {
      previous_error = error; // avoid large derivative kick at the start
    }
    d_signal = damping_gain * (error - previous_error) / control_dt;
    // add low-pass filter to avoid effect due to noise
    if (previous_d_signal == 0)
    {
      previous_d_signal = d_signal;
    }
    d_signal = lp_filter_alpha * d_signal + (1.0 - lp_filter_alpha) * previous_d_signal;
    pid_signal += d_signal;
    previous_d_signal = d_signal;

    previous_error = error;
    if ((error > 0 && error_sum < 0) || (error < 0 && error_sum > 0)) {
      error_sum = (1.0 - integral_decay_rate) * error_sum + integral_decay_rate * error; // faster integral windup when error sign changes
    }
    saturate_integral_error_sum(&error_sum, &integral_clamping_limit);
    i_signal = integral_gain * error_sum;
    pid_signal += i_signal;
  }

  void MotionSpecificationActionServer::get_force_and_torque_from_controller_described_in_desired_frame_to_apply_at_EE(
      const double &stiffness_pos_x_axis_data,
      const double &stiffness_pos_y_axis_data,
      const double &stiffness_pos_z_axis_data,
      const double &integral_pos_x_axis_data,
      const double &integral_pos_y_axis_data,
      const double &integral_pos_z_axis_data,
      const double &damping_pos_x_axis_data,
      const double &damping_pos_y_axis_data,
      const double &damping_pos_z_axis_data,
      double &previous_error_x_pos,
      double &previous_error_y_pos,
      double &previous_error_z_pos,
      double &previous_d_signal_x_pos,
      double &previous_d_signal_y_pos,
      double &previous_d_signal_z_pos,
      double &error_sum_pos_x_axis_data,
      double &error_sum_pos_y_axis_data,
      double &error_sum_pos_z_axis_data,
      const double &integral_clamping_limit_pos,
      const double &integral_decay_rate_pos,
      const double &dead_zone_limit_pos,
      double &lp_filter_alpha_pos,
      const double &measured_pos_x_axis_data,
      const double &measured_pos_y_axis_data,
      const double &measured_pos_z_axis_data,
      const double &pos_sp_x_axis_data,
      const double &pos_sp_y_axis_data,
      const double &pos_sp_z_axis_data,
      double &p_signal_x_pos,
      double &i_signal_x_pos,
      double &d_signal_x_pos,
      double &p_signal_y_pos,
      double &i_signal_y_pos,
      double &d_signal_y_pos,
      double &p_signal_z_pos,
      double &i_signal_z_pos,
      double &d_signal_z_pos,
      const bool &log_pid_pos,
      const double &stiffness_vel_x_axis_data,
      const double &stiffness_vel_y_axis_data,
      const double &stiffness_vel_z_axis_data,
      const double &integral_vel_x_axis_data,
      const double &integral_vel_y_axis_data,
      const double &integral_vel_z_axis_data,
      const double &damping_vel_x_axis_data,
      const double &damping_vel_y_axis_data,
      const double &damping_vel_z_axis_data,
      double &previous_error_x_vel,
      double &previous_error_y_vel,
      double &previous_error_z_vel,
      double &previous_d_signal_x_vel,
      double &previous_d_signal_y_vel,
      double &previous_d_signal_z_vel,
      double &error_sum_vel_x_axis_data,
      double &error_sum_vel_y_axis_data,
      double &error_sum_vel_z_axis_data,
      const double &integral_clamping_limit_vel,
      const double &integral_decay_rate_vel,
      const double &dead_zone_limit_vel,
      double &lp_filter_alpha_vel,
      const double &measured_vel_x_axis_data,
      const double &measured_vel_y_axis_data,
      const double &measured_vel_z_axis_data,
      double &filtered_measured_vel_x_axis_data,
      double &filtered_measured_vel_y_axis_data,
      double &filtered_measured_vel_z_axis_data,
      const double &lp_filter_alpha_measured_vel,
      const double &vel_sp_x_axis_data,
      const double &vel_sp_y_axis_data,
      const double &vel_sp_z_axis_data,
      double &p_signal_x_vel,
      double &i_signal_x_vel,
      double &d_signal_x_vel,
      double &p_signal_y_vel,
      double &i_signal_y_vel,
      double &d_signal_y_vel,
      double &p_signal_z_vel,
      double &i_signal_z_vel,
      double &d_signal_z_vel,
      const bool &log_pid_vel,
      const std::array<double, 4> &desired_quat_desired_frame,
      const double &stiffness_roll_axis_data,
      const double &stiffness_pitch_axis_data,
      const double &stiffness_yaw_axis_data,
      const double &force_to_apply_x_axis,
      const double &force_to_apply_y_axis,
      const double &force_to_apply_z_axis,
      double &apply_ee_force_x_axis_data,
      double &apply_ee_force_y_axis_data,
      double &apply_ee_force_z_axis_data,
      double &apply_ee_torque_x_axis_data,
      double &apply_ee_torque_y_axis_data,
      double &apply_ee_torque_z_axis_data,
      KDL::Frame &desired_endEffPose_desired_frame,
      const KDL::Frame &measured_endEffPose_desired_frame,
      const int &per_condition_constraint_count,
      KDL::Vector &angle_axis_diff_desired_frame,
      const double &control_dt,
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
                  pid_controller(
                    stiffness_pos_x_axis_data,
                    integral_pos_x_axis_data,
                    damping_pos_x_axis_data,
                    previous_error_x_pos,
                    previous_d_signal_x_pos,
                    lp_filter_alpha_pos,
                    control_dt,
                    error_sum_pos_x_axis_data,
                    dead_zone_limit_pos,
                    integral_decay_rate_pos,
                    integral_clamping_limit_pos,
                    measured_pos_x_axis_data,
                    p_signal_x_pos,
                    i_signal_x_pos,
                    d_signal_x_pos,
                    pos_sp_x_axis_data,
                    apply_ee_force_x_axis_data);
                }
                else if (k == 1)
                {
                  pid_controller(
                    stiffness_pos_y_axis_data,
                    integral_pos_y_axis_data,
                    damping_pos_y_axis_data,
                    previous_error_y_pos,
                    previous_d_signal_y_pos,
                    lp_filter_alpha_pos,
                    control_dt,
                    error_sum_pos_y_axis_data,
                    dead_zone_limit_pos,
                    integral_decay_rate_pos,
                    integral_clamping_limit_pos,
                    measured_pos_y_axis_data,
                    p_signal_y_pos,
                    i_signal_y_pos,
                    d_signal_y_pos,
                    pos_sp_y_axis_data,
                    apply_ee_force_y_axis_data);
                }
                else if (k == 2)
                {
                  pid_controller(
                    stiffness_pos_z_axis_data,
                    integral_pos_z_axis_data,
                    damping_pos_z_axis_data,
                    previous_error_z_pos,
                    previous_d_signal_z_pos,
                    lp_filter_alpha_pos,
                    control_dt,
                    error_sum_pos_z_axis_data,
                    dead_zone_limit_pos,
                    integral_decay_rate_pos,
                    integral_clamping_limit_pos,
                    measured_pos_z_axis_data,
                    p_signal_z_pos,
                    i_signal_z_pos,
                    d_signal_z_pos,
                    pos_sp_z_axis_data,
                    apply_ee_force_z_axis_data);
                }
              }
            }
            break;

          case VELOCITY_XYZ:
            if (filtered_measured_vel_x_axis_data == 0.0)
            {
              filtered_measured_vel_x_axis_data = measured_vel_x_axis_data;
            };
            if (filtered_measured_vel_y_axis_data == 0.0)
            {
              filtered_measured_vel_y_axis_data = measured_vel_y_axis_data;
            };
            if (filtered_measured_vel_z_axis_data == 0.0)
            {
              filtered_measured_vel_z_axis_data = measured_vel_z_axis_data;
            };
            std::cout << "lp_filter_alpha_measured_vel: " << lp_filter_alpha_measured_vel << std::endl;
            filtered_measured_vel_x_axis_data = lp_filter_alpha_measured_vel*measured_vel_x_axis_data + (1-lp_filter_alpha_measured_vel) * filtered_measured_vel_x_axis_data;
            filtered_measured_vel_y_axis_data = lp_filter_alpha_measured_vel*measured_vel_y_axis_data + (1-lp_filter_alpha_measured_vel) * filtered_measured_vel_y_axis_data;
            filtered_measured_vel_z_axis_data = lp_filter_alpha_measured_vel*measured_vel_z_axis_data + (1-lp_filter_alpha_measured_vel) * filtered_measured_vel_z_axis_data;
            for (int k = 0; k < 3; k++)
            {
              std::string constraint_str = constraint_value_list[k].as<std::string>("");
              if (!(constraint_str == "None"))
              {
                if (k == 0)
                {
                  pid_controller(
                    stiffness_vel_x_axis_data,
                    integral_vel_x_axis_data,
                    damping_vel_x_axis_data,
                    previous_error_x_vel,
                    previous_d_signal_x_vel,
                    lp_filter_alpha_vel,
                    control_dt,
                    error_sum_vel_x_axis_data,
                    dead_zone_limit_vel,
                    integral_decay_rate_vel,
                    integral_clamping_limit_vel,
                    filtered_measured_vel_x_axis_data,
                    p_signal_x_vel,
                    i_signal_x_vel,
                    d_signal_x_vel,
                    vel_sp_x_axis_data,
                    apply_ee_force_x_axis_data);                
                }
                else if (k == 1)
                {
                  pid_controller(
                    stiffness_vel_y_axis_data,
                    integral_vel_y_axis_data,
                    damping_vel_y_axis_data,
                    previous_error_y_vel,
                    previous_d_signal_y_vel,
                    lp_filter_alpha_vel,
                    control_dt,
                    error_sum_vel_y_axis_data,
                    dead_zone_limit_vel,
                    integral_decay_rate_vel,
                    integral_clamping_limit_vel,
                    filtered_measured_vel_y_axis_data,
                    p_signal_y_vel,
                    i_signal_y_vel,
                    d_signal_y_vel,
                    vel_sp_y_axis_data,
                    apply_ee_force_y_axis_data);
                }
                else if (k == 2)
                {
                  pid_controller(
                    stiffness_vel_z_axis_data,
                    integral_vel_z_axis_data,
                    damping_vel_z_axis_data,
                    previous_error_z_vel,
                    previous_d_signal_z_vel,
                    lp_filter_alpha_vel,
                    control_dt,
                    error_sum_vel_z_axis_data,
                    dead_zone_limit_vel,
                    integral_decay_rate_vel,
                    integral_clamping_limit_vel,
                    filtered_measured_vel_z_axis_data,
                    p_signal_z_vel,
                    i_signal_z_vel,
                    d_signal_z_vel,
                    vel_sp_z_axis_data,
                    apply_ee_force_z_axis_data);                
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
            desired_endEffPose_desired_frame.M = KDL::Rotation::Quaternion(desired_quat_desired_frame[0], desired_quat_desired_frame[1], desired_quat_desired_frame[2], desired_quat_desired_frame[3]);
            angle_axis_diff_desired_frame = KDL::diff(measured_endEffPose_desired_frame.M, desired_endEffPose_desired_frame.M);
            apply_ee_torque_x_axis_data = stiffness_roll_axis_data * angle_axis_diff_desired_frame(0);
            apply_ee_torque_y_axis_data = stiffness_pitch_axis_data * angle_axis_diff_desired_frame(1);
            apply_ee_torque_z_axis_data = stiffness_yaw_axis_data * angle_axis_diff_desired_frame(2);

            break;

          default:

            break;
          }
        }
        else
        {
          std::cout << "[get_force_and_torque_from_controller_described_in_desired_frame_to_apply_at_EE] Constraint type not found" << std::endl;
          flag = 1; // stop the execution
        }
      };
      auto e_pos_x = pos_sp_x_axis_data - measured_pos_x_axis_data;
      auto e_pos_y = pos_sp_y_axis_data - measured_pos_y_axis_data;
      auto e_pos_z = pos_sp_z_axis_data - measured_pos_z_axis_data;

      auto e_vel_x = vel_sp_x_axis_data - filtered_measured_vel_x_axis_data;
      auto e_vel_y = vel_sp_y_axis_data - filtered_measured_vel_y_axis_data;
      auto e_vel_z = vel_sp_z_axis_data - filtered_measured_vel_z_axis_data;
      if (log_pid_pos)
      {
        data_array_log_pos.push_back({e_pos_x,e_pos_y,e_pos_z,pos_sp_x_axis_data,pos_sp_y_axis_data,pos_sp_z_axis_data,measured_pos_x_axis_data,measured_pos_y_axis_data,measured_pos_z_axis_data,stiffness_pos_x_axis_data,stiffness_pos_y_axis_data,stiffness_pos_z_axis_data,integral_pos_x_axis_data,integral_pos_y_axis_data,integral_pos_z_axis_data,error_sum_pos_x_axis_data,error_sum_pos_y_axis_data,error_sum_pos_z_axis_data,damping_pos_x_axis_data,damping_pos_y_axis_data,damping_pos_z_axis_data,p_signal_x_pos,p_signal_y_pos,p_signal_z_pos,i_signal_x_pos,i_signal_y_pos,i_signal_z_pos,d_signal_x_pos,d_signal_y_pos,d_signal_z_pos,apply_ee_force_x_axis_data,apply_ee_force_y_axis_data,apply_ee_force_z_axis_data});
      };
      if (log_pid_vel)
      {
        data_array_log_vel.push_back({e_vel_x,e_vel_y,e_vel_z,vel_sp_x_axis_data,vel_sp_y_axis_data,vel_sp_z_axis_data,filtered_measured_vel_x_axis_data,filtered_measured_vel_y_axis_data,filtered_measured_vel_z_axis_data,stiffness_vel_x_axis_data,stiffness_vel_y_axis_data,stiffness_vel_z_axis_data,integral_vel_x_axis_data,integral_vel_y_axis_data,integral_vel_z_axis_data,error_sum_vel_x_axis_data,error_sum_vel_y_axis_data,error_sum_vel_z_axis_data,damping_vel_x_axis_data,damping_vel_y_axis_data,damping_vel_z_axis_data,p_signal_x_vel,p_signal_y_vel,p_signal_z_vel,i_signal_x_vel,i_signal_y_vel,i_signal_z_vel,d_signal_x_vel,d_signal_y_vel,d_signal_z_vel,apply_ee_force_x_axis_data,apply_ee_force_y_axis_data,apply_ee_force_z_axis_data});
      };
    }
  }

  void MotionSpecificationActionServer::read_ms_conditions_count(
    const YAML::Node &motion_specification_params_object,
    const std::string &arm_name,
    int &pre_condition_constraint_count,
    int &per_condition_constraint_count,
    int &post_condition_constraint_count)
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
      for (unsigned int i = 0; i < NUM_LINKS; i++)
      {
        RCLCPP_INFO(this->get_logger(), "Link %d: %s", i, chain_urdf.getSegment(i).getName().c_str());
      }
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "Error parsing URDF file: %s", e.what());
      return;
    }
  }

  void MotionSpecificationActionServer::get_pre_configuration_joint_angles(
      const std::string &arm_name,
      const YAML::Node &motion_specification_params_object,
      std::vector<double> &pre_configuration_joint_angles_radians,
      double &pre_configuration_joint_angles_tolerance_radians,
      bool &reach_pre_configuration_joint_angles,
      KDL::JntArray &pre_configuration_jnt_positions_kdl_array,
      kinova_mediator &kinova_arm_mediator)
  {
    try
    {
      const auto &arm_params = motion_specification_params_object[arm_name];

      reach_pre_configuration_joint_angles = arm_params["reach_pre_configuration_joint_angles"].as<bool>();

      if (!reach_pre_configuration_joint_angles)
      {
        RCLCPP_INFO(this->get_logger(), "Skipping reading pre-configuration joint angles as it is disabled");
        return;
      }
      pre_configuration_joint_angles_tolerance_radians = kinova_arm_mediator.DEG_TO_RAD(arm_params["pre_configuration_joint_angles_tolerance_deg"].as<double>());

      pre_configuration_max_deviation_radians = kinova_arm_mediator.DEG_TO_RAD(arm_params["pre_configuration_max_deviation_deg"].as<double>());


      pre_configuration_joint_angles_radians.clear();
      pre_configuration_joint_angles_radians.reserve(kinova_constants::NUMBER_OF_JOINTS);

      for (int i = 0; i < kinova_constants::NUMBER_OF_JOINTS; ++i)
      {
        std::string key = "pre_configuration_joint_angle_" + std::to_string(i) + "_deg";
        double angle_deg = arm_params[key].as<double>();
        pre_configuration_joint_angles_radians.push_back(kinova_arm_mediator.DEG_TO_RAD(angle_deg));
      }
    }
    catch (const YAML::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "Error reading pre-configuration joint angles: %s. Skipping reaching pre-configuration joint angles.", e.what());
      reach_pre_configuration_joint_angles = false;
      return;
    }
    if (pre_configuration_joint_angles_radians.size() != kinova_constants::NUMBER_OF_JOINTS)
    {
      RCLCPP_ERROR(this->get_logger(), "Pre-configuration joint angles size does not match the number of joints. Skipping reaching pre-configuration joint angles.");
      reach_pre_configuration_joint_angles = false;
      return;
    }
    
    pre_configuration_jnt_positions_kdl_array = KDL::JntArray(kinova_constants::NUMBER_OF_JOINTS);
    for (int i = 0; i < kinova_constants::NUMBER_OF_JOINTS; i++)
    {
      pre_configuration_jnt_positions_kdl_array(i) = pre_configuration_joint_angles_radians[i];
    }
    RCLCPP_INFO(this->get_logger(), "Read pre-configuration joint angles");
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

      STIFFNESS_GAIN_X_POS = config_file_object[arm_name]["STIFFNESS_GAIN_X_POS"].as<double>();
      STIFFNESS_GAIN_Y_POS = config_file_object[arm_name]["STIFFNESS_GAIN_Y_POS"].as<double>();
      STIFFNESS_GAIN_Z_POS = config_file_object[arm_name]["STIFFNESS_GAIN_Z_POS"].as<double>();
      DAMPING_GAIN_X_POS = config_file_object[arm_name]["DAMPING_GAIN_X_POS"].as<double>();
      DAMPING_GAIN_Y_POS = config_file_object[arm_name]["DAMPING_GAIN_Y_POS"].as<double>();
      DAMPING_GAIN_Z_POS = config_file_object[arm_name]["DAMPING_GAIN_Z_POS"].as<double>();
      INTEGRAL_GAIN_X_POS = config_file_object[arm_name]["INTEGRAL_GAIN_X_POS"].as<double>();
      INTEGRAL_GAIN_Y_POS = config_file_object[arm_name]["INTEGRAL_GAIN_Y_POS"].as<double>();
      INTEGRAL_GAIN_Z_POS = config_file_object[arm_name]["INTEGRAL_GAIN_Z_POS"].as<double>();
      INTEGRAL_DECAY_RATE_POS = config_file_object[arm_name]["INTEGRAL_DECAY_RATE_POS"].as<double>();
      INTEGRAL_CLAMPING_LIMIT_POS = config_file_object[arm_name]["INTEGRAL_CLAMPING_LIMIT_POS"].as<double>();
      DEADZONE_POS_CTRL = config_file_object[arm_name]["DEADZONE_POS_CTRL"].as<double>();
      LOW_PASS_FILTER_ALPHA_POS = config_file_object[arm_name]["LOW_PASS_FILTER_ALPHA_POS"].as<double>();
      LOG_PID_POS = config_file_object[arm_name]["LOG_PID_POS"].as<bool>();

      STIFFNESS_GAIN_X_VEL = config_file_object[arm_name]["STIFFNESS_GAIN_X_VEL"].as<double>();
      STIFFNESS_GAIN_Y_VEL = config_file_object[arm_name]["STIFFNESS_GAIN_Y_VEL"].as<double>();
      STIFFNESS_GAIN_Z_VEL = config_file_object[arm_name]["STIFFNESS_GAIN_Z_VEL"].as<double>();
      DAMPING_GAIN_X_VEL = config_file_object[arm_name]["DAMPING_GAIN_X_VEL"].as<double>();
      DAMPING_GAIN_Y_VEL = config_file_object[arm_name]["DAMPING_GAIN_Y_VEL"].as<double>();
      DAMPING_GAIN_Z_VEL = config_file_object[arm_name]["DAMPING_GAIN_Z_VEL"].as<double>();
      INTEGRAL_GAIN_X_VEL = config_file_object[arm_name]["INTEGRAL_GAIN_X_VEL"].as<double>();
      INTEGRAL_GAIN_Y_VEL = config_file_object[arm_name]["INTEGRAL_GAIN_Y_VEL"].as<double>();
      INTEGRAL_GAIN_Z_VEL = config_file_object[arm_name]["INTEGRAL_GAIN_Z_VEL"].as<double>();
      INTEGRAL_DECAY_RATE_VEL = config_file_object[arm_name]["INTEGRAL_DECAY_RATE_VEL"].as<double>();
      INTEGRAL_CLAMPING_LIMIT_VEL = config_file_object[arm_name]["INTEGRAL_CLAMPING_LIMIT_VEL"].as<double>();
      DEADZONE_VEL_CTRL = config_file_object[arm_name]["DEADZONE_VEL_CTRL"].as<double>();
      LOW_PASS_FILTER_ALPHA_VEL = config_file_object[arm_name]["LOW_PASS_FILTER_ALPHA_VEL"].as<double>();
      LOW_PASS_FILTER_ALPHA_MEASURED_VEL = config_file_object[arm_name]["LOW_PASS_FILTER_ALPHA_MEASURED_VEL"].as<double>();
      LOG_PID_VEL = config_file_object[arm_name]["LOG_PID_VEL"].as<bool>();

      STIFFNESS_GAIN_ROLL = config_file_object[arm_name]["STIFFNESS_GAIN_ROLL"].as<double>();
      STIFFNESS_GAIN_PITCH = config_file_object[arm_name]["STIFFNESS_GAIN_PITCH"].as<double>();
      STIFFNESS_GAIN_YAW = config_file_object[arm_name]["STIFFNESS_GAIN_YAW"].as<double>();
      STIFFNESS_GAIN_JOINT_IMPEDANCE_CTRL = config_file_object[arm_name]["STIFFNESS_GAIN_JOINT_IMPEDANCE_CTRL"].as<double>();
      STIFFNESS_GAIN_JOINT_IMPEDANCE_CTRL_PRE_JNT_CONFIG = config_file_object[arm_name]["STIFFNESS_GAIN_JOINT_IMPEDANCE_CTRL_PRE_JNT_CONFIG"].as<double>();

      DEADBAND_FOREARM_IN_DEG = config_file_object[arm_name]["DEADBAND_FOREARM_IN_DEG"].as<double>();
      FOREARM_Y_AXIS_DESIRED_ANGLE_TO_BL_X_AXIS_IN_DEG = config_file_object[arm_name]["FOREARM_Y_AXIS_DESIRED_ANGLE_TO_BL_X_AXIS_IN_DEG"].as<double>();
      STIFFNESS_FOREARM_JNT_LIMIT = config_file_object[arm_name]["STIFFNESS_FOREARM_JNT_LIMIT"].as<double>();
      TORQUE_MAGNITUDE_LIMIT_FOREARM_LINK = config_file_object[arm_name]["TORQUE_MAGNITUDE_LIMIT_FOREARM_LINK"].as<double>();

      gravitational_acceleration = config_file_object[arm_name]["gravitational_acceleration"].as<std::vector<float>>();
      WRENCH_THRESHOLD_LINEAR = config_file_object[arm_name]["WRENCH_THRESHOLD_LINEAR"].as<double>();
      WRENCH_THRESHOLD_ROTATIONAL = config_file_object[arm_name]["WRENCH_THRESHOLD_ROTATIONAL"].as<double>();

      JOINT_TORQUE_THRESHOLD_UNTIL_JNT_4 = config_file_object[arm_name]["JOINT_TORQUE_THRESHOLD_UNTIL_JNT_4"].as<double>();
      JOINT_TORQUE_THRESHOLD_FROM_JNT_5_TO_7 = config_file_object[arm_name]["JOINT_TORQUE_THRESHOLD_FROM_JNT_5_TO_7"].as<double>();

      JOINT_1_ANGLE_LIMIT_DEG = config_file_object[arm_name]["JOINT_1_ANGLE_LIMIT_DEG"].as<double>();
      JOINT_3_ANGLE_LIMIT_DEG = config_file_object[arm_name]["JOINT_3_ANGLE_LIMIT_DEG"].as<double>();
      JOINT_5_ANGLE_LIMIT_DEG = config_file_object[arm_name]["JOINT_5_ANGLE_LIMIT_DEG"].as<double>();

      DESIRED_TIME_STEP = config_file_object[arm_name]["DESIRED_TIME_STEP"].as<double>();
      SAVE_LOG_EVERY_NTH_STEP = config_file_object[arm_name]["SAVE_LOG_EVERY_NTH_STEP"].as<int>();
      control_dt = DESIRED_TIME_STEP;

      
      stiffness_pos_x_axis_data = STIFFNESS_GAIN_X_POS;
      stiffness_pos_y_axis_data = STIFFNESS_GAIN_Y_POS;
      stiffness_pos_z_axis_data = STIFFNESS_GAIN_Z_POS;
      
      damping_pos_x_axis_data = DAMPING_GAIN_X_POS;
      damping_pos_y_axis_data = DAMPING_GAIN_Y_POS;
      damping_pos_z_axis_data = DAMPING_GAIN_Z_POS;
      
      integral_pos_x_axis_data = INTEGRAL_GAIN_X_POS;
      integral_pos_y_axis_data = INTEGRAL_GAIN_Y_POS;
      integral_pos_z_axis_data = INTEGRAL_GAIN_Z_POS;
      
      integral_decay_rate_pos = INTEGRAL_DECAY_RATE_POS;
      integral_clamping_limit_pos = INTEGRAL_CLAMPING_LIMIT_POS;
      dead_zone_limit_pos = DEADZONE_POS_CTRL;
      lp_filter_alpha_pos = LOW_PASS_FILTER_ALPHA_POS;
      lp_filter_alpha_measured_vel = LOW_PASS_FILTER_ALPHA_MEASURED_VEL;
      log_pid_pos = LOG_PID_POS;
      
      stiffness_vel_x_axis_data = STIFFNESS_GAIN_X_VEL;
      stiffness_vel_y_axis_data = STIFFNESS_GAIN_Y_VEL;
      stiffness_vel_z_axis_data = STIFFNESS_GAIN_Z_VEL;

      damping_vel_x_axis_data = DAMPING_GAIN_X_VEL;
      damping_vel_y_axis_data = DAMPING_GAIN_Y_VEL;
      damping_vel_z_axis_data = DAMPING_GAIN_Z_VEL;

      integral_vel_x_axis_data = INTEGRAL_GAIN_X_VEL;
      integral_vel_y_axis_data = INTEGRAL_GAIN_Y_VEL;
      integral_vel_z_axis_data = INTEGRAL_GAIN_Z_VEL;
      integral_decay_rate_vel = INTEGRAL_DECAY_RATE_VEL;
      integral_clamping_limit_vel = INTEGRAL_CLAMPING_LIMIT_VEL;
      dead_zone_limit_vel = DEADZONE_VEL_CTRL;
      lp_filter_alpha_vel = LOW_PASS_FILTER_ALPHA_VEL;
      log_pid_vel = LOG_PID_VEL;

      stiffness_forearm_y_axis_angle = STIFFNESS_FOREARM_JNT_LIMIT;
      forearm_link_y_axis_angle_sp = FOREARM_Y_AXIS_DESIRED_ANGLE_TO_BL_X_AXIS_IN_DEG * M_PI / 180;
      deadband_forearm_y_axis_angle = DEADBAND_FOREARM_IN_DEG * M_PI / 180;
      torque_limit_forearm_link = TORQUE_MAGNITUDE_LIMIT_FOREARM_LINK;

      stiffness_roll_axis_data = STIFFNESS_GAIN_ROLL;
      stiffness_pitch_axis_data = STIFFNESS_GAIN_PITCH;
      stiffness_yaw_axis_data = STIFFNESS_GAIN_YAW;
      stiffness_joint_impedance_ctrl = STIFFNESS_GAIN_JOINT_IMPEDANCE_CTRL;
      stiffness_joint_impedance_ctrl_pre_jnt_config = STIFFNESS_GAIN_JOINT_IMPEDANCE_CTRL_PRE_JNT_CONFIG;

      BL_x_axis_wrt_GF_vector = config_file_object[arm_name]["BL_x_axis_wrt_GF"].as<std::vector<double>>();
      BL_y_axis_wrt_GF_vector = config_file_object[arm_name]["BL_y_axis_wrt_GF"].as<std::vector<double>>();
      BL_z_axis_wrt_GF_vector = config_file_object[arm_name]["BL_z_axis_wrt_GF"].as<std::vector<double>>();
      BL_position_wrt_GF_vector = config_file_object[arm_name]["BL_position_wrt_GF"].as<std::vector<double>>();
    }
    catch (const YAML::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "Error reading configuration parameters: %s", e.what());
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Configuration file read successfully");
    configuration_file_read = true;
  }

  void MotionSpecificationActionServer::control_loop()
  {
    rclcpp::Rate loop_rate(1/control_dt); // Control loop is running at 1kHz
    kinova_setup_communication(robot_to_control, kinova_arm_mediator);

    while (!configuration_file_read && rclcpp::ok()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto previous_state_publish_time = std::chrono::high_resolution_clock::now();

    if (rclcpp::ok() && control_loop_active_ && flag == 0)
    {
      kinova_feedback(kinova_arm_mediator, jnt_positions, jnt_velocities,
        jnt_torques_read);
        
      get_end_effector_pose_and_twist(
          jnt_velocity, jnt_positions, jnt_velocities,
          measured_endEffPose_BL, measured_endEffTwist_BL,
          measured_endEffPose_desired_frame, measured_endEffTwist_desired_frame,
          fkSolverPos, fkSolverVel, BL_wrt_desired_frame);

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
    }

    while (rclcpp::ok() && control_loop_active_ && flag == 0)
    {
      kinova_feedback(kinova_arm_mediator, jnt_positions, jnt_velocities,
        jnt_torques_read);
        
      get_end_effector_pose_and_twist(
          jnt_velocity, jnt_positions, jnt_velocities,
          measured_endEffPose_BL, measured_endEffTwist_BL,
          measured_endEffPose_desired_frame, measured_endEffTwist_desired_frame,
          fkSolverPos, fkSolverVel, BL_wrt_desired_frame);

      measured_pos_x_axis_data = measured_endEffPose_desired_frame.p.x();
      measured_vel_x_axis_data = measured_endEffTwist_desired_frame.GetTwist().vel.x();
      measured_pos_y_axis_data = measured_endEffPose_desired_frame.p.y();
      measured_vel_y_axis_data = measured_endEffTwist_desired_frame.GetTwist().vel.y();
      measured_pos_z_axis_data = measured_endEffPose_desired_frame.p.z();
      measured_vel_z_axis_data = measured_endEffTwist_desired_frame.GetTwist().vel.z();
      measured_endEffPose_desired_frame.M.GetQuaternion(measured_quat_desired_frame[0], measured_quat_desired_frame[1], measured_quat_desired_frame[2], measured_quat_desired_frame[3]);
      measured_endEffPose_desired_frame.M.GetRPY(measured_roll_data, measured_pitch_data, measured_yaw_data);

      // filtered_measured_vel_x_axis_data = vel_filter_x.filter(measured_vel_x_axis_data);
      // filtered_measured_vel_y_axis_data = vel_filter_y.filter(measured_vel_y_axis_data);
      // filtered_measured_vel_z_axis_data = vel_filter_z.filter(measured_vel_z_axis_data);

      auto current_time = std::chrono::high_resolution_clock::now();
      auto time_since_last_publish = std::chrono::duration<double>(current_time-previous_state_publish_time);
      if (time_since_last_publish.count() > state_publish_time_step)
      {
        publish_joint_states(jnt_positions);
        publish_ee_pose(measured_pos_x_axis_data, measured_pos_y_axis_data, measured_pos_z_axis_data, measured_quat_desired_frame, frame_name);
        previous_state_publish_time = current_time;
      };

      // Helper lambda to normalize angle differences to [-pi, pi]
      auto normalize_angle_diff = [](double diff) -> double {
          if (diff > M_PI) return diff - 2 * M_PI;
          if (diff < -M_PI) return diff + 2 * M_PI;
          return diff;
      };

      if (goal_accepted_and_executing)
      {
        switch_to_joint_impendance_control = false;
        jnt_impedance_setpoint_is_set = false;
      }

      if (goal_accepted_and_executing && reach_pre_configuration_joint_angles && !pre_configuration_joint_angles_reached)
      {
        for (int i = 0; i < kinova_constants::NUMBER_OF_JOINTS; ++i)
        {
          jnt_angle_diff = pre_configuration_joint_angles_radians[i] - jnt_positions(i);
          // Normalize angular difference for continuous revolute joints (0,2,4,6)
          if (i % 2 == 0)
          {
            jnt_angle_diff = normalize_angle_diff(jnt_angle_diff);
          }
          if (std::abs(jnt_angle_diff) > pre_configuration_max_deviation_radians && std::abs(jnt_velocities(i)) < 0.05)
          {
            std::cout << "[INFO] Pre-configuration check failed: joint " << i 
                      << " deviates by " << jnt_angle_diff << " rad." << std::endl;
            reach_pre_configuration_joint_angles = false;
            abort_motion_execution = true;
            goal_accepted_and_executing = false;
            break;
          }
        }

        if (reach_pre_configuration_joint_angles)
        {
          calculate_joint_torques_RNEA(jacobDotSolver, ikSolverAcc, idSolver,
                                      jnt_velocity, jd_qd, xdd,
                                      xdd_minus_jd_qd, jnt_accelerations,
                                      jnt_positions, jnt_velocities,
                                      linkWrenches_zero, torques_gravity_compensation);

          int jnt_angle_within_tolerance_cnt = 0;
          for (int i = 0; i < kinova_constants::NUMBER_OF_JOINTS; i++)
          {
            jnt_angle_diff = pre_configuration_joint_angles_radians[i] - jnt_positions(i);
            // Normalize angular difference for continuous revolute joints (0,2,4,6)
            if (i % 2 == 0)
            {
              jnt_angle_diff = normalize_angle_diff(jnt_angle_diff);
            }

            if (std::abs(jnt_angle_diff) < pre_configuration_joint_angles_tolerance_radians)
            {
              ++jnt_angle_within_tolerance_cnt;
            }
            jnt_torques_cmd(i) = stiffness_joint_impedance_ctrl_pre_jnt_config * jnt_angle_diff + torques_gravity_compensation(i);
          }
          if (jnt_angle_within_tolerance_cnt == kinova_constants::NUMBER_OF_JOINTS)
          {
            pre_configuration_joint_angles_reached = true;
            for (size_t i = 0; i < kinova_constants::NUMBER_OF_JOINTS; i++)
            {
              jnt_positions_setpoint(i) = pre_configuration_joint_angles_radians[i];
            }
            jnt_impedance_setpoint_is_set = true;
            goal_accepted_and_executing = false;
            std::cout << "Entering joint impedance mode as pre-config is reached and there are no further goals" << std::endl;
          }
        }
      }
      else if(goal_accepted_and_executing && !reach_pre_configuration_joint_angles)
      {
        // check if any motion specification satisfies pre condition
        if (pre_condition_exists && !pre_condition_satisfied)
        {
          check_pre_or_post_condition_satisfaction(
              measured_pos_x_axis_data,
              measured_pos_y_axis_data,
              measured_pos_z_axis_data,
              measured_roll_data,
              measured_pitch_data,
              measured_yaw_data,
              measured_vel_x_axis_data,
              measured_vel_y_axis_data,
              measured_vel_z_axis_data,
              time_since_start_per_condition_seconds,
              linkWrenches[kinova_constants::NUMBER_OF_JOINTS],
              pre_condition_constraint_count,
              constraint_type_str,
              arm_name,
              pre_condition_satisfied,
              post_condition_indices,
              motion_specification_params_object,
              condition_type::PRE_CONDITION);

          if (pre_condition_satisfied)
          {
            std::cout << "Pre condition satisfied. Now running controller to achieve per-condition until post-condition is satisfied." << std::endl;
          }
        }

        if (pre_condition_satisfied || !pre_condition_exists)
        {
          if (!ms_start_time_set)
          {
            ms_start_time = std::chrono::high_resolution_clock::now();
            ms_start_time_set = true;
          }
          auto ms_current_time = std::chrono::high_resolution_clock::now();
          time_since_start_per_condition_seconds = std::chrono::duration<double>(ms_current_time - ms_start_time).count();
          // check if the motion specification satisfies post condition
          if (post_condition_exists && !post_condition_satisfied)
          {
            check_pre_or_post_condition_satisfaction(
                measured_pos_x_axis_data,
                measured_pos_y_axis_data,
                measured_pos_z_axis_data,
                measured_roll_data,
                measured_pitch_data,
                measured_yaw_data,
                measured_vel_x_axis_data,
                measured_vel_y_axis_data,
                measured_vel_z_axis_data,
                time_since_start_per_condition_seconds,
                linkWrenches[kinova_constants::NUMBER_OF_JOINTS],
                post_condition_constraint_count,
                constraint_type_str,
                arm_name,
                post_condition_satisfied,
                post_condition_indices,
                motion_specification_params_object,
                condition_type::POST_CONDITION);
          }

          if (post_condition_satisfied)
          {
            if (!switch_to_joint_impendance_control)
            {
              RCLCPP_INFO(this->get_logger(), "Post condition satisfied. Switching to impedance control mode.");
              switch_to_joint_impendance_control = true;
              goal_accepted_and_executing = false;
            }
          }
          else
          {
            get_setpoints_from_motion_specification(
                pos_sp_x_axis_data,
                pos_sp_y_axis_data,
                pos_sp_z_axis_data,
                vel_sp_x_axis_data,
                vel_sp_y_axis_data,
                vel_sp_z_axis_data,
                force_to_apply_x_axis,
                force_to_apply_y_axis,
                force_to_apply_z_axis,
                per_condition_constraint_count,
                desired_quat_desired_frame,
                motion_specification_params_object,
                arm_name);

            get_force_and_torque_from_controller_described_in_desired_frame_to_apply_at_EE(
                stiffness_pos_x_axis_data,
                stiffness_pos_y_axis_data,
                stiffness_pos_z_axis_data,
                integral_pos_x_axis_data,
                integral_pos_y_axis_data,
                integral_pos_z_axis_data,
                damping_pos_x_axis_data,
                damping_pos_y_axis_data,
                damping_pos_z_axis_data,
                previous_error_x_pos,
                previous_error_y_pos,
                previous_error_z_pos,
                previous_d_signal_x_pos,
                previous_d_signal_y_pos,
                previous_d_signal_z_pos,
                error_sum_pos_x_axis_data,
                error_sum_pos_y_axis_data,
                error_sum_pos_z_axis_data,
                integral_clamping_limit_pos,
                integral_decay_rate_pos,
                dead_zone_limit_pos,
                lp_filter_alpha_pos,
                measured_pos_x_axis_data,
                measured_pos_y_axis_data,
                measured_pos_z_axis_data,
                pos_sp_x_axis_data,
                pos_sp_y_axis_data,
                pos_sp_z_axis_data,
                p_signal_x_pos,
                i_signal_x_pos,
                d_signal_x_pos,
                p_signal_y_pos,
                i_signal_y_pos,
                d_signal_y_pos,
                p_signal_z_pos,
                i_signal_z_pos,
                d_signal_z_pos,
                log_pid_pos,
                stiffness_vel_x_axis_data,
                stiffness_vel_y_axis_data,
                stiffness_vel_z_axis_data,
                integral_vel_x_axis_data,
                integral_vel_y_axis_data,
                integral_vel_z_axis_data,
                damping_vel_x_axis_data,
                damping_vel_y_axis_data,
                damping_vel_z_axis_data,
                previous_error_x_vel,
                previous_error_y_vel,
                previous_error_z_vel,
                previous_d_signal_x_vel,
                previous_d_signal_y_vel,
                previous_d_signal_z_vel,
                error_sum_vel_x_axis_data,
                error_sum_vel_y_axis_data,
                error_sum_vel_z_axis_data,
                integral_clamping_limit_vel,
                integral_decay_rate_vel,
                dead_zone_limit_vel,
                lp_filter_alpha_vel,
                measured_vel_x_axis_data,
                measured_vel_y_axis_data,
                measured_vel_z_axis_data,
                filtered_measured_vel_x_axis_data,
                filtered_measured_vel_y_axis_data,
                filtered_measured_vel_z_axis_data,
                lp_filter_alpha_measured_vel,
                vel_sp_x_axis_data,
                vel_sp_y_axis_data,
                vel_sp_z_axis_data,
                p_signal_x_vel,
                i_signal_x_vel,
                d_signal_x_vel,
                p_signal_y_vel,
                i_signal_y_vel,
                d_signal_y_vel,
                p_signal_z_vel,
                i_signal_z_vel,
                d_signal_z_vel,
                log_pid_vel,
                desired_quat_desired_frame,
                stiffness_roll_axis_data,
                stiffness_pitch_axis_data,
                stiffness_yaw_axis_data,
                force_to_apply_x_axis,
                force_to_apply_y_axis,
                force_to_apply_z_axis,
                apply_ee_force_x_axis_data,
                apply_ee_force_y_axis_data,
                apply_ee_force_z_axis_data,
                apply_ee_torque_x_axis_data,
                apply_ee_torque_y_axis_data,
                apply_ee_torque_z_axis_data,
                desired_endEffPose_desired_frame,
                measured_endEffPose_desired_frame,
                per_condition_constraint_count,
                angle_axis_diff_desired_frame,
                control_dt,
                motion_specification_params_object,
                arm_name
                );

            get_ForeArm_Link_wrench(jnt_positions, 
                                    measured_ForeArm_Link_Pose_BL, 
                                    fkSolverPos, 
                                    apply_forearm_x_axis_torque, 
                                    apply_forearm_y_axis_torque, 
                                    apply_forearm_z_axis_torque,
                                    stiffness_forearm_y_axis_angle,
                                    forearm_link_y_axis_angle_sp,
                                    deadband_forearm_y_axis_angle,
                                    torque_limit_forearm_link);
          }
        }
      }

      if (switch_to_joint_impendance_control || !goal_accepted_and_executing)
      {
        calculate_joint_torques_RNEA(jacobDotSolver, ikSolverAcc, idSolver,
                                    jnt_velocity, jd_qd, xdd,
                                    xdd_minus_jd_qd, jnt_accelerations,
                                    jnt_positions, jnt_velocities,
                                    linkWrenches_zero, torques_gravity_compensation);

        if (!jnt_impedance_setpoint_is_set)
        {
          RCLCPP_INFO(this->get_logger(), "Entering joint impedance mode");
          jnt_positions_setpoint = jnt_positions;
          jnt_impedance_setpoint_is_set = true;
        }
        else{
          for (int i = 0; i < kinova_constants::NUMBER_OF_JOINTS; i++)
          {
            jnt_angle_diff = jnt_positions_setpoint(i) - jnt_positions(i);
            // Normalize angular difference for continuous revolute joints (0,2,4,6)
            if (i % 2 == 0)
            {
              jnt_angle_diff = normalize_angle_diff(jnt_angle_diff);
            }
            jnt_torques_cmd(i) = stiffness_joint_impedance_ctrl * jnt_angle_diff + torques_gravity_compensation(i);
          }
        }
      }
      else if (!reach_pre_configuration_joint_angles)
      {
        // write the ee torques to linkWrenches
        linkWrenches[kinova_constants::NUMBER_OF_JOINTS].force(0) = -apply_ee_force_x_axis_data;
        linkWrenches[kinova_constants::NUMBER_OF_JOINTS].force(1) = -apply_ee_force_y_axis_data;
        linkWrenches[kinova_constants::NUMBER_OF_JOINTS].force(2) = -apply_ee_force_z_axis_data;
        linkWrenches[kinova_constants::NUMBER_OF_JOINTS].torque(0) = -apply_ee_torque_x_axis_data;
        linkWrenches[kinova_constants::NUMBER_OF_JOINTS].torque(1) = -apply_ee_torque_y_axis_data;
        linkWrenches[kinova_constants::NUMBER_OF_JOINTS].torque(2) = -apply_ee_torque_z_axis_data;

        // thresholding in cartesian space of the end effector
        for (int i = 0; i < 3; i++)
        {
          if (linkWrenches[kinova_constants::NUMBER_OF_JOINTS].force(i) > 0.0)
          {
            linkWrenches[kinova_constants::NUMBER_OF_JOINTS].force(i) = std::min(WRENCH_THRESHOLD_LINEAR, linkWrenches[kinova_constants::NUMBER_OF_JOINTS].force(i));
          }
          else
          {
            linkWrenches[kinova_constants::NUMBER_OF_JOINTS].force(i) = std::max(-WRENCH_THRESHOLD_LINEAR, linkWrenches[kinova_constants::NUMBER_OF_JOINTS].force(i));
          }

          if (linkWrenches[kinova_constants::NUMBER_OF_JOINTS].torque(i) > 0.0)
          {
            linkWrenches[kinova_constants::NUMBER_OF_JOINTS].torque(i) = std::min(WRENCH_THRESHOLD_ROTATIONAL, linkWrenches[kinova_constants::NUMBER_OF_JOINTS].torque(i));
          }
          else
          {
            linkWrenches[kinova_constants::NUMBER_OF_JOINTS].torque(i) = std::max(-WRENCH_THRESHOLD_ROTATIONAL, linkWrenches[kinova_constants::NUMBER_OF_JOINTS].torque(i));
          }
        };

        // LinkWrenches are set wrt BL frame. As RNE solver requires them in the frame of respective link, 
        // the wrenches are transformed from frame in which motion specification is described to the EE frame
        linkWrenches[NUM_LINKS - 1].force = measured_endEffPose_desired_frame.M.Inverse() * linkWrenches[NUM_LINKS - 1].force;
        linkWrenches[NUM_LINKS - 1].torque = measured_endEffPose_desired_frame.M.Inverse() * linkWrenches[NUM_LINKS - 1].torque;
        
        // The forearm torques are already determined in its link
        linkWrenches[2].torque(0) = -apply_forearm_x_axis_torque;
        linkWrenches[2].torque(1) = -apply_forearm_y_axis_torque;
        linkWrenches[2].torque(2) = -apply_forearm_z_axis_torque;

        calculate_joint_torques_RNEA(jacobDotSolver, ikSolverAcc, idSolver,
                                    jnt_velocity, jd_qd, xdd,
                                    xdd_minus_jd_qd, jnt_accelerations,
                                    jnt_positions, jnt_velocities,
                                    linkWrenches, jnt_torques_cmd);
      }

      // thresholding the jnt_torques_cmd before sending to the robot
      for (int i = 0; i < kinova_constants::NUMBER_OF_JOINTS; i++)
      {
        if (i < 4)
        {
          joint_torque_threshold = JOINT_TORQUE_THRESHOLD_UNTIL_JNT_4;
        }
        else
        {
          joint_torque_threshold = JOINT_TORQUE_THRESHOLD_FROM_JNT_5_TO_7;
        }
        if (std::abs(jnt_torques_cmd(i)) > joint_torque_threshold)
        {
          // TODO: remove this line
          // std::cout << "[WARNING] Joint (starting from 1): [" << i + 1 << "] torque command " << jnt_torques_cmd(i)
          //           << " Nm exceeds threshold of " << joint_torque_threshold
          //           << " Nm. Limiting torque command." << std::endl;
          continue;
        }
        if (jnt_torques_cmd(i) > 0.0)
        {
          jnt_torques_cmd(i) = std::min(joint_torque_threshold, jnt_torques_cmd(i));
        }
        else
        {
          jnt_torques_cmd(i) = std::max(-joint_torque_threshold, jnt_torques_cmd(i));
        }
      }

      // Ressist crossing joint angle limits
      auto limit_joint_torque = [&](int joint_index, double angle_limit_deg) {
        double angle_deg = kinova_arm_mediator.RAD_TO_DEG(jnt_positions(joint_index)); // no need to normalize, as it is done in kinova mediator for joints 1,3,5
        if (std::abs(angle_deg) > angle_limit_deg) {
            double direction = angle_deg > 0.0 ? 1.0 : -1.0;
            jnt_torques_cmd(joint_index) = 4.0 * (direction * angle_limit_deg - angle_deg);
            // TODO: remove this line
            // std::cout << "[WARNING] Joint " << joint_index + 1 << " angle " << angle_deg 
            //           << " deg exceeds limit of " << angle_limit_deg 
            //           << " deg. Applying resisting torque: " << jnt_torques_cmd(joint_index) << " Nm." << std::endl;
        }
      };

      // Apply torque limits to joints 1, 3, and 5
      limit_joint_torque(1, JOINT_1_ANGLE_LIMIT_DEG);
      limit_joint_torque(3, JOINT_3_ANGLE_LIMIT_DEG);
      limit_joint_torque(5, JOINT_5_ANGLE_LIMIT_DEG);

      kinova_arm_mediator.set_joint_torques(jnt_torques_cmd);
      apply_ee_force_x_axis_data = 0.0;
      apply_ee_force_y_axis_data = 0.0;
      apply_ee_force_z_axis_data = 0.0;
      apply_ee_torque_x_axis_data = 0.0;
      apply_ee_torque_y_axis_data = 0.0;
      apply_ee_torque_z_axis_data = 0.0;
      apply_forearm_x_axis_torque = 0.0;
      apply_forearm_y_axis_torque = 0.0;
      apply_forearm_z_axis_torque = 0.0;

      // Example: Check system state
      if (!control_loop_active_)
      {
        RCLCPP_INFO(this->get_logger(), "Control loop not active. Exiting.");
        break;
      }
      if (flag == 1)
      {
        RCLCPP_INFO(this->get_logger(), "Flag set to 1. Exiting control loop.");
        break;
      }
      loop_rate.sleep(); // Maintain the loop at 1kHz
      iteration_count++;

      if (log_pid_pos && iteration_count % SAVE_LOG_EVERY_NTH_STEP == 0)
      {
        appendDataToFile_dynamic_size(pos_pid_data_stream_log, data_array_log_pos);
        data_array_log_pos.clear();
      }
      if (log_pid_vel && iteration_count % SAVE_LOG_EVERY_NTH_STEP == 0)
      {
        appendDataToFile_dynamic_size(vel_pid_data_stream_log, data_array_log_vel);
        data_array_log_vel.clear();
      }
    }
    if (log_pid_pos)
    {
      close_log_files(data_array_log_pos, pos_pid_data_stream_log);
      std::cout << "Data collection completed. Log file name: " << pos_pid_log_file_name << "\n";
    }
    if (log_pid_vel)
    {
      close_log_files(data_array_log_vel, vel_pid_data_stream_log);
      std::cout << "Data collection completed. Log file name: " << vel_pid_log_file_name << "\n";
    }
    kinova_arm_mediator.set_control_mode(control_mode::POSITION, nullptr);
  }

  void MotionSpecificationActionServer::get_transform_BL_wrt_desired_frame(
      const std::string &frame_name,
      KDL::Frame &BL_wrt_desired_frame,
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
      if (tf_buffer_->canTransform(frame_name, arm_base_link_name, tf2::TimePointZero))
      {
        try {
          transform_stamped = tf_buffer_->lookupTransform(
            frame_name,
            arm_base_link_name,
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
        RCLCPP_INFO(this->get_logger(), "Waiting for transform from %s to base_link of arm ", frame_name.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Spin the node to process incoming messages
        rclcpp::spin_some(this->get_node_base_interface());
      }
    }
    BL_wrt_desired_frame = tf2::transformToKDL(transform_stamped);
  }

  template <size_t N>
  void MotionSpecificationActionServer::close_log_files(std::vector<std::array<double, N>>& data_array_log, 
                                                        std::ofstream &data_stream_log)
  {
    if (!data_array_log.empty())
    {
      appendDataToFile_dynamic_size(data_stream_log, data_array_log);
      data_array_log.clear();
    }
    data_stream_log.close();
  }

  void MotionSpecificationActionServer::execute(const std::shared_ptr<GoalHandleMotionSpecification> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal");
    rclcpp::Rate loop_rate(1);
    goal_accepted_and_executing = false;
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<MotionSpecification::Feedback>();
    auto &tcp_wrt_desired_frame = feedback->tcp_position;
    tcp_wrt_desired_frame = {0.0, 0.0, 0.0};
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
    try
    {
      read_ms_conditions_count(motion_specification_params_object,
                               arm_name,
                               pre_condition_constraint_count,
                               per_condition_constraint_count,
                               post_condition_constraint_count);
      read_frame_name(motion_specification_params_object);
      get_pre_configuration_joint_angles(
          arm_name,
          motion_specification_params_object,
          pre_configuration_joint_angles_radians,
          pre_configuration_joint_angles_tolerance_radians,
          reach_pre_configuration_joint_angles,
          pre_configuration_jnt_positions_kdl_array,
          kinova_arm_mediator);
    }
    catch (const YAML::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "Error reading motion specification parameters: %s", e.what());
      result->motion_successful = false;
      goal_handle->abort(result);
      return;
    } // if there is an error while reading the motion specification, abort the goal

    if (pre_condition_constraint_count == 0)
    {
      pre_condition_exists = false;
    }
    else{      
      pre_condition_exists = true;
    }
    if (post_condition_constraint_count == 0)
    {
      post_condition_exists = false;
    }
    else
    {
      post_condition_exists = true;
    }

    get_transform_BL_wrt_desired_frame(
        frame_name,
        BL_wrt_desired_frame,
        transform_stamped,
        transform_timeout_duration,
        transform_available);

    // Initialize the parameters
    reset_flags();
    goal_accepted_and_executing = true;
    goal_handle_result_published = false;

    while (!goal_handle_result_published && rclcpp::ok())
    {
      tcp_wrt_desired_frame = {measured_pos_x_axis_data, measured_pos_y_axis_data, measured_pos_z_axis_data};
      goal_handle->publish_feedback(feedback);

      if (goal_handle->is_canceling())
      {
        result->motion_successful = false;
        goal_handle->canceled(result);
        RCLCPP_INFO(this->get_logger(), "Goal canceled");
        goal_accepted_and_executing = false;
        goal_handle_result_published = true;
        return;
      }
      if (post_condition_satisfied)
      {
        goal_handle_result_published = true;
        goal_accepted_and_executing = false;
        result->motion_successful = true;
        result->post_condition_indices = post_condition_indices;
        post_condition_indices.clear();
        goal_handle->succeed(result);
        RCLCPP_INFO(this->get_logger(), "Goal succeeded");
        return;
      }
      if (pre_configuration_joint_angles_reached && reach_pre_configuration_joint_angles)
      {
        goal_handle_result_published = true;
        goal_accepted_and_executing = false;
        result->motion_successful = true;
        post_condition_indices.clear();
        goal_handle->succeed(result);
        RCLCPP_INFO(this->get_logger(), "Pre-configuration joint angles reached. Now waiting for any motion specification.");
        reach_pre_configuration_joint_angles = false;
      }
      if (abort_motion_execution)
      {
        result->motion_successful = false;
        post_condition_indices.clear();
        goal_handle->abort(result);
        RCLCPP_INFO(this->get_logger(), "Goal aborted due to abort signal.");
        goal_handle_result_published = true;
        return;
      }
      if (flag == 1)
      {
        result->motion_successful = false;
        post_condition_indices.clear();
        goal_handle->abort(result);
        RCLCPP_INFO(this->get_logger(), "Key interruption detected. Stopping server.");
        goal_handle_result_published = true;
        goal_accepted_and_executing = false;
        return;
      }
      loop_rate.sleep();
    }

    if (!rclcpp::ok())
    {
      result->motion_successful = false;
      post_condition_indices.clear();
      goal_handle->canceled(result);
      RCLCPP_INFO(this->get_logger(), "Goal canceled as ros node is terminated");
      goal_accepted_and_executing = false;
      return;
    }
  }
} // namespace motion_specification_action

RCLCPP_COMPONENTS_REGISTER_NODE(motion_specification_action::MotionSpecificationActionServer)
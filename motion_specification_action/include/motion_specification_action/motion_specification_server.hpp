#ifndef MOTION_SPECIFICATION_ACTION_SERVER_HPP
#define MOTION_SPECIFICATION_ACTION_SERVER_HPP

#include <functional>
#include <memory>
#include <thread>
#include <csignal>
#include <random>

#include <motion_specification_interfaces/action/motion_specification.hpp>
#include <rclcpp/rclcpp.hpp>
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2_ros/transform_listener.h>
#include "tf2_ros/static_transform_broadcaster.h"
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_kdl/tf2_kdl.hpp>

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

#include "motion_specification_action/visibility_control.h"
#include "control_blocks.h"
#include "monitors.h"

namespace motion_specification_action
{

  // enum for arms being controlled
  enum robot_controlled
  {
    KINOVA_GEN3_1_LEFT = 1,  // "192.168.1.10"
    KINOVA_GEN3_2_RIGHT = 2, // "192.168.1.12"
  };

  enum constraint_type
  {
    POSITION_XYZ = 1,
    FORCE_XYZ = 2,
    VELOCITY_XYZ = 3,
    TORQUE_RPY = 4,
    ORIENTATION_QUATERNION = 5,
    ORIENTATION_ROLL = 6,
    ORIENTATION_PITCH = 7,
    ORIENTATION_YAW = 8,
    TIME_LIMIT = 9,
  };

  enum operator_type
  {
    GREATER_THAN = 1,
    LESS_THAN = 2,
    EQUAL = 3,
  };

  enum condition_type
  {
    PRE_CONDITION = 1,
    PER_CONDITION = 2,
    POST_CONDITION = 3
  };

  class MotionSpecificationActionServer : public rclcpp::Node
  {
  public:
    using MotionSpecification = motion_specification_interfaces::action::MotionSpecification;
    using GoalHandleMotionSpecification = rclcpp_action::ServerGoalHandle<MotionSpecification>;

    MOTION_SPECIFICATION_ACTION_PUBLIC
    explicit MotionSpecificationActionServer(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
    ~MotionSpecificationActionServer();

  private:
    rclcpp_action::Server<MotionSpecification>::SharedPtr action_server_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
    std::vector<std::string> joint_names_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
    geometry_msgs::msg::TransformStamped transform_stamped;
    std::chrono::duration<double> transform_timeout_duration;
    std::chrono::high_resolution_clock::time_point ms_start_time;
    bool transform_available;

    std::thread control_loop_thread_;
    // Atomic flag to control loop execution, used to stop loop when destructor is called. 
    // atomic<bool> ensures safe access across threads.
    std::atomic<bool> control_loop_active_;
    volatile sig_atomic_t flag;                           // to break control loop
    std::atomic<bool> goal_accepted_and_executing;        // decide when to run while loop in execute block
    std::atomic<bool> switch_to_joint_impendance_control; // when control loop is running and no active ms is specified after the first one onwards
    bool configuration_file_read;
    bool jnt_impedance_setpoint_is_set;
    bool abort_motion_execution;
    std::atomic<bool> pre_condition_satisfied;
    std::atomic<bool> post_condition_satisfied;
    std::vector<int> post_condition_indices; // to store the indices of the post condition constraints that are satisfied

    // Control loop related: kinova communicatoin, KDL data structure handling
    struct sigaction sa;
    // initialise data by reading from the config file
    double WRENCH_THRESHOLD_LINEAR;
    double WRENCH_THRESHOLD_ROTATIONAL;
    double JOINT_TORQUE_THRESHOLD_UNTIL_JNT_3;
    double JOINT_TORQUE_THRESHOLD_FROM_JNT_4_TO_7;
    double STIFFNESS_GAIN_X;
    double STIFFNESS_GAIN_Y;
    double STIFFNESS_GAIN_Z;
    double STIFFNESS_GAIN_X_VELOCITY;
    double STIFFNESS_GAIN_Y_VELOCITY;
    double STIFFNESS_GAIN_Z_VELOCITY;
    double DEADBAND_FOREARM_IN_DEG;
    double FOREARM_Y_AXIS_DESIRED_ANGLE_TO_BL_X_AXIS_IN_DEG;
    double STIFFNESS_FOREARM_JNT_LIMIT;

    double INTEGRAL_GAIN_X;
    double INTEGRAL_GAIN_Y;
    double INTEGRAL_GAIN_Z;
    double INTEGRAL_CLAMPING_LIMIT;
    double STIFFNESS_GAIN_ROLL;
    double STIFFNESS_GAIN_PITCH;
    double STIFFNESS_GAIN_YAW;
    double STIFFNESS_GAIN_JOINT_IMPEDANCE_CTRL;
    double STIFFNESS_GAIN_JOINT_IMPEDANCE_CTRL_PRE_JNT_CONFIG;
    double DESIRED_TIME_STEP;
    double JOINT_1_ANGLE_LIMIT_DEG;
    double JOINT_3_ANGLE_LIMIT_DEG;
    double JOINT_5_ANGLE_LIMIT_DEG;
    double joint_torque_threshold;
    double pre_configuration_joint_angles_tolerance_radians;
    double pre_configuration_joint_angle_0_rad;
    double pre_configuration_joint_angle_1_rad;
    double pre_configuration_joint_angle_2_rad;
    double pre_configuration_joint_angle_3_rad;
    double pre_configuration_joint_angle_4_rad;
    double pre_configuration_joint_angle_5_rad;
    double pre_configuration_joint_angle_6_rad;
    std::vector<double> pre_configuration_joint_angles_radians;
    bool reach_pre_configuration_joint_angles;
    bool pre_configuration_joint_angles_reached;
    bool pre_condition_exists;
    bool post_condition_exists;
    bool ms_start_time_set;
    double pre_configuration_max_deviation_radians;
    std::string arm_name;

    std::string frame_name;
    std::string arm_base_link_name;
    std::string robot_base_link_name;
    int pre_condition_constraint_count;
    int per_condition_constraint_count;
    int post_condition_constraint_count;
    int frequency_of_state_publish;

    std::vector<float> gravitational_acceleration; // Example values

    // urdf and KDL
    KDL::Tree kinematic_tree;
    KDL::Chain chain_urdf;

    unsigned int NUM_LINKS;

    /* KDL solvers */
    std::shared_ptr<KDL::ChainJntToJacDotSolver> jacobDotSolver;
    std::shared_ptr<KDL::ChainFkSolverPos_recursive> fkSolverPos;
    std::shared_ptr<KDL::ChainFkSolverVel_recursive> fkSolverVel;
    std::shared_ptr<KDL::ChainIkSolverVel_pinv> ikSolverAcc;
    std::shared_ptr<KDL::ChainIdSolver_RNE> idSolver;

    // Declarations of the transformation-related variables
    KDL::Frame BL_wrt_FrameName_frame;
    std::vector<double> BL_x_axis_wrt_GF_vector;
    std::vector<double> BL_y_axis_wrt_GF_vector;
    std::vector<double> BL_z_axis_wrt_GF_vector;
    std::vector<double> BL_position_wrt_GF_vector;

    KDL::Vector BL_x_axis_wrt_GF;
    KDL::Vector BL_y_axis_wrt_GF;
    KDL::Vector BL_z_axis_wrt_GF;
    KDL::Vector BL_position_wrt_GF;

    KDL::Rotation BL_wrt_GF;
    KDL::Frame BL_wrt_GF_frame;

    // end effector Pose
    KDL::Frame measured_endEffPose_BL;
    KDL::Frame measured_endEffPose_FrameName;
    KDL::FrameVel measured_endEffTwist_BL;
    KDL::FrameVel measured_endEffTwist_FrameName;

    // Joint variables
    KDL::JntArray jnt_positions;
    KDL::JntArray pre_configuration_jnt_positions_kdl_array; // to set pre-configuration joint angles
    KDL::JntArray jnt_positions_setpoint;
    KDL::JntArray jnt_velocities; // has only joint velocities of all joints
    KDL::JntArray torques_gravity_compensation;
    KDL::JntArray jnt_torques_read; // to read from the robot

    KDL::JntArray jnt_torques_cmd; // to send to the robot
    KDL::JntArrayVel jnt_velocity; // has both joint position and joint velocity of all joints

    KDL::JntArray jnt_accelerations;
    KDL::JntArray zero_jnt_velocities;

    KDL::Wrenches linkWrenches_FrameName;
    KDL::Wrenches linkWrenches_EE;
    KDL::Wrenches linkWrenches_zero;

    // cartesian acceleration
    KDL::Twist xdd;
    KDL::Twist xdd_minus_jd_qd;
    KDL::Twist jd_qd;
    double time_since_start_per_condition_seconds;
    double state_publish_time_step;
    double time_period_of_complete_controller_cycle_data;
    double control_dt;
    double jnt_angle_diff;
    double stiffness_lin_x_axis_data;
    double stiffness_lin_y_axis_data;
    double stiffness_lin_z_axis_data;

    double stiffness_lin_vel_x_axis_data;
    double stiffness_lin_vel_y_axis_data;
    double stiffness_lin_vel_z_axis_data;

    double integral_lin_x_axis_data;
    double integral_lin_y_axis_data;
    double integral_lin_z_axis_data;
    double integral_clamping_limit;

    double error_sum_lin_x_axis_data;
    double error_sum_lin_y_axis_data;
    double error_sum_lin_z_axis_data;
    double error_sum_lin_axis_data;

    double stiffness_roll_axis_data;
    double stiffness_pitch_axis_data;
    double stiffness_yaw_axis_data;
    double stiffness_joint_impedance_ctrl;
    double stiffness_joint_impedance_ctrl_pre_jnt_config;

    double measured_lin_pos_x_axis_data;
    double measured_lin_pos_y_axis_data;
    double measured_lin_pos_z_axis_data;

    std::array<double, 4> measured_quat_FrameName;

    double measured_lin_vel_x_axis_data;
    double measured_lin_vel_y_axis_data;
    double measured_lin_vel_z_axis_data;

    double lin_pos_sp_x_axis_data;
    double lin_pos_sp_y_axis_data;
    double lin_pos_sp_z_axis_data;

    double lin_vel_sp_x_axis_data;
    double lin_vel_sp_y_axis_data;
    double lin_vel_sp_z_axis_data;

    double stiffness_term_lin_x_axis_data;
    double stiffness_term_lin_y_axis_data;
    double stiffness_term_lin_z_axis_data;

    double damping_term_x_axis_data;
    double damping_term_y_axis_data;
    double damping_term_z_axis_data;

    double lin_pos_error_stiffness_x_axis_data;
    double lin_pos_error_stiffness_y_axis_data;
    double lin_pos_error_stiffness_z_axis_data;

    double lin_vel_error_damping_x_axis_data;
    double lin_vel_error_damping_y_axis_data;
    double lin_vel_error_damping_z_axis_data;

    double stiffness_damping_terms_summation_x_axis_data;
    double stiffness_damping_terms_summation_y_axis_data;
    double stiffness_damping_terms_summation_z_axis_data;

    double apply_ee_force_x_axis_data;
    double apply_ee_force_y_axis_data;
    double apply_ee_force_z_axis_data;

    double apply_forearm_x_axis_torque;
    double apply_forearm_y_axis_torque;
    double apply_forearm_z_axis_torque;

    double apply_ee_torque_x_axis_data;
    double apply_ee_torque_y_axis_data;
    double apply_ee_torque_z_axis_data;

    double measured_roll_data;
    double measured_pitch_data;
    double measured_yaw_data;

    double force_to_apply_x_axis;
    double force_to_apply_y_axis;
    double force_to_apply_z_axis;

    KDL::Vector angle_axis_diff_FrameName;
    KDL::Frame desired_endEffPose_FrameName;
    std::array<double, 4> desired_quat_FrameName;

    // initialise multi-dimensional array to store data
    std::vector<std::vector<double>> data_array_log;
    int iterationCount;

    // joint torques that will be calculated before setting the control mode
    std::vector<double> rne_output_jnt_torques_vector_to_set_control_mode;

    kinova_mediator kinova_arm_mediator; // 192.168.1.10 (KINOVA_GEN3_1) // 192.168.1.12 (KINOVA_GEN3_2)

    // set robots to control
    robot_controlled robot_to_control;
    std::string config_file_path;
    std::string urdf_file_path;
    std::string package_share_directory;
    std::string constraint_type_str;

    YAML::Node config_file_object;
    YAML::Node motion_specification_params_object;

    void publish_joint_states(KDL::JntArray& jnt_positions);
    void publish_ee_pose(const double &measured_lin_pos_x_axis_data, const double &measured_lin_pos_y_axis_data, const double &measured_lin_pos_z_axis_data, const std::array<double, 4> &measured_quat_FrameName, const std::string &frame_name);
    void read_config_file(const YAML::Node &config_file_object);
    void parse_urdf_file(const std::string &urdf_file_path, KDL::Tree &kinematic_tree, KDL::Chain &chain_urdf, unsigned int &NUM_LINKS);
    void reset_flags();
    void initialise_solvers(
        std::shared_ptr<KDL::ChainJntToJacDotSolver> &jacobDotSolver,
        std::shared_ptr<KDL::ChainFkSolverPos_recursive> &fkSolverPos,
        std::shared_ptr<KDL::ChainFkSolverVel_recursive> &fkSolverVel,
        std::shared_ptr<KDL::ChainIkSolverVel_pinv> &ikSolverAcc,
        std::shared_ptr<KDL::ChainIdSolver_RNE> &idSolver,
        const std::vector<float> &gravitational_acceleration,
        const KDL::Chain &chain_urdf);

    void kinova_setup_communication(
        const robot_controlled &robot_to_control,
        kinova_mediator &kinova_arm_mediator);

    void read_ms_conditions_count(
      const YAML::Node &motion_specification_params_object,
      const std::string &arm_name,
      int &pre_condition_constraint_count,
      int &per_condition_constraint_count,
      int &post_condition_constraint_count);
    void read_frame_name(const YAML::Node &motion_specification_params_object);
    void publish_static_transform_from_GF_to_BL(
      const std::string &robot_base_link_name, 
      const std::string &arm_base_link_name, 
      KDL::Frame &BL_wrt_GF_frame);
    void get_transform_BL_wrt_desired_frame(
      const std::string &frame_name,
      KDL::Frame &BL_wrt_FrameName_frame,
      geometry_msgs::msg::TransformStamped &transform_stamped,
      std::chrono::duration<double> &transform_timeout_duration,
      bool &transform_available);
    void get_pre_configuration_joint_angles(
      const std::string &arm_name,
      const YAML::Node &motion_specification_params_object,
      std::vector<double> &pre_configuration_joint_angles_radians,
      double &pre_configuration_joint_angles_tolerance_radians,
      bool &reach_pre_configuration_joint_angles,
      KDL::JntArray &pre_configuration_jnt_positions_kdl_array,
      kinova_mediator &kinova_arm_mediator);
    void saturate_integral_error_sum(
      double* error_sum_lin_axis_data,
      const double* integral_clamping_limit
    );
    // void handle_signal(int sig);

    void kinova_feedback(kinova_mediator &kinova_arm_mediator,
                         KDL::JntArray &jnt_positions,
                         KDL::JntArray &jnt_velocities,
                         KDL::JntArray &jnt_torques);

    void get_ForeArm_Link_wrench(const KDL::JntArray &jnt_positions,
                                std::shared_ptr<KDL::ChainFkSolverPos_recursive> &fkSolverPos,
                                double apply_forearm_x_axis_torque,
                                double apply_forearm_y_axis_torque,
                                double apply_forearm_z_axis_torque);

    void get_end_effector_pose_and_twist(KDL::JntArrayVel &jnt_velocity,
                                         const KDL::JntArray &jnt_positions,
                                         const KDL::JntArray &jnt_velocities,
                                         KDL::Frame &measured_endEffPose_BL,
                                         KDL::FrameVel &measured_endEffTwist_BL,
                                         KDL::Frame &measured_endEffPose_FrameName,
                                         KDL::FrameVel &measured_endEffTwist_FrameName,
                                         std::shared_ptr<KDL::ChainFkSolverPos_recursive> &fkSolverPos,
                                         std::shared_ptr<KDL::ChainFkSolverVel_recursive> &fkSolverVel,
                                         const KDL::Frame &BL_wrt_FrameName_frame);

    void calculate_joint_torques_RNEA(
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
        KDL::JntArray &jnt_torques);

    template <size_t N>
    void appendDataToFile_dynamic_size(std::ofstream &file, const std::vector<std::array<double, N>> &data);

    std::string getTimestamp();

    void check_3D_vector_constraint_satisfaction(
        const double &measured_x_axis_data,
        const double &measured_y_axis_data,
        const double &measured_z_axis_data,
        bool &constraint_satisfied,
        const int &constraint_idx,
        const YAML::Node &motion_specification_params_object,
        const std::string &arm_name,
        const condition_type &condition_type_value);

    void check_1D_vector_constraint_satisfaction(
        const double &measured_data,
        bool &constraint_satisfied,
        const int &constraint_idx,
        const YAML::Node &motion_specification_params_object,
        const std::string &arm_name,
        const condition_type &condition_type_value);

    void check_pre_or_post_condition_satisfaction(
        const double &measured_lin_pos_x_axis_data,
        const double &measured_lin_pos_y_axis_data,
        const double &measured_lin_pos_z_axis_data,
        const double &measured_roll_data,
        const double &measured_pitch_data,
        const double &measured_yaw_data,
        const double &measured_lin_vel_x_axis_data,
        const double &measured_lin_vel_y_axis_data,
        const double &measured_lin_vel_z_axis_data,
        const double &time_since_start_per_condition_seconds,
        KDL::Wrench &linkWrenches_EE,
        const int &condition_constraint_count,
        std::string &constraint_type_str,
        const std::string &arm_name,
        std::atomic<bool> &condition_satisfied,
        std::vector<int> &post_condition_indices,
        const YAML::Node &motion_specification_params_object,
        const condition_type &condition_type_value);

    void get_setpoints_from_motion_specification(
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
      const std::string &arm_name);

    void get_force_and_torque_from_controller_described_in_FrameName_to_apply_at_EE(
        const double &stiffness_lin_x_axis_data,
        const double &stiffness_lin_y_axis_data,
        const double &stiffness_lin_z_axis_data,
        const double &stiffness_lin_vel_x_axis_data,
        const double &stiffness_lin_vel_y_axis_data,
        const double &stiffness_lin_vel_z_axis_data,
        const double &integral_lin_x_axis_data,
        const double &integral_lin_y_axis_data,
        const double &integral_lin_z_axis_data,
        double &error_sum_lin_x_axis_data,
        double &error_sum_lin_y_axis_data,
        double &error_sum_lin_z_axis_data,
        const double &integral_clamping_limit,
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
        KDL::Frame &desired_endEffPose_FrameName,
        const KDL::Frame &measured_endEffPose_FrameName,
        const int &per_condition_constraint_count,
        KDL::Vector &angle_axis_diff_FrameName,
        const YAML::Node &motion_specification_params_object,
        const std::string &arm_name);

    // Assuming constraint_type, operator_type, and condition_type are enums
    const std::unordered_map<std::string, constraint_type> &getConstraintTypeMap();
    const std::unordered_map<std::string, operator_type> &getOperatorTypeMap();
    const std::unordered_map<std::string, condition_type> &getConditionTypeMap();

    // Control loop related: end

    void control_loop();
    void execute(const std::shared_ptr<GoalHandleMotionSpecification> goal_handle);

    // Callbacks for the action server
    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID &uuid,
        std::shared_ptr<const MotionSpecification::Goal> goal);

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleMotionSpecification> goal_handle);

    void handle_accepted(
        const std::shared_ptr<GoalHandleMotionSpecification> goal_handle);
  };
} // namespace motion_specification_action

#endif // MOTION_SPECIFICATION_ACTION_SERVER_HPP
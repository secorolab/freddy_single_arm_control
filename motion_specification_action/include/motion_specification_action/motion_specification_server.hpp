#ifndef MOTION_SPECIFICATION_ACTION_SERVER_HPP
#define MOTION_SPECIFICATION_ACTION_SERVER_HPP

#include <functional>
#include <memory>
#include <thread>
#include <csignal>

#include <motion_specification_interfaces/action/motion_specification.hpp>
#include <rclcpp/rclcpp.hpp>
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2_ros/transform_listener.h>
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
    ORIENTATION_YAW = 8
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
    POST_CONDITION = 3,
    PREVAIL_CONDITION = 4
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
    geometry_msgs::msg::TransformStamped transform_stamped;

    std::thread control_loop_thread_;
    // Atomic flag to control loop execution, used to stop loop when destructor is called. 
    // atomic<bool> ensures safe access across threads.
    std::atomic<bool> control_loop_active_;
    volatile sig_atomic_t flag;                           // to break control loop
    std::atomic<bool> goal_accepted_and_executing;        // decide when to run while loop in execute block
    std::atomic<bool> motion_unsuccessful;                // flag set when prevail_condition is not met
    std::atomic<bool> switch_to_joint_impendance_control; // when control loop is running and no active ms is specified after the first one onwards
    bool configuration_file_read;
    bool jnt_impedance_setpoint_is_set;
    std::atomic<bool> pre_condition_satisfied;
    std::atomic<bool> post_condition_satisfied;
    std::atomic<bool> prevail_condition_satisfied;

    // Control loop related: kinova communicatoin, KDL data structure handling
    struct sigaction sa;
    // initialise data by reading from the config file
    double WRENCH_THRESHOLD_LINEAR;
    double WRENCH_THRESHOLD_ROTATIONAL;
    double JOINT_TORQUE_THRESHOLD;
    double STIFFNESS_GAIN_X;
    double STIFFNESS_GAIN_Y;
    double STIFFNESS_GAIN_Z;
    double DAMPING_GAIN_X;
    double DAMPING_GAIN_Y;
    double DAMPING_GAIN_Z;
    double STIFFNESS_GAIN_ROLL;
    double STIFFNESS_GAIN_PITCH;
    double STIFFNESS_GAIN_YAW;
    double STIFFNESS_GAIN_JOINT_IMPEDANCE_CTRL;
    // int MOTION_SPECIFICATION_READ;
    std::string arm_name;

    std::string frame_name;
    int pre_condition_constraint_count;
    int per_condition_constraint_count;
    int post_condition_constraint_count;
    int prevail_condition_constraint_count;
    // int motion_specification_read;
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

    // end effector Pose
    KDL::Frame measured_endEffPose_BL_arm;
    KDL::Frame measured_endEffPose_FrameName_arm;
    KDL::FrameVel measured_endEffTwist_BL_arm;
    KDL::FrameVel measured_endEffTwist_FrameName_arm;

    // Joint variables
    KDL::JntArray jnt_positions;
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
    double state_publish_time_step;
    double time_period_of_complete_controller_cycle_data;
    double jnt_angle_diff;
    double stiffness_lin_x_axis_data;
    double stiffness_lin_y_axis_data;
    double stiffness_lin_z_axis_data;

    double damping_lin_x_axis_data;
    double damping_lin_y_axis_data;
    double damping_lin_z_axis_data;

    double stiffness_roll_axis_data;
    double stiffness_pitch_axis_data;
    double stiffness_yaw_axis_data;
    double stiffness_joint_impedance_ctrl;

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

    double apply_ee_torque_x_axis_data;
    double apply_ee_torque_y_axis_data;
    double apply_ee_torque_z_axis_data;

    double measured_roll_data;
    double measured_pitch_data;
    double measured_yaw_data;

    double force_to_apply_x_axis;
    double force_to_apply_y_axis;
    double force_to_apply_z_axis;

    KDL::Vector angle_axis_diff_FrameName_arm;
    KDL::Frame desired_endEffPose_FrameName_arm;
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
    void lookup_transformation(const std::string &target_frame, const std::string &source_frame, geometry_msgs::msg::TransformStamped &transform);
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

    void read_ms_conditions_count(const YAML::Node &motion_specification_params_object);
    // void handle_signal(int sig);

    void kinova_feedback(kinova_mediator &kinova_arm_mediator,
                         KDL::JntArray &jnt_positions,
                         KDL::JntArray &jnt_velocities,
                         KDL::JntArray &jnt_torques);

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

    void check_pre_or_post_or_prevail_condition_satisfaction(
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
        std::atomic<bool> &condition_satisfied,
        const YAML::Node &motion_specification_params_object,
        const condition_type &condition_type_value);

    void get_setpoints_from_motion_specification(
        double &measured_lin_pos_x_axis_data,
        double &measured_lin_pos_y_axis_data,
        double &measured_lin_pos_z_axis_data,
        double &measured_lin_vel_x_axis_data,
        double &measured_lin_vel_y_axis_data,
        double &measured_lin_vel_z_axis_data,
        double &measured_roll_data,
        double &measured_pitch_data,
        double &measured_yaw_data,
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
        const KDL::Frame &measured_endEffPose_FrameName_arm,
        const int &per_condition_constraint_count,
        KDL::Vector &angle_axis_diff_FrameName_arm,
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
#include <Eigen/Core>
#include <chrono>
#include <controllers/control_blocks.h>
#include <functions/monitors.h>
#include <iostream>
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

/*
For pre-conditions and post-conditions, only position, lin velocity, orientation roll, pitch, yaw
For per-conditions only position, velocity, orientation quat (stiffness controller), force (feedforward force controller)
*/

// handling signals: referred from https://github.com/RoboticsCosmos/motion_spec_gen/blob/0b48adc779ae6d6d593c1d7bcdb87e5535482fc3/gen/freddy_uc1_ref_log.cpp
// this can be used later in the while loop to save log files before exiting the program
#include <csignal>

volatile sig_atomic_t flag = 0;

void handle_signal(int sig)
{
    flag = 1;
    std::cout << "Received signal: " << sig << std::endl;
}

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
};

enum condition_type
{
    PRE_CONDITION = 1,
    PER_CONDITION = 2,
    POST_CONDITION = 3,
    PREVAIL_CONDITION = 4
};

// get joint states from kinova mediator
void kinova_feedback(kinova_mediator &kinova_arm_mediator,
                     KDL::JntArray &jnt_positions,
                     KDL::JntArray &jnt_velocities,
                     KDL::JntArray &jnt_torques)
{
    kinova_arm_mediator.get_joint_state(jnt_positions,
                                        jnt_velocities,
                                        jnt_torques);
}

// get end effector pose and twist
void get_end_effector_pose_and_twist(KDL::JntArrayVel &jnt_velocity,
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

// calculate joint torques for given EE force and acceleration (Inverse Dynamics) using RNEA
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
    KDL::JntArray &jnt_torques)
{
    jacobDotSolver->JntToJacDot(jnt_velocity, jd_qd);
    xdd_minus_jd_qd = xdd - jd_qd;
    ikSolverAcc->CartToJnt(jnt_positions, xdd_minus_jd_qd, jnt_accelerations);
    idSolver->CartToJnt(jnt_positions, jnt_velocities, jnt_accelerations, linkWrenches_EE, jnt_torques);
}

// Function to append data to the file with dynamic size of vector
template <size_t N>
void appendDataToFile_dynamic_size(std::ofstream &file, const std::vector<std::array<double, N>> &data)
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

std::unordered_map<std::string, constraint_type> constraint_type_map = {
    {"POSITION_XYZ", constraint_type::POSITION_XYZ},
    {"FORCE_XYZ", constraint_type::FORCE_XYZ},
    {"VELOCITY_XYZ", constraint_type::VELOCITY_XYZ},
    {"TORQUE_RPY", constraint_type::TORQUE_RPY},
    {"ORIENTATION_QUATERNION", constraint_type::ORIENTATION_QUATERNION},
    {"ORIENTATION_ROLL", constraint_type::ORIENTATION_ROLL},
    {"ORIENTATION_PITCH", constraint_type::ORIENTATION_PITCH},
    {"ORIENTATION_YAW", constraint_type::ORIENTATION_YAW}};

std::unordered_map<std::string, operator_type> operator_type_map = {
    {"GREATER_THAN", operator_type::GREATER_THAN},
    {"LESS_THAN", operator_type::LESS_THAN}};

std::unordered_map<std::string, condition_type> condition_type_map = {
    {"PRE_CONDITION", condition_type::PRE_CONDITION},
    {"PER_CONDITION", condition_type::PER_CONDITION},
    {"POST_CONDITION", condition_type::POST_CONDITION},
    {"PREVAIL_CONDITION", condition_type::PREVAIL_CONDITION}};
// measured_lin_pos_x_axis_data, measured_lin_pos_y_axis_data, measured_lin_pos_z_axis_data, constraint_satisfied, i, motion_specification_params);

void check_3D_vector_constraint_satisfaction(
    const double &measured_x_axis_data,
    const double &measured_y_axis_data,
    const double &measured_z_axis_data,
    bool &constraint_satisfied,
    const int &constraint_idx,
    const YAML::Node &motion_specification_params,
    const std::string &arm_name,
    const condition_type &condition_type_value)
{
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
        operator_type_str = motion_specification_params[arm_name][condition_type_str]["constraints"][constraint_idx]["operator"][j].as<std::string>();
        if (operator_type_str != "null")
        {
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

void check_1D_vector_constraint_satisfaction(
    const double &measured_data,
    bool &constraint_satisfied,
    const int &constraint_idx,
    const YAML::Node &motion_specification_params,
    const std::string &arm_name,
    const condition_type &condition_type_value)
{
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

    operator_type_str = motion_specification_params[arm_name][condition_type_str]["constraints"][constraint_idx]["operator"].as<std::string>();

    if (operator_type_str != "null")
    {
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
    else{
        std::cout << "[check_1D_vector_constraint_satisfaction] Null operator in 1D constraint check is not meaningful" << std::endl;
    }
}

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
    bool &condition_satisfied,
    const YAML::Node &motion_specification_params,
    const condition_type &condition_type_value)
{
    bool constraint_satisfied = true;
    std::string condition_type_str;
    constraint_type constraint_type_;

    // for every constraint in the pre-condition
    if (condition_constraint_count > 0)
    { 
        for (int i = 1; i < condition_constraint_count+1; i++)
        {
            if (condition_type_value
                == condition_type::PRE_CONDITION)
                {
                    condition_type_str = "PRE_CONDITION";
                }
            else if (condition_type_value
                == condition_type::POST_CONDITION)
                {
                    condition_type_str = "POST_CONDITION";
                }
            else if (condition_type_value
                == condition_type::PREVAIL_CONDITION)
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
    double desired_quat_GF[4],
    const YAML::Node &motion_specification_params,
    const std::string &arm_name)
{
    std::string condition_type_str = "PER_CONDITION";
    YAML::Node constraint_value_list;

    if (per_condition_constraint_count > 0)
    {
        for (int i = 1; i < per_condition_constraint_count+1; i++)
        {
            std::string constraint_type_str = motion_specification_params[arm_name][condition_type_str]["constraints"][i]["type"].as<std::string>();
            auto constraint_iterator = constraint_type_map.find(constraint_type_str);

            if (constraint_iterator != constraint_type_map.end())
            {
                constraint_value_list = motion_specification_params[arm_name][condition_type_str]["constraints"][i]["value"];
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

void get_force_and_torque_from_controller_described_in_GF_to_apply_at_EE(
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
    const double desired_quat_GF[4],
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
    YAML::Node constraint_value_list;
    std::string condition_type_str = "PER_CONDITION";

    if (per_condition_constraint_count > 0)
    {
        for (int i = 1; i < per_condition_constraint_count+1; i++)
        {
            std::string constraint_type_str = motion_specification_params[arm_name][condition_type_str]["constraints"][i]["type"].as<std::string>();
            auto constraint_iterator = constraint_type_map.find(constraint_type_str);

            if (constraint_iterator != constraint_type_map.end())
            {
                constraint_type constraint_type_ = constraint_iterator->second;
                constraint_value_list = motion_specification_params[arm_name][condition_type_str]["constraints"][i]["value"];

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

int main()
{
    // handling signals
    struct sigaction sa;
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    for (int i = 1; i < NSIG; ++i)
    {
        if (sigaction(i, &sa, NULL) == -1)
        {
            perror("sigaction");
        }
    };

    // initialise data by reading from the config file
    double TIMEOUT_DURATION_TASK; // in seconds
    double WRENCH_THRESHOLD_LINEAR;
    double WRENCH_THRESHOLD_ROTATIONAL;
    double JOINT_TORQUE_THRESHOLD;
    double DESIRED_TIME_STEP;
    double STIFFNESS_GAIN_X;
    double STIFFNESS_GAIN_Y;
    double STIFFNESS_GAIN_Z;
    double DAMPING_GAIN_X;
    double DAMPING_GAIN_Y;
    double DAMPING_GAIN_Z;
    double STIFFNESS_GAIN_ROLL;
    double STIFFNESS_GAIN_PITCH;
    double STIFFNESS_GAIN_YAW;
    int SAVE_LOG_EVERY_NTH_STEP;
    std::string arm_name;

    int pre_condition_constraint_count = 0;
    int per_condition_constraint_count = 0;
    int post_condition_constraint_count = 0;
    int prevail_condition_constraint_count = 0;

    // set robots to control
    robot_controlled robots_to_control = robot_controlled::KINOVA_GEN3_2_RIGHT;

    YAML::Node config_file = YAML::LoadFile("kinova_arm_control/config/parameters.yaml");
    YAML::Node impedance_gains = YAML::LoadFile("kinova_arm_control/config/impedance_gains.yaml");                  // in global frame
    YAML::Node motion_specification_params = YAML::LoadFile("kinova_arm_control/config/motion_specification.yaml"); // in global frame

    arm_name = motion_specification_params["arm_name"].as<std::string>();

    STIFFNESS_GAIN_X = impedance_gains[arm_name]["STIFFNESS_GAIN_X"].as<double>();
    STIFFNESS_GAIN_Y = impedance_gains[arm_name]["STIFFNESS_GAIN_Y"].as<double>();
    STIFFNESS_GAIN_Z = impedance_gains[arm_name]["STIFFNESS_GAIN_Z"].as<double>();

    DAMPING_GAIN_X = impedance_gains[arm_name]["DAMPING_GAIN_X"].as<double>();
    DAMPING_GAIN_Y = impedance_gains[arm_name]["DAMPING_GAIN_Y"].as<double>();
    DAMPING_GAIN_Z = impedance_gains[arm_name]["DAMPING_GAIN_Z"].as<double>();

    STIFFNESS_GAIN_ROLL = impedance_gains[arm_name]["STIFFNESS_GAIN_ROLL"].as<double>();
    STIFFNESS_GAIN_PITCH = impedance_gains[arm_name]["STIFFNESS_GAIN_PITCH"].as<double>();
    STIFFNESS_GAIN_YAW = impedance_gains[arm_name]["STIFFNESS_GAIN_YAW"].as<double>();

    pre_condition_constraint_count = motion_specification_params[arm_name]["PRE_CONDITION"]["constraint_count"].as<int>();
    per_condition_constraint_count = motion_specification_params[arm_name]["PER_CONDITION"]["constraint_count"].as<int>();
    post_condition_constraint_count = motion_specification_params[arm_name]["POST_CONDITION"]["constraint_count"].as<int>();
    prevail_condition_constraint_count = motion_specification_params[arm_name]["PREVAIL_CONDITION"]["constraint_count"].as<int>();

    std::vector<float> gravitational_acceleration = config_file[arm_name]["gravitational_acceleration"].as<std::vector<float>>();
    TIMEOUT_DURATION_TASK = config_file[arm_name]["TIMEOUT_DURATION_TASK"].as<double>(); // seconds
    WRENCH_THRESHOLD_LINEAR = config_file[arm_name]["WRENCH_THRESHOLD_LINEAR"].as<double>();
    WRENCH_THRESHOLD_ROTATIONAL = config_file[arm_name]["WRENCH_THRESHOLD_ROTATIONAL"].as<double>();
    JOINT_TORQUE_THRESHOLD = config_file[arm_name]["JOINT_TORQUE_THRESHOLD"].as<double>();
    DESIRED_TIME_STEP = config_file[arm_name]["DESIRED_TIME_STEP"].as<double>();
    SAVE_LOG_EVERY_NTH_STEP = config_file[arm_name]["SAVE_LOG_EVERY_NTH_STEP"].as<int>();


    // urdf and KDL
    KDL::Tree kinematic_tree;
    KDL::Chain chain_urdf;

    kdl_parser::treeFromFile("urdf/Kinova_1.urdf", kinematic_tree);
    kinematic_tree.getChain("base_link", "EndEffector_Link", chain_urdf);
    const unsigned int NUM_LINKS = chain_urdf.getNrOfSegments();

    /* KDL solvers */
    std::shared_ptr<KDL::ChainJntToJacDotSolver> jacobDotSolver;
    std::shared_ptr<KDL::ChainFkSolverPos_recursive> fkSolverPos;
    std::shared_ptr<KDL::ChainFkSolverVel_recursive> fkSolverVel;
    std::shared_ptr<KDL::ChainIkSolverVel_pinv> ikSolverAcc;
    std::shared_ptr<KDL::ChainIdSolver_RNE> idSolver;

    const KDL::Vector GRAVITY(gravitational_acceleration[0], gravitational_acceleration[1], gravitational_acceleration[2]);

    // intialise KDL solvers using the chain
    jacobDotSolver = std::make_shared<KDL::ChainJntToJacDotSolver>(chain_urdf);
    fkSolverPos = std::make_shared<KDL::ChainFkSolverPos_recursive>(chain_urdf);
    fkSolverVel = std::make_shared<KDL::ChainFkSolverVel_recursive>(chain_urdf);
    ikSolverAcc = std::make_shared<KDL::ChainIkSolverVel_pinv>(chain_urdf);
    idSolver = std::make_shared<KDL::ChainIdSolver_RNE>(chain_urdf, GRAVITY);

    // left arm
    // const KDL::Vector BL_z_axis_wrt_GF(0.6992, 0.7092, -0.09);
    // const KDL::Vector BL_y_axis_wrt_GF(0.7122, -0.702, 0.0);
    // const KDL::Vector BL_x_axis_wrt_GF(0.0631, 0.064, 0.9959);
    // right arm
    const KDL::Vector BL_x_axis_wrt_GF(0.0631, -0.064, 0.9959);    
    const KDL::Vector BL_y_axis_wrt_GF(-0.7122, -0.702, 0.0);
    const KDL::Vector BL_z_axis_wrt_GF(0.6992, -0.7092, -0.09);
    const KDL::Vector BL_position_wrt_GF(0., -0.08, 0.0);

    const KDL::Rotation BL_wrt_GF(BL_x_axis_wrt_GF, BL_y_axis_wrt_GF, BL_z_axis_wrt_GF); // added as columns
    const KDL::Frame BL_wrt_GF_frame(BL_wrt_GF, BL_position_wrt_GF);

    // end effector Pose
    KDL::Frame measured_endEffPose_BL_arm;
    KDL::Frame measured_endEffPose_GF_arm;
    KDL::FrameVel measured_endEffTwist_BL_arm;
    KDL::FrameVel measured_endEffTwist_GF_arm;

    // Joint variables
    KDL::JntArray jnt_positions(kinova_constants::NUMBER_OF_JOINTS);
    KDL::JntArray jnt_velocities(kinova_constants::NUMBER_OF_JOINTS);   // has only joint velocities of all joints
    KDL::JntArray jnt_torques_read(kinova_constants::NUMBER_OF_JOINTS); // to read from the robot

    KDL::JntArray jnt_torques_cmd(kinova_constants::NUMBER_OF_JOINTS); // to send to the robot
    KDL::JntArrayVel jnt_velocity(kinova_constants::NUMBER_OF_JOINTS); // has both joint position and joint velocity of all joints

    KDL::JntArray jnt_accelerations(kinova_constants::NUMBER_OF_JOINTS);

    KDL::JntArray zero_jnt_velocities(kinova_constants::NUMBER_OF_JOINTS);
    zero_jnt_velocities.data.setZero();

    // Link wrenches
    KDL::Wrenches linkWrenches_GF(NUM_LINKS, KDL::Wrench::Zero());
    KDL::Wrenches linkWrenches_EE(NUM_LINKS, KDL::Wrench::Zero());

    // cartesian acceleration
    KDL::Twist xdd;
    KDL::Twist xdd_minus_jd_qd;
    KDL::Twist jd_qd;

    // TODO: initialise based on the motion specification

    double time_period_of_complete_controller_cycle_data = 0.0;

    double stiffness_lin_x_axis_data = STIFFNESS_GAIN_X;
    double stiffness_lin_y_axis_data = STIFFNESS_GAIN_Y;
    double stiffness_lin_z_axis_data = STIFFNESS_GAIN_Z;

    double damping_lin_x_axis_data = DAMPING_GAIN_X;
    double damping_lin_y_axis_data = DAMPING_GAIN_Y;
    double damping_lin_z_axis_data = DAMPING_GAIN_Z;

    double stiffness_roll_axis_data = STIFFNESS_GAIN_ROLL;
    double stiffness_pitch_axis_data = STIFFNESS_GAIN_PITCH;
    double stiffness_yaw_axis_data = STIFFNESS_GAIN_YAW;

    double measured_lin_pos_x_axis_data = 0.0;
    double measured_lin_pos_y_axis_data = 0.0;
    double measured_lin_pos_z_axis_data = 0.0;

    double measured_lin_vel_x_axis_data = 0.0;
    double measured_lin_vel_y_axis_data = 0.0;
    double measured_lin_vel_z_axis_data = 0.0;

    double lin_pos_sp_x_axis_data = 0.0;
    double lin_pos_sp_y_axis_data = 0.0;
    double lin_pos_sp_z_axis_data = 0.0;

    double lin_vel_sp_x_axis_data = 0.0;
    double lin_vel_sp_y_axis_data = 0.0;
    double lin_vel_sp_z_axis_data = 0.0;

    double stiffness_term_lin_x_axis_data = 0.0;
    double stiffness_term_lin_y_axis_data = 0.0;
    double stiffness_term_lin_z_axis_data = 0.0;

    double damping_term_x_axis_data = 0.0;
    double damping_term_y_axis_data = 0.0;
    double damping_term_z_axis_data = 0.0;

    double lin_pos_error_stiffness_x_axis_data = 0.0;
    double lin_pos_error_stiffness_y_axis_data = 0.0;
    double lin_pos_error_stiffness_z_axis_data = 0.0;

    double lin_vel_error_damping_x_axis_data = 0.0;
    double lin_vel_error_damping_y_axis_data = 0.0;
    double lin_vel_error_damping_z_axis_data = 0.0;

    double stiffness_damping_terms_summation_x_axis_data = 0.0;
    double stiffness_damping_terms_summation_y_axis_data = 0.0;
    double stiffness_damping_terms_summation_z_axis_data = 0.0;

    double apply_ee_force_x_axis_data = 0.0;
    double apply_ee_force_y_axis_data = 0.0;
    double apply_ee_force_z_axis_data = 0.0;

    double apply_ee_torque_x_axis_data = 0.0;
    double apply_ee_torque_y_axis_data = 0.0;
    double apply_ee_torque_z_axis_data = 0.0;

    double measured_roll_data = 0.0;
    double measured_pitch_data = 0.0;
    double measured_yaw_data = 0.0;

    double force_to_apply_x_axis = 0.0;
    double force_to_apply_y_axis = 0.0;
    double force_to_apply_z_axis = 0.0;

    KDL::Vector angle_axis_diff_GF_arm;
    KDL::Frame desired_endEffPose_GF_arm;
    double desired_quat_GF[4] = {0.0, 0.0, 0.0, 1.0};


    // initialise multi-dimensional array to store data
    std::vector<std::array<double, 20>> data_array_log;
    int iterationCount = 0;
    
    // logging
    std::string log_file = "log_files/kinova_arm_ctrl_log_file.csv";
    std::ofstream data_stream_log(log_file);

    if (!data_stream_log.is_open())
    {
        std::cerr << "Failed to open file: " << log_file << std::endl;
        return 0;
    }
    // adding header
    data_stream_log << "time_period_of_complete_controller_cycle_data,measured_lin_pos_x_axis_data,measured_lin_pos_y_axis_data,measured_lin_pos_z_axis_data,measured_lin_vel_x_axis_data,measured_lin_vel_y_axis_data,measured_lin_vel_z_axis_data,apply_ee_force_x_axis_data,apply_ee_force_y_axis_data,apply_ee_force_z_axis_data,apply_ee_torque_x_axis_data,apply_ee_torque_y_axis_data,apply_ee_torque_z_axis_data,jnt_torque_command_0,jnt_torque_command_1,jnt_torque_command_2,jnt_torque_command_3,jnt_torque_command_4,jnt_torque_command_5,jnt_torque_command_6\n";


    // joint torques that will be calculated before setting the control mode
    std::vector<double> rne_output_jnt_torques_vector_to_set_control_mode(kinova_constants::NUMBER_OF_JOINTS);

    kinova_mediator kinova_arm; // 192.168.1.10 (KINOVA_GEN3_1) // 192.168.1.12 (KINOVA_GEN3_2)

    // robot communication
    if (robots_to_control == robot_controlled::KINOVA_GEN3_1_LEFT)
    {
        kinova_arm.kinova_id = KINOVA_GEN3_1;
        kinova_arm.initialize(kinova_environment::REAL, robot_id::KINOVA_GEN3_1,
                              0.0);
    }
    else if (robots_to_control == robot_controlled::KINOVA_GEN3_2_RIGHT)
    {
        kinova_arm.kinova_id = KINOVA_GEN3_2;
        kinova_arm.initialize(kinova_environment::REAL, robot_id::KINOVA_GEN3_2,
                              0.0);
    }
    else
    {
        std::cout << "Invalid robot to control" << std::endl;
        flag = 1; // stop the execution
        return 0;
    }

    kinova_feedback(kinova_arm, jnt_positions, jnt_velocities,
                    jnt_torques_read);

    get_end_effector_pose_and_twist(
        jnt_velocity, jnt_positions, jnt_velocities,
        measured_endEffPose_BL_arm, measured_endEffTwist_BL_arm,
        measured_endEffPose_GF_arm, measured_endEffTwist_GF_arm,
        fkSolverPos, fkSolverVel, BL_wrt_GF_frame);

    calculate_joint_torques_RNEA(jacobDotSolver, ikSolverAcc, idSolver,
                                 jnt_velocity, jd_qd, xdd,
                                 xdd_minus_jd_qd, jnt_accelerations,
                                 jnt_positions, jnt_velocities,
                                 linkWrenches_EE, jnt_torques_cmd);

    // convert JntArray to double array
    for (int i = 0; i < kinova_constants::NUMBER_OF_JOINTS; i++)
    {
        rne_output_jnt_torques_vector_to_set_control_mode[i] =
            jnt_torques_cmd(i);
        std::cout << "jnt_torques_cmd: [" << i << "]  " << rne_output_jnt_torques_vector_to_set_control_mode[i] << std::endl;
    }
    kinova_arm.set_control_mode(control_mode::TORQUE, rne_output_jnt_torques_vector_to_set_control_mode.data());

    auto start_time_of_task = std::chrono::high_resolution_clock::now(); // start_time_of_task.count() gives time in seconds
    const auto task_time_out = std::chrono::duration<double>(TIMEOUT_DURATION_TASK);
    auto previous_time = std::chrono::high_resolution_clock::now();
    auto time_elapsed = std::chrono::duration<double>(previous_time - start_time_of_task);

    bool pre_condition_satisfied = false;
    bool post_condition_satisfied = false;
    bool prevail_condition_satisfied = false;
    std::string constraint_type_str;
    std::cout << "Waiting for pre-condition satisfaction" << std::endl;

    while (time_elapsed < task_time_out)
    {
        // if any interruption (Ctrl+C or window resizing) is detected
        if (flag == 1)
        {
        // logging
        // Write remaining data to file
        if (!data_array_log.empty())
        {
            appendDataToFile_dynamic_size(data_stream_log, data_array_log);
            data_array_log.clear();
        }

        data_stream_log.close();
        std::cout << "Data collection completed.\n";            
        break;
        }

        kinova_feedback(kinova_arm, jnt_positions, jnt_velocities,
                        jnt_torques_read);
        get_end_effector_pose_and_twist(
            jnt_velocity, jnt_positions, jnt_velocities,
            measured_endEffPose_BL_arm, measured_endEffTwist_BL_arm,
            measured_endEffPose_GF_arm, measured_endEffTwist_GF_arm,
            fkSolverPos, fkSolverVel, BL_wrt_GF_frame);

        // update the time variables
        auto current_time = std::chrono::high_resolution_clock::now();
        auto time_period = std::chrono::duration<double>(current_time - previous_time);

        while (time_period.count() < DESIRED_TIME_STEP)
        {
            current_time = std::chrono::high_resolution_clock::now();
            time_period = std::chrono::duration<double>(current_time - previous_time);
        }

        time_elapsed = std::chrono::duration<double>(current_time - start_time_of_task);
        previous_time = current_time;
        time_period_of_complete_controller_cycle_data = time_period.count();
        // std::cout << "time_period: " << time_period_of_complete_controller_cycle_data << std::endl;

        measured_lin_pos_x_axis_data = measured_endEffPose_GF_arm.p.x();
        measured_lin_vel_x_axis_data = measured_endEffTwist_GF_arm.GetTwist().vel.x();
        measured_lin_pos_y_axis_data = measured_endEffPose_GF_arm.p.y();
        measured_lin_vel_y_axis_data = measured_endEffTwist_GF_arm.GetTwist().vel.y();
        measured_lin_pos_z_axis_data = measured_endEffPose_GF_arm.p.z();
        measured_lin_vel_z_axis_data = measured_endEffTwist_GF_arm.GetTwist().vel.z();
        measured_endEffPose_GF_arm.M.GetRPY(measured_roll_data, measured_pitch_data, measured_yaw_data);

        // std::cout << "measured_lin_pos_x_axis_data: " << measured_lin_pos_x_axis_data << std::endl;

        // TODO: gather f-t values from ft sensor
        // check if any motion specification satisfies pre condition
        if (!pre_condition_satisfied)
        {
            check_pre_or_post_or_prevail_condition_satisfaction(
                measured_lin_pos_x_axis_data,
                measured_lin_pos_y_axis_data,
                measured_lin_pos_z_axis_data,
                measured_roll_data,
                measured_pitch_data,
                measured_yaw_data,
                measured_lin_vel_x_axis_data,
                measured_lin_vel_y_axis_data,
                measured_lin_vel_z_axis_data,
                linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS],
                pre_condition_constraint_count,
                constraint_type_str,
                arm_name,
                pre_condition_satisfied,
                motion_specification_params,
                condition_type::PRE_CONDITION);

            if (pre_condition_satisfied)
            {
                std::cout << "Pre condition satisfied. Now running controller to achieve per-condition until post-condition is satisfied." << std::endl;
            }
        }

        if (pre_condition_satisfied)
        {
            // check if the motion specification satisfies post condition
            check_pre_or_post_or_prevail_condition_satisfaction(
                measured_lin_pos_x_axis_data,
                measured_lin_pos_y_axis_data,
                measured_lin_pos_z_axis_data,
                measured_roll_data,
                measured_pitch_data,
                measured_yaw_data,
                measured_lin_vel_x_axis_data,
                measured_lin_vel_y_axis_data,
                measured_lin_vel_z_axis_data,
                linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS],
                post_condition_constraint_count,
                constraint_type_str,
                arm_name,
                post_condition_satisfied,
                motion_specification_params,
                condition_type::POST_CONDITION);

            check_pre_or_post_or_prevail_condition_satisfaction(
                measured_lin_pos_x_axis_data,
                measured_lin_pos_y_axis_data,
                measured_lin_pos_z_axis_data,
                measured_roll_data,
                measured_pitch_data,
                measured_yaw_data,
                measured_lin_vel_x_axis_data,
                measured_lin_vel_y_axis_data,
                measured_lin_vel_z_axis_data,
                linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS],
                prevail_condition_constraint_count,
                constraint_type_str,
                arm_name,
                prevail_condition_satisfied,
                motion_specification_params,
                condition_type::PREVAIL_CONDITION);

            if (post_condition_satisfied)
            {
                std::cout << "Post condition satisfied. Motion specification execution successful. Stoping execution." << std::endl;
                // setting to Position mode: only safe when joint velocities are close to zero
                // kinova_arm.set_control_mode(control_mode::POSITION, rne_output_jnt_torques_vector_to_set_control_mode.data());
                flag = 1; // stop the execution
                // TODO: go to position control mode and end the execution
            }
            else if (!prevail_condition_satisfied)
            {
                std::cout << "Prevail condition is not satisfied. Motion specification execution unsuccessful. Stoping execution." << std::endl;
                // setting to Position mode: only safe when joint velocities are close to zero
                // kinova_arm.set_control_mode(control_mode::POSITION, rne_output_jnt_torques_vector_to_set_control_mode.data());
                flag = 1; // stop the execution
                // TODO: go to position control mode and end the execution
            }
            else
            {
                get_setpoints_from_motion_specification(
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
                    desired_quat_GF,
                    motion_specification_params,
                    arm_name);

                get_force_and_torque_from_controller_described_in_GF_to_apply_at_EE(
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
                    desired_quat_GF,
                    apply_ee_force_x_axis_data,
                    apply_ee_force_y_axis_data,
                    apply_ee_force_z_axis_data,
                    apply_ee_torque_x_axis_data,
                    apply_ee_torque_y_axis_data,
                    apply_ee_torque_z_axis_data,
                    desired_endEffPose_GF_arm,
                    measured_endEffPose_GF_arm,
                    per_condition_constraint_count,
                    angle_axis_diff_GF_arm,
                    motion_specification_params,
                    arm_name);
            }
        }

        // write the ee torques to linkWrenches_GF
        linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(0) = -apply_ee_force_x_axis_data;
        linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(1) = -apply_ee_force_y_axis_data;
        linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(2) = -apply_ee_force_z_axis_data;
        // TODO: apply only when a per condition constraint exists for orientation
        linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(0) = -apply_ee_torque_x_axis_data;
        linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(1) = -apply_ee_torque_y_axis_data;
        linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(2) = -apply_ee_torque_z_axis_data;

        apply_ee_force_x_axis_data = 0.0;
        apply_ee_force_y_axis_data = 0.0;
        apply_ee_force_z_axis_data = 0.0;

        apply_ee_torque_x_axis_data = 0.0;
        apply_ee_torque_y_axis_data = 0.0;
        apply_ee_torque_z_axis_data = 0.0;

        // thresholding in cartesian space of the end effector
        for (int i = 0; i < 3; i++)
        {
            if (linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(i) > 0.0)
            {
                linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(i) = std::min(WRENCH_THRESHOLD_LINEAR, linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(i));
            }
            else
            {
                linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(i) = std::max(-WRENCH_THRESHOLD_LINEAR, linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(i));
            }

            if (linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(i) > 0.0)
            {
                linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(i) = std::min(WRENCH_THRESHOLD_ROTATIONAL, linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(i));
            }
            else
            {
                linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(i) = std::max(-WRENCH_THRESHOLD_ROTATIONAL, linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(i));
            }
        }

        // LinkWrenches are calculated in BL frame. As RNE solver requires them in EE frame, the wrenches are transformed from BL to EE frame
        linkWrenches_EE[NUM_LINKS - 1].force = measured_endEffPose_GF_arm.M.Inverse() * linkWrenches_GF[NUM_LINKS - 1].force;
        linkWrenches_EE[NUM_LINKS - 1].torque = measured_endEffPose_GF_arm.M.Inverse() * linkWrenches_GF[NUM_LINKS - 1].torque;

        calculate_joint_torques_RNEA(jacobDotSolver, ikSolverAcc, idSolver,
                                     jnt_velocity, jd_qd, xdd,
                                     xdd_minus_jd_qd, jnt_accelerations,
                                     jnt_positions, jnt_velocities,
                                     linkWrenches_EE, jnt_torques_cmd);

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
        kinova_arm.set_joint_torques(jnt_torques_cmd);
        data_array_log.push_back({time_period_of_complete_controller_cycle_data,measured_lin_pos_x_axis_data,measured_lin_pos_y_axis_data,measured_lin_pos_z_axis_data,measured_lin_vel_x_axis_data,measured_lin_vel_y_axis_data,measured_lin_vel_z_axis_data,linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(0),linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(1),linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(2),linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(0),linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(1),linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(2),jnt_torques_cmd(0),jnt_torques_cmd(1),jnt_torques_cmd(2),jnt_torques_cmd(3),jnt_torques_cmd(4),jnt_torques_cmd(5),jnt_torques_cmd(6)});

        // Check if we should write to file
        iterationCount++;
        if (iterationCount % SAVE_LOG_EVERY_NTH_STEP == 0)
        {
        appendDataToFile_dynamic_size(data_stream_log, data_array_log);
        data_array_log.clear();
        }
    }

  if (!data_array_log.empty())
  {
    appendDataToFile_dynamic_size(data_stream_log, data_array_log);
    data_array_log.clear();
  }

  data_stream_log.close();
  std::cout << "Data collection completed.\n";
  return 0;
}
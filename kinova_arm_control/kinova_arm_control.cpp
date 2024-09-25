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
    KINOVA_GEN3_1_LEFT = 1,
    KINOVA_GEN3_2_RIGHT = 2,
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
    std::string arm_name;

    int pre_condition_constraint_count = 0;
    int per_condition_constraint_count = 0;
    int post_condition_constraint_count = 0;

    // set robots to control
    robot_controlled robots_to_control = robot_controlled::KINOVA_GEN3_2_RIGHT;

    YAML::Node config_file = YAML::LoadFile("parameters.yaml");
    YAML::Node impedance_gains = YAML::LoadFile("impedance_gains.yaml");                  // in global frame
    YAML::Node motion_specification_params = YAML::LoadFile("motion_specification.yaml"); // in global frame

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

    pre_condition_constraint_count = motion_specification_params[arm_name]["pre_condition"]["constraint_count"].as<int>();
    per_condition_constraint_count = motion_specification_params[arm_name]["per_condition"]["constraint_count"].as<int>();
    post_condition_constraint_count = motion_specification_params[arm_name]["post_condition"]["constraint_count"].as<int>();

    std::vector<float> gravitational_acceleration = motion_specification_params[arm_name]["gravitational_acceleration"].as<std::vector<float>>();
    TIMEOUT_DURATION_TASK = config_file["TIMEOUT_DURATION_TASK"].as<double>(); // seconds
    WRENCH_THRESHOLD_LINEAR = config_file["WRENCH_THRESHOLD_LINEAR"].as<double>();
    WRENCH_THRESHOLD_ROTATIONAL = config_file["WRENCH_THRESHOLD_ROTATIONAL"].as<double>();
    JOINT_TORQUE_THRESHOLD = config_file["JOINT_TORQUE_THRESHOLD"].as<double>();
    DESIRED_TIME_STEP = config_file["DESIRED_TIME_STEP"].as<double>();

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

    // transformations
    const KDL::Vector BL_x_axis_wrt_GF(1.0, 0.0, 0.0);
    const KDL::Vector BL_y_axis_wrt_GF(0.0, 1.0, 0.0);
    const KDL::Vector BL_z_axis_wrt_GF(0.0, 0.0, 1.0);
    const KDL::Vector BL_position_wrt_GF(0., 0.0, 0.0);

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

    KDL::Vector angle_axis_diff_GF_arm;
    KDL::Frame desired_endEffPose_GF_arm;
    double desired_quat_GF[4] = {0.0, 0.0, 0.0, 0.};
    desired_endEffPose_GF_arm.M = KDL::Rotation::Quaternion(desired_quat_GF[0], desired_quat_GF[1], desired_quat_GF[2], desired_quat_GF[3]);    

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

    double stiffness_angular_arm[3] = {STIFFNESS_GAIN_ROLL, STIFFNESS_GAIN_PITCH, STIFFNESS_GAIN_YAW};

    auto start_time_of_task = std::chrono::high_resolution_clock::now(); // start_time_of_task.count() gives time in seconds
    const auto task_time_out = std::chrono::duration<double>(TIMEOUT_DURATION_TASK);
    auto previous_time = std::chrono::high_resolution_clock::now();
    auto time_elapsed = std::chrono::duration<double>(previous_time - start_time_of_task);

    while (time_elapsed < task_time_out)
    {

        // if any interruption (Ctrl+C or window resizing) is detected
        if (flag == 1)
        {
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
        std::cout << "time_period: " << time_period_of_complete_controller_cycle_data << std::endl;

        measured_lin_pos_x_axis_data = measured_endEffPose_GF_arm.p.x();
        measured_lin_vel_x_axis_data = measured_endEffTwist_GF_arm.GetTwist().vel.x();
        measured_lin_pos_y_axis_data = measured_endEffPose_GF_arm.p.y();
        measured_lin_vel_y_axis_data = measured_endEffTwist_GF_arm.GetTwist().vel.y();
        measured_lin_pos_z_axis_data = measured_endEffPose_GF_arm.p.z();
        measured_lin_vel_z_axis_data = measured_endEffTwist_GF_arm.GetTwist().vel.z();

        angle_axis_diff_GF_arm = KDL::diff(measured_endEffPose_GF_arm.M, desired_endEffPose_GF_arm.M);


        // TODO: implement the motion specification realisation
        for (int i = 0; i < pre_condition_constraint_count; i++)
        {
            std::string constraint_name = motion_specification_params[arm_name]["pre_condition"]["constraints"][i]["type"].as<std::string>();
            double constraint_val_x = motion_specification_params[arm_name]["pre_condition"]["constraints"][i]["value"][0].as<double>();
            double constraint_val_y = motion_specification_params[arm_name]["pre_condition"]["constraints"][i]["value"][1].as<double>();
            double constraint_val_z = motion_specification_params[arm_name]["pre_condition"]["constraints"][i]["value"][2].as<double>();
            std::vector<float> constraint_values = motion_specification_params[arm_name]["pre_condition"]["constraints"][i]["values"].as<std::vector<float>>();
        };

        // controller (per-condition)
        subtraction(&lin_pos_sp_z_axis_data, &measured_lin_pos_z_axis_data, &lin_pos_error_stiffness_z_axis_data);

        multiply2(&stiffness_lin_z_axis_data, &lin_pos_error_stiffness_z_axis_data, &stiffness_term_lin_z_axis_data);

        subtraction(&lin_vel_sp_z_axis_data, &measured_lin_vel_z_axis_data, &lin_vel_error_damping_z_axis_data);

        multiply2(&damping_lin_z_axis_data, &lin_vel_error_damping_z_axis_data, &damping_term_z_axis_data);

        summation2(&stiffness_term_lin_z_axis_data, &damping_term_z_axis_data, &stiffness_damping_terms_summation_z_axis_data);

        summation2(&stiffness_damping_terms_summation_z_axis_data, &apply_ee_force_z_axis_data, &apply_ee_force_z_axis_data);

        // write the ee torques to to linkWrenches_GF
        linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(0) = -apply_ee_force_x_axis_data;
        linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(1) = -apply_ee_force_y_axis_data;
        linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].force(2) = -apply_ee_force_z_axis_data;
        linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(0) = -stiffness_angular_arm[0] * angle_axis_diff_GF_arm(0);
        linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(1) = -stiffness_angular_arm[1] * angle_axis_diff_GF_arm(1);
        linkWrenches_GF[kinova_constants::NUMBER_OF_JOINTS].torque(2) = -stiffness_angular_arm[2] * angle_axis_diff_GF_arm(2);

        apply_ee_force_x_axis_data = 0.0;
        apply_ee_force_y_axis_data = 0.0;
        apply_ee_force_z_axis_data = 0.0;

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
    }

    return 0;
}
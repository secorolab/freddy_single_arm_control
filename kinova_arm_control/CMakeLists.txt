# specifying name of the executable as the same as project name
# add_executable(${PROJECT_NAME} src/adaptive_control.cpp)
# add_executable(simple_adaptive_pid_binary simple_adaptive_pid.cpp)
add_executable(kinova_arm_control ${CMAKE_CURRENT_SOURCE_DIR}/kinova_arm_control.cpp)

# PUBLIC keyword is used to specify thet other projects that depend on the current one should also be linked to the executable
target_link_libraries(
    kinova_arm_control PUBLIC orocos-kdl
    control_blocks_binary
    monitors_binary
    yaml-cpp
    kdl_parser
    ${CMAKE_CURRENT_SOURCE_DIR}/../kinova_mediator/kinova_api/lib/release/libKortexApiCpp.a
    ${CMAKE_CURRENT_SOURCE_DIR}/../kinova_mediator/build/libkinova_mediator.so
    )

    target_include_directories(kinova_arm_control PUBLIC)
# set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "output_name")

# set compiler version to c++17
target_compile_features(kinova_arm_control PRIVATE cxx_std_17)
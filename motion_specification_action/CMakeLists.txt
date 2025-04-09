cmake_minimum_required(VERSION 3.8)
project(motion_specification_action)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_compile_definitions(_OS_UNIX)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(motion_specification_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(kinova_mediator REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Define Kinova API include path
set(KINOVA_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../kinova_mediator/kinova_api)

# =============================
# Action Client Library
# =============================
add_library(action_client SHARED src/motion_specification_client.cpp)

target_include_directories(action_client PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_definitions(action_client PRIVATE "MOTION_SPECIFICATION_ACTION_BUILDING_DLL")

ament_target_dependencies(action_client
  "motion_specification_interfaces"
  "rclcpp"
  "rclcpp_action"
  "rclcpp_components"
)

rclcpp_components_register_node(action_client PLUGIN "motion_specification_action::MotionSpecificationActionClient" EXECUTABLE motion_specification_action_client)

install(TARGETS action_client
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# =============================
# Action Server Library
# =============================
add_library(action_server SHARED 
  src/motion_specification_server.cpp
  src/monitors.c
  src/control_blocks.c)

target_include_directories(action_server PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${orocos_kdl_INCLUDE_DIRS}
  ${KINOVA_API_PATH}/include
  ${KINOVA_API_PATH}/include/client
  ${KINOVA_API_PATH}/include/client_stubs
  ${KINOVA_API_PATH}/include/common
  ${KINOVA_API_PATH}/include/google
  ${KINOVA_API_PATH}/include/messages
)

target_compile_definitions(action_server PRIVATE "MOTION_SPECIFICATION_ACTION_BUILDING_DLL")

# ament_target_dependencies() works best with ament-based dependencies. For system libraries, target-link_libraries() is preferred.
ament_target_dependencies(action_server
  "motion_specification_interfaces"
  "rclcpp"
  "rclcpp_action"
  "rclcpp_components"
  "kdl_parser"
  "orocos_kdl"
  "Eigen3"
  "yaml-cpp"
  "kinova_mediator"
  "ament_index_cpp"
  "sensor_msgs"
  "geometry_msgs"
)

target_link_libraries(action_server yaml-cpp)

rclcpp_components_register_node(action_server PLUGIN "motion_specification_action::MotionSpecificationActionServer" EXECUTABLE motion_specification_action_server)

install(TARGETS action_server
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install the config folder (with the YAML files)
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install the urdf folder (with the robot URDF files)
install(
  DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

# =============================
# Linting & Testing
# =============================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

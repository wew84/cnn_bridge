cmake_minimum_required(VERSION 2.8.3)
project(cnn_bridge)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
  geometry_msgs
)

catkin_python_setup()

add_message_files(
  FILES
  Netmask.msg
  Detection.msg
  SegmentationClass.msg
)

add_service_files(
  FILES
  getSegmentationMetadata.srv
  getDetectionMetadata.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime roscpp rospy
)
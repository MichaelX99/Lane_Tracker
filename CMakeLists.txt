cmake_minimum_required(VERSION 2.8.3)
project(lane_tracker)

set (CMAKE_CXX_STANDARD 11)

find_package(CUDA)
FIND_PACKAGE(Boost)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs
)

include_directories(
  include
  ${CUDA_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

cuda_add_executable(${PROJECT_NAME}
  src/Tools.cpp
  src/Cuda_Functions.cu
  src/Particle_Filter.cpp
  src/main.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${CUDA_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_CUBLAS_LIBRARIES}
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
)

## Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

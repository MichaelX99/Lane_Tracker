cmake_minimum_required(VERSION 2.8.3)
project(lane_tracker)

set (CMAKE_CXX_STANDARD 11)

find_package(CUDA)

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
  ${catkin_INCLUDE_DIRS}
)

cuda_add_executable(particle_filter
  src/Cuda_Functions.cu
  src/Particle_Filter.cpp
  src/main.cpp
)

target_link_libraries(particle_filter
  ${CUDA_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_CUBLAS_LIBRARIES}
  ${catkin_LIBRARIES}
)

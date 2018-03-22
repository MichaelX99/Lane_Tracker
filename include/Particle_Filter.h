#ifndef _PARTICLE_FILTER_
#define _PARTICLE_FILTER_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32.h>

#include <cublas_v2.h>
#include <Cuda_Functions.h>

#include <vector>

#include <boost/filesystem.hpp>

#include "Tools.h"

class Particle_Filter
{
public:
  Particle_Filter();
  ~Particle_Filter();
  void img_callback(const sensor_msgs::Image& msg);
  void obs_callback(const std_msgs::Float32& msg);

private:
  ros::NodeHandle _nh;
  ros::Subscriber _img_sub = _nh.subscribe("lane/image", 1, &Particle_Filter::img_callback, this);
  ros::Subscriber _obs_sub = _nh.subscribe("lane/observed_state", 1, &Particle_Filter::obs_callback, this);
  ros::Publisher _pub = _nh.advertise<std_msgs::Float32>("lane/predicted_state", 1);

  cublasHandle_t _handle;

  std::string _transition_path;
  std::string _state_path;
  int _num_states;
  int _num_particles;

  std::vector<float> _states;

  // Memory for particles
  float* _d_particle_matrix;
  float* _d_avg_particle;

  // Memory for transition matrix
  float* _h_transition_matrix;
  float* _d_transition_matrix;

  float* _d_sensor_observation;

  float *_d_row_sum, _d_col_sum;

  void load_state_vector();

  // Function to read in the transition matrix file
  void load_transition_matrix();
};

#endif

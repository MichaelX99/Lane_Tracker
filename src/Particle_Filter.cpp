#include <Particle_Filter.h>

Particle_Filter::Particle_Filter()
{
  _nh.getParam("state_path", _state_path);
  _nh.getParam("transform_path", _transform_path);
  _nh.getParam("num_states", _num_states);
  _nh.getParam("num_particles", _num_particles);

  cublasCreate(&_handle);

  cuda_initialize_particles(_d_particles, _num_states, _num_particles);

  load_transition_matrix();
  load_state_vector();
}

Particle_Filter::~Particle_Filter()
{
  cublasDestroy(_handle);
  free(_h_transition_matrix);

  cuda_destroy(_d_particles);
  cuda_destroy(_d_transition_matrix);
}

void Particle_Filter::img_callback(const sensor_msgs::Image& msg)
{
  cuda_apply_transition(_handle);

  int index = cuda_compute_argmax_state(_d_particles, _num_states, _num_particles);

  // Find which state the index is
  float state;

  std_msgs::Float32 output_msg;
  output_msg.data = state;
  _pub.publish(output_msg);
}

void Particle_Filter::obs_callback(const std_msgs::Float32& msg)
{
  float state = msg.data;

  cuda_form_obs_vector();
  cuda_reweight_particles();
  cuda_resample_particles();
}

void Particle_Filter::load_state_vector()
{

}

void Particle_Filter::load_transition_matrix()
{
  cuda_copy_transition_matrix(_h_transition_matrix, _d_transition_matrix, _num_states);
}

#include <Particle_Filter.h>

Particle_Filter::Particle_Filter()
{
  if(!_nh.getParam("state_path", _state_path)) { printf("State vector path parameter not found\n"); }
  if(!_nh.getParam("transition_path", _transition_path)) { printf("Tansition matrix path parameter not found\n"); }
  if(!_nh.getParam("num_states", _num_states)) { printf("Number of states variables not found\n"); }
  if(!_nh.getParam("num_particles", _num_particles)) { printf("Number of particles variable not found\n"); }

  // wait for the python script to generate these files
  while ( !boost::filesystem::exists(_state_path) ) { }
  while ( !boost::filesystem::exists(_transition_path) ) { }
  std::cout << "Done waiting on file generation\n";

  cublasCreate(&_handle);

  _d_particle_ones = initialize_gpu_ones(_d_particle_ones, _num_states);
  _d_state_ones = initialize_gpu_ones(_d_state_ones, _num_states);
  _d_sensor_observation = initialize_gpu_array(_d_sensor_observation, _num_states);
  _d_avg_particle = initialize_gpu_array(_d_avg_particle, _num_states);
  _d_row_sum = initialize_gpu_array(_d_row_sum, _num_particles);
  _d_col_sum = initialize_gpu_array(_d_col_sum, _num_states);
  _d_state_sum = initialize_gpu_array(_d_state_sum, 1);

  _d_particle_matrix = cuda_initialize_particles(_handle, _d_particle_matrix, _d_particle_ones, _d_row_sum, _num_states, _num_particles);

  /*
  load_transition_matrix();
  load_state_vector();

  printf("done initializing\n\n");

  print_matrix("particle matrix", _d_particle_matrix, _num_particles, _num_states);

  int index = cuda_compute_argmax_state(_handle, _d_particle_matrix, _d_avg_particle, _d_particle_ones, _d_state_sum, _d_state_ones, _d_col_sum, _num_states, _num_particles);
  */

}

Particle_Filter::~Particle_Filter()
{
  cublasDestroy(_handle);
  free(_h_transition_matrix);

  cuda_destroy(_d_particle_matrix);
  cuda_destroy(_d_avg_particle);
  cuda_destroy(_d_transition_matrix);
  cuda_destroy(_d_sensor_observation);
  cuda_destroy(_d_state_sum);
  cuda_destroy(_d_particle_ones);
  cuda_destroy(_d_state_ones);
  cuda_destroy(_d_row_sum);
  cuda_destroy(_d_col_sum);
}

void Particle_Filter::img_callback(const sensor_msgs::Image& msg)
{
  cuda_apply_transition(_handle, _d_particle_matrix, _d_transition_matrix, _d_particle_ones, _d_row_sum, _num_states, _num_particles);

  int index = cuda_compute_argmax_state(_handle, _d_particle_matrix, _d_avg_particle, _d_particle_ones, _d_state_sum, _d_state_ones, _d_col_sum, _num_states, _num_particles);

  // Find which state the index is
  float state = _states[index];

  std_msgs::Float32 output_msg;
  output_msg.data = state;
  _pub.publish(output_msg);
}

void Particle_Filter::obs_callback(const std_msgs::Float32& msg)
{
  float state = msg.data;

  // Garbage used to find the closest state index to what the observed state is
  int low_index, high_index;
  int closest_ind = -1;
  for (int i = 0; i < _states.size(); i++)
  {
    if (state == _states[i])
    {
      closest_ind = i;
    }
    else
    {
      if (_states[i] < state)
      {
        low_index = i;
      }
      else if (_states[i] > state)
      {
        high_index = i;
      }
    }
  }
  if (closest_ind == -1)
  {
    float low_diff, high_diff;

    low_diff = state - _states[low_index];
    high_diff = _states[high_index] - state;
    if (high_diff > low_diff)
    {
      closest_ind = low_index;
    }
    else
    {
      closest_ind = high_index;
    }
  }

  _d_sensor_observation = cuda_form_obs_vector(_d_sensor_observation, closest_ind);
  _d_particle_matrix = cuda_reweight_particles(_d_particle_matrix, _d_sensor_observation, _num_states, _num_particles);
  _d_particle_matrix = cuda_resample_particles(_d_particle_matrix);
}

void Particle_Filter::load_state_vector()
{
  _states = Populate_States(_state_path);
}

void Particle_Filter::load_transition_matrix()
{
  _h_transition_matrix = Populate_Transform(_transition_path);

  _d_transition_matrix = cuda_copy_transition_matrix(_h_transition_matrix, _d_transition_matrix, _num_states);
}

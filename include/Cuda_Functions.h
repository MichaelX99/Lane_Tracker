#ifndef _CUDA_FUNCTIONS_
#define _CUDA_FUNCTIONS_

#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <stdio.h>



float* cuda_initialize_particles(cublasHandle_t handle, float* d_particles, float* d_ones, const int num_states, const int num_particles);

float* cuda_apply_transition(cublasHandle_t handle, float* particles, float* transition, float* d_ones, const int num_states, const int num_particles);

float* cuda_copy_transition_matrix(float* h_transition_matrix, float* d_transition_matrix, const int num_states);

int cuda_compute_argmax_state(cublasHandle_t handle, float* d_particle_matrix, float* d_avg_particle, const int num_states, const int num_particles);

float* initialize_gpu_array(float* A, const int num_states);

float* initialize_gpu_ones(float* A, const int size);

float* cuda_form_obs_vector(float* sensor_observation, const int index);
float* cuda_reweight_particles(float* particle_matrix, float* sensor_observation, const int num_states, const int num_particles);
float* cuda_resample_particles(float* particle_matrix);

void cuda_destroy(float* d_A);
#endif

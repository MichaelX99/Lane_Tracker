#ifndef _CUDA_FUNCTIONS_
#define _CUDA_FUNCTIONS_

#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <stdio.h>

void cuda_initialize_particles(float* d_particles, const int num_states, const int num_particles);

void cuda_apply_transition(cublasHandle_t handle);

void cuda_copy_transition_matrix(float* h_transition_matrix, float* d_transition_matrix, const int num_states);

int cuda_compute_argmax_state(float* d_particles, const int num_states, const int num_particles);

void cuda_form_obs_vector();
void cuda_reweight_particles();
void cuda_resample_particles();

void cuda_destroy(float* d_A);
#endif

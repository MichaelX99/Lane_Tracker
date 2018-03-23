#include <Cuda_Functions.h>

const int BLOCK_SIZE = 16;

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

static void HandleCUDAError( cudaError_t err,
                             const char *file,
                             int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_CUDA_ERROR( err ) (HandleCUDAError( err, __FILE__, __LINE__ ))

static void HandleCUBLASError( cublasStatus_t err,
                               const char *file,
                               int line ) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf( "%s in %s at line %d\n", cublasGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_CUBLAS_ERROR( err ) (HandleCUBLASError( err, __FILE__, __LINE__ ))

const char* curandGetErrorString(curandStatus_t status)
{
    switch(status)
    {
        case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

static void HandleCURANDError( curandStatus_t err,
                               const char *file,
                               int line ) {
    if (err != CURAND_STATUS_SUCCESS) {
        printf( "%s in %s at line %d\n", curandGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_CURAND_ERROR( err ) (HandleCURANDError( err, __FILE__, __LINE__ ))

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void normalize(float* d_particle_matrix, float* d_row_sum, const int num_states, const int num_particles)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int matrix_ind;
  int vector_ind;

  if (row < num_particles && col < num_states)
  {
    matrix_ind = num_states * row + col;
    vector_ind = row;
    d_particle_matrix[matrix_ind] /= d_row_sum[vector_ind];
  }
}

__global__ void average_matrix(float* d_particle_matrix, float* d_avg_particle, const int num_states, const int num_particles)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < num_states && col < num_particles)
  {

  }
}

void print_matrix(char* name, float* A, const int rows, const int cols)
{
  float* h_A = (float*)malloc(rows * cols * sizeof(float));

  HANDLE_CUDA_ERROR( cudaMemcpy(h_A, A, rows * cols * sizeof(float), cudaMemcpyDeviceToHost) );

  int index;

  printf(name);
  printf("\n");

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      index = i * cols + j;
      printf("%f ", h_A[index]);
    }
    printf("\n");
  }
  printf("\n\n");

  free(h_A);
}

float* cuda_fill_rand(float *d_particle_matrix, const int num_states, const int num_particles)
{
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	HANDLE_CURAND_ERROR( curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT) );

	// Set the seed for the random number generator using the system clock
	HANDLE_CURAND_ERROR( curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock()) );

	// Fill the array with random numbers on the device
	HANDLE_CURAND_ERROR( curandGenerateUniform(prng, d_particle_matrix, num_states * num_particles) );

  return d_particle_matrix;
}

float* initialize_gpu_ones(float* A, const int size)
{
  HANDLE_CUDA_ERROR( cudaMalloc((void**)&A, size * sizeof(float)) );
  float* h_A = (float*) malloc(size * sizeof(float));
  for (int i = 0; i < size; i++)
  {
    h_A[i] = 1.0;
  }
  HANDLE_CUDA_ERROR( cudaMemcpy(A, h_A, size * sizeof(float), cudaMemcpyHostToDevice) );

  free(h_A);

  return A;
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
float* gpu_blas_mmul(float *A, float *B, float *C, const int m, const int k, const int n, cublasHandle_t handle)
{
	//int lda=m,ldb=k,ldc=m;
  int lda=m,ldb=m,ldc=n;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Do the actual multiplication
	//HANDLE_CUBLAS_ERROR( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) );

  HANDLE_CUBLAS_ERROR( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, m, B, k, beta, C, m) );

  return C;
}

// Matrix Vector Multiplication
// c(m,1) = A(m,n) * b(n,1)
float* gpu_blas_vmul(char* type, const float *A, const float *b, float *c, const int m, const int n, cublasHandle_t handle)
{
  int lda=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

  if (strcmp(type, "row") == 0)
  {
    HANDLE_CUBLAS_ERROR( cublasSgemv(handle, CUBLAS_OP_T, m, n, alpha, A, lda, b, 1, beta, c, 1) );
  }
  else if (strcmp(type, "col") == 0)
  {
    HANDLE_CUBLAS_ERROR( cublasSgemv(handle, CUBLAS_OP_N, m, n, alpha, A, lda, b, 1, beta, c, 1) );
  }


  return c;
}

float* cuda_normalize_particles(float *d_particle_matrix, float* d_particle_ones, float* d_row_sum, const int num_states, const int num_particles, cublasHandle_t handle)
{
  d_row_sum = gpu_blas_vmul("row", d_particle_matrix, d_particle_ones, d_row_sum, num_states, num_particles, handle);

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((num_states + dimBlock.x - 1) / dimBlock.x,
               (num_particles + dimBlock.y - 1) / dimBlock.y);

  normalize<<<dimGrid, dimBlock>>> (d_particle_matrix, d_row_sum, num_states, num_particles);

  HANDLE_CUDA_ERROR( cudaThreadSynchronize() );

  return d_particle_matrix;
}

float* cuda_initialize_particles(cublasHandle_t handle, float* d_particles, float* d_ones, float* d_row_sum, const int num_states, const int num_particles)
{
  HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_particles, num_states * num_particles * sizeof(float)) );
  //d_particles = cuda_fill_rand(d_particles, num_particles, num_states);
  float* particles = (float*)malloc(num_states * num_particles * sizeof(float));

  float val;
  for (int i = 0; i < num_states*num_particles; i++)
  {
    particles[i] = i;
  }

  HANDLE_CUDA_ERROR( cudaMemcpy(d_particles, particles, num_particles * num_states * sizeof(float), cudaMemcpyHostToDevice) );
  free(particles);


  d_particles = cuda_normalize_particles(d_particles, d_ones, d_row_sum, num_states, num_particles, handle);

  return d_particles;
}

float* cuda_copy_transition_matrix(float* h_transition_matrix, float* d_transition_matrix, const int num_states)
{
  HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_transition_matrix, num_states * num_states * sizeof(float)) );
  HANDLE_CUDA_ERROR( cudaMemcpy(d_transition_matrix, h_transition_matrix, num_states * num_states * sizeof(float), cudaMemcpyHostToDevice) );

  return d_transition_matrix;
}

int cuda_compute_argmax_state(cublasHandle_t handle,float* d_particle_matrix, float* d_avg_particle, const int num_states, const int num_particles)
{
  int state;

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((num_states + dimBlock.x - 1) / dimBlock.x,
               (num_particles + dimBlock.y - 1) / dimBlock.y);

  // compute the average particle state
  average_matrix<<<dimGrid, dimBlock>>> (d_particle_matrix, d_avg_particle, num_states, num_particles);

  HANDLE_CUDA_ERROR( cudaThreadSynchronize() );

  // find the argmax
  HANDLE_CUBLAS_ERROR( cublasIsamax(handle, num_states, d_avg_particle, 1, &state) );

  return state;
}

float* cuda_apply_transition(cublasHandle_t handle, float* particles, float* transition, float* d_ones, float* d_sum, const int num_states, const int num_particles)
{
  // matrix multiply
  particles = gpu_blas_mmul(transition, particles, particles, num_states, num_states, num_particles, handle);

  // normalize matrix
  particles = cuda_normalize_particles(particles, d_ones, d_sum, num_states, num_particles, handle);

  return particles;
}

float* initialize_gpu_array(float* A, const int num_states)
{
  HANDLE_CUDA_ERROR( cudaMalloc((void**)&A, num_states * sizeof(float)) );

  return A;
}

float* cuda_form_obs_vector(float* sensor_observation, const int index)
{

  return sensor_observation;
}

float* cuda_reweight_particles(float* particle_matrix, float* sensor_observation, const int num_states, const int num_particles)
{
  // write element wise multiplcation kernel

  return particle_matrix;
}

float* cuda_resample_particles(float* particle_matrix)
{

  return particle_matrix;
}

void cuda_destroy(float* d_A)
{
  cudaFree(d_A);
}

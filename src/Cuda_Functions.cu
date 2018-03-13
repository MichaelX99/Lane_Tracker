#include <Cuda_Functions.h>

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

void cuda_fill_rand(float *A, int rows, int cols)
{
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, rows * cols);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n, cublasHandle_t handle)
{
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Do the actual multiplication
	HANDLE_CUBLAS_ERROR( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) );
}

void cuda_normalize_particles(float *A, int rows, int cols)
{

}

void cuda_initialize_particles(float* d_particles, const int num_states, const int num_particles)
{
  HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_particles, num_states * num_particles * sizeof(float)) );
  cuda_fill_rand(d_particles, num_particles, num_states);
  cuda_normalize_particles(d_particles, num_particles, num_states);
}

void cuda_copy_transition_matrix(float* h_transition_matrix, float* d_transition_matrix, const int num_states)
{
  HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_transition_matrix, num_states * num_states * sizeof(float)) );
  HANDLE_CUDA_ERROR( cudaMemcpy(d_transition_matrix, h_transition_matrix, num_states * num_states * sizeof(float), cudaMemcpyHostToDevice) );
}

int cuda_compute_argmax_state(float* d_particles, const int num_states, const int num_particles)
{
  int state = 0;

  return state;
}

void cuda_apply_transition(cublasHandle_t handle)
{
  // matrix multiply

  // normalize matrix
}

void cuda_form_obs_vector()
{

}

void cuda_reweight_particles()
{

}

void cuda_resample_particles()
{

}

void cuda_destroy(float* d_A)
{
  cudaFree(d_A);
}

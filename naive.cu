#include "cuda_gemm.h"

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}



void generate_data(float *x, int n, int m) {
    static std::random_device dev;
    static std::mt19937 rng(dev());

    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (int i = 0; i < n*m; i++) x[i] = dist(rng);
}

// Matrix multiplication on GPU device
__global__ 
void cuda_mul(float *a, float *b, float *c, int n, int m, int p) {
    // In CUDA, the ordering of dimensions are reversed i.e. a matrix of dim (N, M, P) will be represented as (P, M, N) in CUDA
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        float res = 0.0;
        for (int i = 0; i < m; i++) res += a[row*m+i]*b[i*p+col];
        c[row*p+col] = res;
    }
}

// Matrix multiplication on CPU
void mat_mul(float *a, float *b, float *c, int n, int m, int p) {
    for (int i = 0; i < n*p; i++) c[i] = 0.0;

    omp_set_num_threads(8);
    #pragma omp parallel for shared(a, b, c)
    for(int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < p; k++) c[i*p+k] += a[i*m+j]*b[j*p+k];
        }
    }
}

int main(){
    int n = 2048;
    int m = 2048;
    int p = 2048;

    float *a, *b, *c, *d;

    size_t size_a = sizeof(float)*n*m;
    size_t size_b = sizeof(float)*m*p;
    size_t size_c = sizeof(float)*n*p;

    // a, b and c are defined for both CPU and GPU. Thus they can be accessed from both host code and device code
    cudaMallocManaged(&a, size_a);
    cudaMallocManaged(&b, size_b);
    cudaMallocManaged(&c, size_c);

    generate_data(a, n, m);
    generate_data(b, m, p);

    auto start = std::chrono::high_resolution_clock::now();

    // Launch grid and blocks. Each block is 3d and can have maximum of 1024 threads across all dimensions.
    // Each block has 32 threads in x-direction and 32 in y-direction.
    // Number of blocks in x direction = #columns in out matrix/#threads in x-direction
    // In CUDA, the ordering of dimensions are reversed i.e. a matrix of dim (N, M, P) will be represented as (P, M, N) in CUDA

    dim3 bd(32, 32, 1);
    dim3 gd(ceil(p/32.0), ceil(n/32.0), 1);

    // Launch kernel
    cuda_mul<<<gd, bd>>>(a, b, c, n, m, p);

    // Synchronize the host and device memory before accessing output
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;

    // Compare with multi-threaded matrix multiplication on CPU
    start = std::chrono::high_resolution_clock::now();
    d = (float*)malloc(size_c);
    mat_mul(a, b, d, n, m, p);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Standard Duration = " << duration.count() << " ms" << std::endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    free(d);
}
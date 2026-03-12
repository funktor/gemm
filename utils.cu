#include "cuda_gemm.h"
using namespace nvcuda;

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

void generate_data(float *x, const long n) {
    static std::random_device dev;
    static std::mt19937 rng(dev());

    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (auto i = 0; i < n; i++) x[i] = dist(rng);
}

bool compare_matrices(const float *x, const float *y, const long n) {
    for (auto i = 0; i < n; i++) {
        float v1 = x[i];
        float v2 = y[i];
        float diff  = fabs(v1 - v2);
        float relative_err = diff / v2;
        float eps = 1e-2;
        if ((relative_err >= eps)) {
            std::cout << v1 << " " << v2 << std::endl;
            return false;
        }
    }

    return true;
}

void transpose(const float *a, float *out, const int n, const int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out[j*n + i] = a[i*m + j];
        }
    }
}

void print_arr(const float *x, const long n) {
    for (auto i = 0; i < n; i++) {
        printf("%f, ", x[i]);
    }
    printf("\n");
}

__global__ 
void convertFp32ToFp16 (half *out, const float *in, const long n) {
    long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}
#include "cuda_gemm.h"
using namespace nvcuda;

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }

void gemm_fp16_cublas(
    const __half *a_fp16, 
    const __half *b_fp16, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    cublasHandle_t handle;
    cublasErrCheck(cublasCreate(&handle));
    // Use tensor cores
    cublasErrCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    cublasErrCheck(
        cublasGemmEx(
            handle, 
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            b_fp16, CUDA_R_16F, n,
            a_fp16, CUDA_R_16F, k,
            &beta,
            c_fp32, CUDA_R_32F, n,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
    );

    cublasDestroy(handle);
}

void gemm_fp32_cublas(
    const float *a_fp32, 
    const float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    cublasHandle_t handle;
    cublasErrCheck(cublasCreate(&handle));
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasErrCheck(
        cublasSgemm(
            handle, 
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            b_fp32, n,
            a_fp32, k,
            &beta,
            c_fp32, n
        )
    );

    cublasDestroy(handle);
}
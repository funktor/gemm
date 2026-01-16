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
            a_fp16, CUDA_R_16F, m,
            b_fp16, CUDA_R_16F, k,
            &beta,
            c_fp32, CUDA_R_32F, m,
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

__global__
void gemm_fp32_cuda(
    const float *a_fp32, 
    const float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float res = 0.0f;
        for (int i = 0; i < k; i++) res += a_fp32[row*k+i]*b_fp32[i*n+col];
        c_fp32[row*n+col] = alpha*res + beta*c_fp32[row*n+col];
    }
}


bool compare_matrices(const float *x, const float *y, const long n) {
    for (auto i = 0; i < n; i++) {
        float v1 = x[i];
        float v2 = y[i];
        float diff  = fabs(v1 - v2);
        float relative_err = diff / v2;
        float eps = 1e-4;
        if ((relative_err >= eps)) {
            std::cout << v1 << " " << v2 << std::endl;
            return false;
        }
    }

    return true;
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

int main(){
    int m = 1024;
    int n = 1024;
    int k = 1024;

    float *a_fp32;
    float *b_fp32;

    cudaErrCheck(cudaMallocManaged(&a_fp32, m * k * sizeof(float)));
    cudaErrCheck(cudaMallocManaged(&b_fp32, k * n * sizeof(float)));

    generate_data(a_fp32, m*k);
    generate_data(b_fp32, k*n);

    float cublasTime;
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;
    
    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));



    float *c_cpu_fp32;
    cudaErrCheck(cudaMallocManaged(&c_cpu_fp32, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_cpu_fp32[i] = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();
    gemm_cpu(a_fp32, b_fp32, c_cpu_fp32, 1.0, 0.0, m, n, k);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "CPU GEMM Duration = " << duration.count() << " ms" << std::endl;



    float *c_gpu_fp32_ccores;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_ccores, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32_ccores[i] = 0.0f;

    dim3 bd(32, 32, 1);
    dim3 gd((n+31)/32, (m+31)/32, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp32_cuda<<<gd, bd>>>(a_fp32, b_fp32, c_gpu_fp32_ccores, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUDA FP32 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32_ccores, m*n) << std::endl;




    float *c_gpu_fp32;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32[i] = 0.0f;

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp32_cublas(a_fp32, b_fp32, c_gpu_fp32, 1.0, 0.0, m, n, k);
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUBLAS FP32 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32, m*n) << std::endl;

    
    half *a_fp16;
    half *b_fp16;
    float *d_gpu_fp32;

    cudaErrCheck(cudaMallocManaged(&a_fp16, m * k * sizeof(half)));
    cudaErrCheck(cudaMallocManaged(&b_fp16, k * n * sizeof(half)));
    cudaErrCheck(cudaMallocManaged(&d_gpu_fp32, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) d_gpu_fp32[i] = 0.0f;

    convertFp32ToFp16 <<< (m * k + 255) / 256, 256 >>> (a_fp16, a_fp32, m * k);
    convertFp32ToFp16 <<< (k * n + 255) / 256, 256 >>> (b_fp16, b_fp32, k * n);
    cudaDeviceSynchronize();

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp16_cublas(a_fp16, b_fp16, d_gpu_fp32, 1.0, 0.0, m, n, k);
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));

    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUBLAS FP16 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, d_gpu_fp32, m*n) << std::endl;

    cudaErrCheck(cudaEventDestroy(startcublas));             
    cudaErrCheck(cudaEventDestroy(stopcublas));
    
    cudaErrCheck(cudaFree(a_fp32));
    cudaErrCheck(cudaFree(b_fp32));
    cudaErrCheck(cudaFree(c_cpu_fp32));
    cudaErrCheck(cudaFree(a_fp16));
    cudaErrCheck(cudaFree(b_fp16));
    cudaErrCheck(cudaFree(c_gpu_fp32));
    cudaErrCheck(cudaFree(d_gpu_fp32));

    cudaErrCheck(cudaDeviceReset());
    return 0;
}
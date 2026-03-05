#include "cuda_gemm.h"
using namespace nvcuda;

#define TILE_WIDTH 32
#define TILE_WIDTH_WMMA 64
#define COARSE_FACTOR 4
#define COARSE_FACTOR_2D 4
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define NUM_STAGES_ASYNC_PIPELINE 4


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

__global__ 
void normalize_fp32(const float *in, const float *in, const long n, const int d) {
    long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx]/float(d);
    }
}

__device__ __forceinline__ float atomicMaxF32(float *address, float val) {
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__
void max_row_reduction(float *inp, float *out, int n) {
    __shared__ float out_s[TILE_WIDTH];

    int idx = 2*blockIdx.x*blockDim.x + threadIdx.x;

    // update shared memory array
    if (idx + TILE_WIDTH < n) out_s[threadIdx.x] = inp[idx] + inp[idx + TILE_WIDTH];
    else if (idx < n) out_s[threadIdx.x] = inp[idx];
    else out_s[threadIdx.x] = 0.0f;
    __syncthreads();

    for (int stride = TILE_WIDTH/2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            if (threadIdx.x + stride < TILE_WIDTH) out_s[threadIdx.x] += out_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(&out[0], out_s[0]);
}

__global__
void gemm_fp32_cuda_tiled_2D_vectorize(
    float *a_fp32, 
    float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    __shared__ float Mds[TILE_WIDTH*TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = by*TILE_WIDTH*COARSE_FACTOR_2D + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR_2D + tx*4;

    float Pval[COARSE_FACTOR_2D*COARSE_FACTOR_2D*4];
    for (int r = 0; r < COARSE_FACTOR_2D*COARSE_FACTOR_2D*4; r++) Pval[r] = 0.0f;

    for (int ph = 0; ph < k; ph += TILE_WIDTH) {
        for (int r = 0; r < COARSE_FACTOR_2D; r++) {
            int row = row_start + r*TILE_WIDTH;
            reinterpret_cast<float4 *>(&Mds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&a_fp32[row*k + ph + tx*4])[0];

            for (int c = 0; c < COARSE_FACTOR_2D; c++) {
                int col = col_start + c*TILE_WIDTH;

                reinterpret_cast<float4 *>(&Nds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&b_fp32[(ph + ty)*n + col])[0];
                __syncthreads();

                for (int i = 0; i < TILE_WIDTH; i++) {
                    Pval[r*COARSE_FACTOR_2D*4 + 4*c + 0] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx*4+0];
                    Pval[r*COARSE_FACTOR_2D*4 + 4*c + 1] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx*4+1];
                    Pval[r*COARSE_FACTOR_2D*4 + 4*c + 2] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx*4+2];
                    Pval[r*COARSE_FACTOR_2D*4 + 4*c + 3] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx*4+3];
                }
                __syncthreads();
            }
        }
    }

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;
        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;

            c_fp32[row*n + col + 0] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 0] + beta*c_fp32[row*n + col + 0];
            c_fp32[row*n + col + 1] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 1] + beta*c_fp32[row*n + col + 1];
            c_fp32[row*n + col + 2] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 2] + beta*c_fp32[row*n + col + 2];
            c_fp32[row*n + col + 3] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 3] + beta*c_fp32[row*n + col + 3];
        }
    }
}

__global__
void gemm_fp32_cuda_tiled_2D_vectorize_b_trans(
    float *a_fp32, 
    float *b_fp32, 
    float *c_fp32, 
    float *c_max_row,
    float *c_sum_row,
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    __shared__ float Mds[TILE_WIDTH*TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH*TILE_WIDTH];
    __shared__ float block_max_values[blockDim.x];
    __shared__ float block_sum_values[blockDim.x];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = by*TILE_WIDTH*COARSE_FACTOR_2D + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR_2D + tx*4;

    float Pval[COARSE_FACTOR_2D*COARSE_FACTOR_2D*4];
    for (int r = 0; r < COARSE_FACTOR_2D*COARSE_FACTOR_2D*4; r++) Pval[r] = 0.0f;

    float max_values[COARSE_FACTOR_2D];
    float sum_values[COARSE_FACTOR_2D];

    for (int r = 0; r < COARSE_FACTOR_2D; r++) max_values[r] = -MAXFLOAT;
    for (int r = 0; r < COARSE_FACTOR_2D; r++) sum_values[r] = 0.0f;

    for (int ph = 0; ph < k; ph += TILE_WIDTH) {
        for (int r = 0; r < COARSE_FACTOR_2D; r++) {
            int row = row_start + r*TILE_WIDTH;
            reinterpret_cast<float4 *>(&Mds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&a_fp32[row*k + ph + tx*4])[0];

            for (int c = 0; c < COARSE_FACTOR_2D; c++) {
                int col = col_start + c*TILE_WIDTH;

                Nds[ty*TILE_WIDTH + tx*4 + 0] = b_fp32[(col+0)*k + ph + ty];
                Nds[ty*TILE_WIDTH + tx*4 + 1] = b_fp32[(col+1)*k + ph + ty];
                Nds[ty*TILE_WIDTH + tx*4 + 2] = b_fp32[(col+2)*k + ph + ty];
                Nds[ty*TILE_WIDTH + tx*4 + 3] = b_fp32[(col+3)*k + ph + ty];

                __syncthreads();

                for (int i = 0; i < TILE_WIDTH; i++) {
                    Pval[r*COARSE_FACTOR_2D*4 + 4*c + 0] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx*4+0];
                    Pval[r*COARSE_FACTOR_2D*4 + 4*c + 1] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx*4+1];
                    Pval[r*COARSE_FACTOR_2D*4 + 4*c + 2] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx*4+2];
                    Pval[r*COARSE_FACTOR_2D*4 + 4*c + 3] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx*4+3];
                }
                __syncthreads();
            }
        }
    }

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;

        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;

            c_fp32[row*n + col + 0] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 0] + beta*c_fp32[row*n + col + 0];
            c_fp32[row*n + col + 1] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 1] + beta*c_fp32[row*n + col + 1];
            c_fp32[row*n + col + 2] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 2] + beta*c_fp32[row*n + col + 2];
            c_fp32[row*n + col + 3] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 3] + beta*c_fp32[row*n + col + 3];

            max_values[r] = max(max_values[r], c_fp32[row*n + col + 0]);
            max_values[r] = max(max_values[r], c_fp32[row*n + col + 1]);
            max_values[r] = max(max_values[r], c_fp32[row*n + col + 2]);
            max_values[r] = max(max_values[r], c_fp32[row*n + col + 3]);
        }
    }

    __syncthreads();

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;

        block_max_values[threadIdx.x] = max_values[r];
        __syncthreads();

        for (int stride = blockDim.x/2; stride >= 1; stride /= 2) {
            if (threadIdx.x + stride < blockDim.x) {
                block_max_values[threadIdx.x] = max(block_max_values[threadIdx.x], block_max_values[threadIdx.x + stride]);
            }
            __syncthreads();
        }
                
        atomicMaxF32(&c_max_row[row], block_max_values[0]);
    }

    __syncthreads();

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;
        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;

            c_fp32[row*n + col + 0] = exp(c_fp32[row*n + col + 0]-c_max_row[row]);
            c_fp32[row*n + col + 1] = exp(c_fp32[row*n + col + 1]-c_max_row[row]);
            c_fp32[row*n + col + 2] = exp(c_fp32[row*n + col + 2]-c_max_row[row]);
            c_fp32[row*n + col + 3] = exp(c_fp32[row*n + col + 3]-c_max_row[row]);

            sum_values[r] += c_fp32[row*n + col + 0];
            sum_values[r] += c_fp32[row*n + col + 1];
            sum_values[r] += c_fp32[row*n + col + 2];
            sum_values[r] += c_fp32[row*n + col + 3];
        }
    }

    __syncthreads();

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;

        block_sum_values[threadIdx.x] = sum_values[r];
        __syncthreads();

        for (int stride = blockDim.x/2; stride >= 1; stride /= 2) {
            if (threadIdx.x + stride < blockDim.x) {
                block_sum_values[threadIdx.x] += block_sum_values[threadIdx.x + stride];
            }
            __syncthreads();
        }
                
        atomicAdd(&c_sum_row[row], block_sum_values[0]);
    }
}

void attention_fp32(
    const float *q_fp32, 
    const float *k_fp32, 
    const float *v_fp32, 
    float *out, 
    const int m, 
    const int k
)
{
    float *qk_t;
    float *c_max_row;
    float *c_sum_row;

    cudaErrCheck(cudaMallocManaged(&qk_t, m * m * sizeof(float)));
    cudaErrCheck(cudaMallocManaged(&c_max_row, m * sizeof(float)));
    cudaErrCheck(cudaMallocManaged(&c_sum_row, m * sizeof(float)));

    for (auto i = 0; i < m*m; i++) qk_t[i] = 0.0f;

    for (auto i = 0; i < m; i++) c_max_row[i] = -MAXFLOAT;
    for (auto i = 0; i < m; i++) c_sum_row[i] = 0.0f;

    dim3 bd1(8, 32, 1);
    dim3 gd1((n+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

    gemm_fp32_cuda_tiled_2D_vectorize_b_trans<<<gd1, bd1>>>(q_fp32, k_fp32, qk_t, c_max_row, c_sum_row, 1.0/float(k), 0.0, m, m, k);
    cudaDeviceSynchronize();


    cudaErrCheck(cudaFree(qk_t));
    cudaErrCheck(cudaFree(c_max_row));
    cudaErrCheck(cudaFree(c_sum_row));
}

int main(){
    int m = 4096;
    int k = 4096;

    float *q_fp32;
    float *k_fp32;
    float *v_fp32;
    float *k_fp32_t;

    cudaErrCheck(cudaMallocManaged(&a_fp32, m * k * sizeof(float)));
    cudaErrCheck(cudaMallocManaged(&k_fp32, m * k * sizeof(float)));
    cudaErrCheck(cudaMallocManaged(&v_fp32, m * k * sizeof(float)));
    cudaErrCheck(cudaMallocManaged(&k_fp32_t, k * m * sizeof(float)));

    generate_data(q_fp32, m*k);
    generate_data(k_fp32, m*k);
    generate_data(v_fp32, m*k);
    transpose(k_fp32, k_fp32_t, m, k);

    half *q_fp16;
    half *k_fp16;
    half *v_fp16;
    half *k_fp16_t;

    cudaErrCheck(cudaMallocManaged(&q_fp16, m * k * sizeof(half)));
    cudaErrCheck(cudaMallocManaged(&k_fp16, m * k * sizeof(half)));
    cudaErrCheck(cudaMallocManaged(&v_fp16, m * k * sizeof(half)));
    cudaErrCheck(cudaMallocManaged(&k_fp16_t, k * m * sizeof(half)));

    float cublasTime;
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;
    
    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));

    convertFp32ToFp16 <<< (m * k + 255) / 256, 256 >>> (q_fp16, q_fp32, m * k);
    convertFp32ToFp16 <<< (m * k + 255) / 256, 256 >>> (k_fp16, k_fp32, m * k);
    convertFp32ToFp16 <<< (m * k + 255) / 256, 256 >>> (v_fp16, v_fp32, m * k);
    convertFp32ToFp16 <<< (m * k + 255) / 256, 256 >>> (k_fp16_t, k_fp32_t, k * m);
    cudaDeviceSynchronize();



    float *c_cpu_fp32;
    cudaErrCheck(cudaMallocManaged(&c_cpu_fp32, m * k * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_cpu_fp32[i] = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();
    attention_cpu(q_fp32, k_fp32, v_fp32, c_cpu_fp32, m, k);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "CPU Attention Duration = " << duration.count() << " ms" << std::endl;



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
    cudaErrCheck(cudaFree(c_gpu_fp32_ccores));



    float *c_gpu_fp32_tiled;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32_tiled[i] = 0.0f;

    dim3 bd1(32, 32, 1);
    dim3 gd1((n+32*COARSE_FACTOR-1)/(32*COARSE_FACTOR), (m+31)/32, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp32_cuda_tiled<<<gd1, bd1>>>(a_fp32, b_fp32, c_gpu_fp32_tiled, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUDA TILED FP32 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32_tiled, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_fp32_tiled));



    float *c_gpu_fp32_tiled_2d;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled_2d, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32_tiled_2d[i] = 0.0f;

    dim3 bd2(32, 32, 1);
    dim3 gd2((n+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp32_cuda_tiled_2D<<<gd2, bd2>>>(a_fp32, b_fp32, c_gpu_fp32_tiled_2d, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUDA TILED 2D FP32 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32_tiled_2d, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d));



    float *c_gpu_fp32_tiled_2d_async;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled_2d_async, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32_tiled_2d_async[i] = 0.0f;

    dim3 bd21(8, 32, 1);
    dim3 gd21((n+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp32_cuda_tiled_2D_async<<<gd21, bd21>>>(a_fp32, b_fp32, c_gpu_fp32_tiled_2d_async, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUDA TILED 2D ASYNC FP32 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32_tiled_2d_async, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d_async));




    float *c_gpu_fp32_tiled_2d_async_warp_spl;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled_2d_async_warp_spl, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32_tiled_2d_async_warp_spl[i] = 0.0f;

    dim3 bd22(8, 36, 1);
    dim3 gd22((n+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp32_cuda_tiled_2D_async_warp_spl<<<gd22, bd22>>>(a_fp32, b_fp32, c_gpu_fp32_tiled_2d_async_warp_spl, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUDA TILED 2D ASYNC WARP SPL FP32 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32_tiled_2d_async_warp_spl, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d_async_warp_spl));


    float *c_gpu_fp32_tiled_2d_vec;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled_2d_vec, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32_tiled_2d_vec[i] = 0.0f;

    dim3 bd3(8, 32, 1);
    dim3 gd3((n+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp32_cuda_tiled_2D_vectorize<<<gd3, bd3>>>(a_fp32, b_fp32, c_gpu_fp32_tiled_2d_vec, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUDA TILED 2D VEC FP32 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32_tiled_2d_vec, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d_vec));



    float *c_gpu_fp32_tiled_2d_vec_b_trans;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled_2d_vec_b_trans, m * n * sizeof(float)));

    dim3 bd31(8, 32, 1);
    dim3 gd31((n+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp32_cuda_tiled_2D_vectorize_b_trans<<<gd31, bd31>>>(a_fp32, b_fp32_t, c_gpu_fp32_tiled_2d_vec_b_trans, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUDA TILED 2D VEC B TRANS FP32 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32_tiled_2d_vec_b_trans, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d_vec_b_trans));



    float *c_gpu_fp32_wmma;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_wmma, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32_wmma[i] = 0.0f;

    dim3 bd4(128, 4, 1);
    dim3 gd4((n+WMMA_N*128/32-1)/(WMMA_N*128/32), (m+WMMA_M*4-1)/(WMMA_M*4), 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_wmma<<<gd4, bd4>>>(a_fp16, b_fp16, c_gpu_fp32_wmma, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU WMMA FP16 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32_wmma, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_fp32_wmma));



    float *c_gpu_fp32_wmma_shmm;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_wmma_shmm, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32_wmma_shmm[i] = 0.0f;

    dim3 bd5(128, 4, 1);
    dim3 gd5((n+WMMA_N*128/32-1)/(WMMA_N*128/32), (m+WMMA_M*4-1)/(WMMA_M*4), 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_wmma_shmm<<<gd5, bd5>>>(a_fp16, b_fp16, c_gpu_fp32_wmma_shmm, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU WMMA SHMM FP16 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32_wmma_shmm, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_fp32_wmma_shmm));



    float *c_gpu_mma_sync_fp16;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16[i] = 0.0f;

    dim3 bd6(128, 4, 1);
    dim3 gd6((n+TILE_WIDTH_WMMA-1)/TILE_WIDTH_WMMA, (m+TILE_WIDTH_WMMA-1)/TILE_WIDTH_WMMA, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16<<<gd6, bd6>>>(a_fp16, b_fp16, c_gpu_mma_sync_fp16, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16));




    float *c_gpu_mma_sync_fp16_2d_tiled;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16_2d_tiled, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16_2d_tiled[i] = 0.0f;

    dim3 bd7(64, 2, 1);
    dim3 gd7((n+256-1)/256, (m+256-1)/256, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16_2d_tiled<<<gd7, bd7>>>(a_fp16, b_fp16, c_gpu_mma_sync_fp16_2d_tiled, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 2D TILED GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16_2d_tiled, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16_2d_tiled));


    float *c_gpu_mma_sync_fp16_2d_tiled_b_trans;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16_2d_tiled_b_trans, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16_2d_tiled_b_trans[i] = 0.0f;

    dim3 bd71(64, 2, 1);
    dim3 gd71((n+256-1)/256, (m+256-1)/256, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16_2d_tiled_b_trans<<<gd71, bd71>>>(a_fp16, b_fp16_t, c_gpu_mma_sync_fp16_2d_tiled_b_trans, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 2D B TRANS TILED GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16_2d_tiled_b_trans, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16_2d_tiled_b_trans));




    float *c_gpu_mma_sync_fp16_2d_tiled_swz;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16_2d_tiled_swz, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16_2d_tiled_swz[i] = 0.0f;

    dim3 bd8(64, 2, 1);
    dim3 gd8((n+256-1)/256, (m+256-1)/256, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16_2d_tiled_swizzled<<<gd8, bd8>>>(a_fp16, b_fp16, c_gpu_mma_sync_fp16_2d_tiled_swz, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 2D TILED SWIZZLED GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16_2d_tiled_swz, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16_2d_tiled_swz));




    float *c_gpu_mma_sync_fp16_2d_tiled_swz_async;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16_2d_tiled_swz_async, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16_2d_tiled_swz_async[i] = 0.0f;

    dim3 bd81(64, 2, 1);
    dim3 gd81((n+256-1)/256, (m+256-1)/256, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16_2d_tiled_swizzled_async<<<gd81, bd81>>>(a_fp16, b_fp16, c_gpu_mma_sync_fp16_2d_tiled_swz_async, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 2D TILED SWIZZLED ASYNC GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16_2d_tiled_swz_async, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16_2d_tiled_swz_async));



    float *c_gpu_mma_sync_fp16_2d_tiled_swz_exp;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16_2d_tiled_swz_exp, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16_2d_tiled_swz_exp[i] = 0.0f;

    dim3 bd9(64, 2, 1);
    dim3 gd9((n+256-1)/256, (m+256-1)/256, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16_2d_tiled_swizzled_explicit<<<gd9, bd9>>>(a_fp16, b_fp16, c_gpu_mma_sync_fp16_2d_tiled_swz_exp, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 2D TILED SWIZZLED EXPLICIT GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16_2d_tiled_swz_exp, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16_2d_tiled_swz_exp));



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
    cudaErrCheck(cudaFree(c_gpu_fp32));


    
    float *d_gpu_fp32;
    cudaErrCheck(cudaMallocManaged(&d_gpu_fp32, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) d_gpu_fp32[i] = 0.0f;

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp16_cublas(a_fp16, b_fp16, d_gpu_fp32, 1.0, 0.0, m, n, k);
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));

    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUBLAS FP16 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, d_gpu_fp32, m*n) << std::endl;
    cudaErrCheck(cudaFree(d_gpu_fp32));


    cudaErrCheck(cudaEventDestroy(startcublas));             
    cudaErrCheck(cudaEventDestroy(stopcublas));
    
    cudaErrCheck(cudaFree(a_fp32));
    cudaErrCheck(cudaFree(b_fp32));
    cudaErrCheck(cudaFree(c_cpu_fp32));
    cudaErrCheck(cudaFree(a_fp16));
    cudaErrCheck(cudaFree(b_fp16));
    cudaErrCheck(cudaDeviceReset());
    return 0;
}
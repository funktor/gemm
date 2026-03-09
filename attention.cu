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

__global__
void gemm_fp32_cuda_tiled_2D_vectorize(
    float *a_fp32, 
    float *b_fp32, 
    float *c_fp32, 
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

    float Pval[COARSE_FACTOR_2D*COARSE_FACTOR_2D*32*4];
    for (int r = 0; r < COARSE_FACTOR_2D*COARSE_FACTOR_2D*32*4; r++) Pval[r] = 0.0f;

    int num_tiles = 32;

    float max_val_tiles[32*COARSE_FACTOR_2D];
    float sum_val_tiles[32*COARSE_FACTOR_2D];

    int h = 0;

    for (int ph = 0; ph < k; ph += TILE_WIDTH) {
        for (int r = 0; r < COARSE_FACTOR_2D; r++) {
            int row = row_start + r*TILE_WIDTH;
            reinterpret_cast<float4 *>(&Mds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&a_fp32[row*k + ph + tx*4])[0];

            float max_val_tile = -MAXFLOAT;
            float sum_val_tile = 0.0f;

            for (int i = 0; i < TILE_WIDTH; i++) max_val_tile = max(max_val_tile, Mds[ty*TILE_WIDTH+i]);
            for (int i = 0; i < TILE_WIDTH; i++) sum_val_tile += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile);

            max_val_tiles[r*32 + h] = max_val_tile;
            sum_val_tiles[r*32 + h] = sum_val_tile;

            for (int c = 0; c < COARSE_FACTOR_2D; c++) {
                int col = col_start + c*TILE_WIDTH;
                reinterpret_cast<float4 *>(&Nds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&b_fp32[(ph + ty)*n + col])[0];
                __syncthreads();

                for (int i = 0; i < TILE_WIDTH; i++) {
                    Pval[4*(r*COARSE_FACTOR_2D*32 + 32*c + h) + 0] += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile)*Nds[i*TILE_WIDTH+tx*4+0];
                    Pval[4*(r*COARSE_FACTOR_2D*32 + 32*c + h) + 1] += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile)*Nds[i*TILE_WIDTH+tx*4+1];
                    Pval[4*(r*COARSE_FACTOR_2D*32 + 32*c + h) + 2] += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile)*Nds[i*TILE_WIDTH+tx*4+2];
                    Pval[4*(r*COARSE_FACTOR_2D*32 + 32*c + h) + 3] += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile)*Nds[i*TILE_WIDTH+tx*4+3];
                }
                __syncthreads();
            }
        }

        h += 1;
    }

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;

        float max_val = -MAXFLOAT;
        for (int h = 0; h < num_tiles; h++) max_val = max(max_val, max_val_tiles[r*32 + h]);

        float sum_val = 0.0f;
        for (int h = 0; h < num_tiles; h++) sum_val += sum_val_tiles[r*32 + h] * exp(max_val_tiles[r*32 + h]-max_val);

        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;

            for (int h = 0; h < num_tiles; h++) {
                c_fp32[row*n + col + 0] += Pval[4*(r*COARSE_FACTOR_2D*32 + 32*c + h) + 0] * exp(max_val_tiles[r*32 + h]-max_val)/sum_val;
                c_fp32[row*n + col + 1] += Pval[4*(r*COARSE_FACTOR_2D*32 + 32*c + h) + 1] * exp(max_val_tiles[r*32 + h]-max_val)/sum_val;
                c_fp32[row*n + col + 2] += Pval[4*(r*COARSE_FACTOR_2D*32 + 32*c + h) + 2] * exp(max_val_tiles[r*32 + h]-max_val)/sum_val;
                c_fp32[row*n + col + 3] += Pval[4*(r*COARSE_FACTOR_2D*32 + 32*c + h) + 3] * exp(max_val_tiles[r*32 + h]-max_val)/sum_val;
            }
        }
    }




    // for (int r = 0; r < COARSE_FACTOR_2D; r++) {
    //     int row = row_start + r*TILE_WIDTH;

    //     for (int c = 0; c < COARSE_FACTOR_2D; c++) {
    //         int col = col_start + c*TILE_WIDTH;

    //         int num_tiles = 32;

    //         float max_val_tiles[32];
    //         float sum_val_tiles[32];
    //         float pval_tiles[4*32];
    //         for (int j = 0; j < 4*32; j++) pval_tiles[j] = 0.0f;
            
    //         float max_val = -MAXFLOAT;

    //         int h = 0;

    //         for (int ph = 0; ph < k; ph += TILE_WIDTH) {
    //             reinterpret_cast<float4 *>(&Mds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&a_fp32[row*k + ph + tx*4])[0];
    //             reinterpret_cast<float4 *>(&Nds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&b_fp32[(ph + ty)*n + col])[0];
    //             __syncthreads();

    //             float max_val_tile = -MAXFLOAT;
    //             float sum_val_tile = 0.0f;

    //             for (int i = 0; i < TILE_WIDTH; i++) max_val_tile = max(max_val_tile, Mds[ty*TILE_WIDTH+i]);
    //             for (int i = 0; i < TILE_WIDTH; i++) sum_val_tile += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile);

    //             max_val_tiles[h] = max_val_tile;
    //             sum_val_tiles[h] = sum_val_tile;

    //             max_val = max(max_val, max_val_tile);

    //             for (int i = 0; i < TILE_WIDTH; i++) {
    //                 pval_tiles[4*h + 0] += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile)*Nds[i*TILE_WIDTH+tx*4+0];
    //                 pval_tiles[4*h + 1] += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile)*Nds[i*TILE_WIDTH+tx*4+1];
    //                 pval_tiles[4*h + 2] += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile)*Nds[i*TILE_WIDTH+tx*4+2];
    //                 pval_tiles[4*h + 3] += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile)*Nds[i*TILE_WIDTH+tx*4+3];
    //             }
    //             __syncthreads();

    //             h += 1;
    //         }

    //         float sum_val = 0.0f;
    //         for (int h = 0; h < num_tiles; h++) sum_val += sum_val_tiles[h] * exp(max_val_tiles[h]-max_val);

    //         for (int h = 0; h < num_tiles; h++) {
    //             c_fp32[row*n + col + 0] += pval_tiles[4*h + 0] * exp(max_val_tiles[h]-max_val)/sum_val;
    //             c_fp32[row*n + col + 1] += pval_tiles[4*h + 1] * exp(max_val_tiles[h]-max_val)/sum_val;
    //             c_fp32[row*n + col + 2] += pval_tiles[4*h + 2] * exp(max_val_tiles[h]-max_val)/sum_val;
    //             c_fp32[row*n + col + 3] += pval_tiles[4*h + 3] * exp(max_val_tiles[h]-max_val)/sum_val;
    //         }
    //     }
    // }
}

__global__
void gemm_fp32_cuda_tiled_2D_vectorize_b_trans(
    float *a_fp32, 
    float *b_fp32, 
    float *c_fp32, 
    const float alpha,
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

            c_fp32[row*n + col + 0] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 0];
            c_fp32[row*n + col + 1] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 1];
            c_fp32[row*n + col + 2] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 2];
            c_fp32[row*n + col + 3] = alpha*Pval[r*COARSE_FACTOR_2D*4 + 4*c + 3];
        }
    }
}

__global__
void compute_sum_rows( 
    float *c_fp32, 
    float *c_max_row,
    float *c_sum_row,
    const int m, 
    const int n, 
    const int k
) {
    __shared__ float block_sum_values[32][8];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = by*TILE_WIDTH*COARSE_FACTOR_2D + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR_2D + tx*4;

    float sum_values[COARSE_FACTOR_2D];
    for (int r = 0; r < COARSE_FACTOR_2D; r++) sum_values[r] = 0.0f;

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
        block_sum_values[threadIdx.y][threadIdx.x] = sum_values[r];
        __syncthreads();

        for (int stride = blockDim.x/2; stride >= 1; stride /= 2) {
            if (threadIdx.x < stride) {
                block_sum_values[threadIdx.y][threadIdx.x] += block_sum_values[threadIdx.y][threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) atomicAdd(&c_sum_row[row], block_sum_values[threadIdx.y][0]);
    }
}


void attention_gpu(
    float *q_fp32, 
    float *k_fp32, 
    float *v_fp32, 
    float *out, 
    const int m, 
    const int k
)
{
    float *qk_t;
    cudaErrCheck(cudaMallocManaged(&qk_t, m * m * sizeof(float)));

    for (auto i = 0; i < m*m; i++) qk_t[i] = 0.0f;

    dim3 bd1(8, 32, 1);
    dim3 gd1((m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

    gemm_fp32_cuda_tiled_2D_vectorize_b_trans<<<gd1, bd1>>>(q_fp32, k_fp32, qk_t, 1.0/sqrt(k), m, m, k);
    cudaDeviceSynchronize();

    gemm_fp32_cuda_tiled_2D_vectorize<<<gd1, bd1>>>(qk_t, v_fp32, out, m, k, m);
    cudaDeviceSynchronize();

    cudaErrCheck(cudaFree(qk_t));
}

int main(){
    int m = 1024;
    int k = 1024;

    float *q_fp32;
    float *k_fp32;
    float *v_fp32;
    float *k_fp32_t;

    cudaErrCheck(cudaMallocManaged(&q_fp32, m * k * sizeof(float)));
    cudaErrCheck(cudaMallocManaged(&k_fp32, m * k * sizeof(float)));
    cudaErrCheck(cudaMallocManaged(&v_fp32, m * k * sizeof(float)));
    cudaErrCheck(cudaMallocManaged(&k_fp32_t, k * m * sizeof(float)));

    generate_data(q_fp32, m*k);
    generate_data(k_fp32, m*k);
    generate_data(v_fp32, m*k);
    transpose(k_fp32, k_fp32_t, m, k);

    float cublasTime;
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;
    
    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));

    float *c_cpu_fp32;
    cudaErrCheck(cudaMallocManaged(&c_cpu_fp32, m * k * sizeof(float)));

    for (auto i = 0; i < m*k; i++) c_cpu_fp32[i] = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();
    attention_cpu(q_fp32, k_fp32_t, v_fp32, c_cpu_fp32, m, k);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "CPU Attention Duration = " << duration.count() << " ms" << std::endl;



    float *c_gpu_fp32;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32, m * k * sizeof(float)));

    for (auto i = 0; i < m*k; i++) c_gpu_fp32[i] = 0.0f;

    cudaErrCheck(cudaEventRecord(startcublas));
    attention_gpu(q_fp32, k_fp32_t, v_fp32, c_gpu_fp32, m, k);
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU Attention Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32, m*k) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_fp32));


    cudaErrCheck(cudaEventDestroy(startcublas));             
    cudaErrCheck(cudaEventDestroy(stopcublas));
    
    cudaErrCheck(cudaFree(q_fp32));
    cudaErrCheck(cudaFree(k_fp32));
    cudaErrCheck(cudaFree(v_fp32));
    cudaErrCheck(cudaFree(k_fp32_t));
    cudaErrCheck(cudaFree(c_cpu_fp32));
    cudaErrCheck(cudaDeviceReset());
    return 0;
}
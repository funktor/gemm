#include "cuda_gemm.h"
using namespace nvcuda;

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }


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

__global__
void attn_sftmax_dot_v(
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

    int num_tiles = (k + TILE_WIDTH - 1)/TILE_WIDTH;

    float Pval[COARSE_FACTOR_2D*COARSE_FACTOR_2D*MAX_NUM_TILES*4];
    for (int r = 0; r < COARSE_FACTOR_2D*COARSE_FACTOR_2D*num_tiles*4; r++) Pval[r] = 0.0f;

    float max_val_tiles[MAX_NUM_TILES*COARSE_FACTOR_2D];
    float sum_val_tiles[MAX_NUM_TILES*COARSE_FACTOR_2D];

    int h = 0;

    for (int ph = 0; ph < k; ph += TILE_WIDTH) {
        for (int r = 0; r < COARSE_FACTOR_2D; r++) {
            int row = row_start + r*TILE_WIDTH;
            reinterpret_cast<float4 *>(&Mds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&a_fp32[row*k + ph + tx*4])[0];

            float max_val_tile = -MAXFLOAT;
            float sum_val_tile = 0.0f;

            for (int i = 0; i < TILE_WIDTH; i++) max_val_tile = max(max_val_tile, Mds[ty*TILE_WIDTH+i]);
            for (int i = 0; i < TILE_WIDTH; i++) sum_val_tile += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile);

            max_val_tiles[r*MAX_NUM_TILES + h] = max_val_tile;
            sum_val_tiles[r*MAX_NUM_TILES + h] = sum_val_tile;

            for (int c = 0; c < COARSE_FACTOR_2D; c++) {
                int col = col_start + c*TILE_WIDTH;
                reinterpret_cast<float4 *>(&Nds[ty*TILE_WIDTH + tx*4])[0] = reinterpret_cast<float4 *>(&b_fp32[(ph + ty)*n + col])[0];
                __syncthreads();

                for (int i = 0; i < TILE_WIDTH; i++) {
                    Pval[4*(r*COARSE_FACTOR_2D*MAX_NUM_TILES + MAX_NUM_TILES*c + h) + 0] += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile)*Nds[i*TILE_WIDTH+tx*4+0];
                    Pval[4*(r*COARSE_FACTOR_2D*MAX_NUM_TILES + MAX_NUM_TILES*c + h) + 1] += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile)*Nds[i*TILE_WIDTH+tx*4+1];
                    Pval[4*(r*COARSE_FACTOR_2D*MAX_NUM_TILES + MAX_NUM_TILES*c + h) + 2] += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile)*Nds[i*TILE_WIDTH+tx*4+2];
                    Pval[4*(r*COARSE_FACTOR_2D*MAX_NUM_TILES + MAX_NUM_TILES*c + h) + 3] += exp(Mds[ty*TILE_WIDTH+i]-max_val_tile)*Nds[i*TILE_WIDTH+tx*4+3];
                }
                __syncthreads();
            }
        }

        h += 1;
    }

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;

        float max_val = -MAXFLOAT;
        for (int h = 0; h < num_tiles; h++) max_val = max(max_val, max_val_tiles[r*MAX_NUM_TILES + h]);

        float sum_val = 0.0f;
        for (int h = 0; h < num_tiles; h++) sum_val += sum_val_tiles[r*MAX_NUM_TILES + h] * exp(max_val_tiles[r*MAX_NUM_TILES + h]-max_val);

        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;

            for (int h = 0; h < num_tiles; h++) {
                c_fp32[row*n + col + 0] += Pval[4*(r*COARSE_FACTOR_2D*MAX_NUM_TILES + MAX_NUM_TILES*c + h) + 0] * exp(max_val_tiles[r*MAX_NUM_TILES + h]-max_val)/sum_val;
                c_fp32[row*n + col + 1] += Pval[4*(r*COARSE_FACTOR_2D*MAX_NUM_TILES + MAX_NUM_TILES*c + h) + 1] * exp(max_val_tiles[r*MAX_NUM_TILES + h]-max_val)/sum_val;
                c_fp32[row*n + col + 2] += Pval[4*(r*COARSE_FACTOR_2D*MAX_NUM_TILES + MAX_NUM_TILES*c + h) + 2] * exp(max_val_tiles[r*MAX_NUM_TILES + h]-max_val)/sum_val;
                c_fp32[row*n + col + 3] += Pval[4*(r*COARSE_FACTOR_2D*MAX_NUM_TILES + MAX_NUM_TILES*c + h) + 3] * exp(max_val_tiles[r*MAX_NUM_TILES + h]-max_val)/sum_val;
            }
        }
    }
}

__global__
void attn_qk_t(
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


void attention_gpu(
    float *q_fp32, 
    float *k_fp32, 
    float *v_fp32, 
    float *out, 
    const int m, 
    const int k
) {
    float *qk_t;
    cudaErrCheck(cudaMallocManaged(&qk_t, m * m * sizeof(float)));

    for (auto i = 0; i < m*m; i++) qk_t[i] = 0.0f;

    dim3 bd1(8, 32, 1);
    dim3 gd1((m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

    attn_qk_t<<<gd1, bd1>>>(q_fp32, k_fp32, qk_t, 1.0/sqrt(k), m, m, k);
    cudaDeviceSynchronize();

    dim3 bd2(8, 32, 1);
    dim3 gd2((k+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

    attn_sftmax_dot_v<<<gd2, bd2>>>(qk_t, v_fp32, out, m, k, m);
    cudaDeviceSynchronize();

    cudaErrCheck(cudaFree(qk_t));
}
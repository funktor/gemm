#include "cuda_gemm.h"
using namespace nvcuda;

__global__
void gemm_fp32_cuda_tiled(
    const float *a_fp32, 
    const float *b_fp32, 
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

    int row = by*TILE_WIDTH + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR + tx;

    float Pval[COARSE_FACTOR];
    for (int r = 0; r < COARSE_FACTOR; r++) Pval[r] = 0.0f;

    for (int ph = 0; ph < k; ph += TILE_WIDTH) {
        if (row < m && (ph + tx) < k) Mds[ty*TILE_WIDTH+tx] = a_fp32[row*k + ph + tx];
        else Mds[ty*TILE_WIDTH+tx] = 0.0f;

        for (int r = 0; r < COARSE_FACTOR; r++) {
            int col = col_start + r*TILE_WIDTH;

            if ((ph + ty) < k && col < n) Nds[ty*TILE_WIDTH+tx] = b_fp32[(ph + ty)*n + col];
            else Nds[ty*TILE_WIDTH+tx] = 0.0f;
            __syncthreads();

            for (int i = 0; i < TILE_WIDTH; i++) Pval[r] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx];
            __syncthreads();
        }
    }

    for (int r = 0; r < COARSE_FACTOR; r++) {
        int col = col_start + r*TILE_WIDTH;
        if (row < m && col < n) c_fp32[row*n+col] = alpha*Pval[r] + beta*c_fp32[row*n+col];
    }
}


__global__
void gemm_fp32_cuda_tiled_2D(
    const float *a_fp32, 
    const float *b_fp32, 
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
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR_2D + tx;

    float Pval[COARSE_FACTOR_2D*COARSE_FACTOR_2D];
    for (int r = 0; r < COARSE_FACTOR_2D*COARSE_FACTOR_2D; r++) Pval[r] = 0.0f;

    for (int ph = 0; ph < k; ph += TILE_WIDTH) {
        for (int r = 0; r < COARSE_FACTOR_2D; r++) {
            int row = row_start + r*TILE_WIDTH;

            if (row < m && ph + tx < k) Mds[ty*TILE_WIDTH+tx] = a_fp32[row*k + ph + tx];
            else Mds[ty*TILE_WIDTH+tx] = 0.0f;

            for (int c = 0; c < COARSE_FACTOR_2D; c++) {
                int col = col_start + c*TILE_WIDTH;

                if (ph + ty < k && col < n) Nds[ty*TILE_WIDTH+tx] = b_fp32[(ph + ty)*n + col];
                else Nds[ty*TILE_WIDTH+tx] = 0.0f;
                __syncthreads();

                for (int i = 0; i < TILE_WIDTH; i++) Pval[r*COARSE_FACTOR_2D + c] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx];
                __syncthreads();
            }
        }
    }

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;
        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;
            if (row < m && col < n) c_fp32[row*n+col] = alpha * Pval[r*COARSE_FACTOR_2D + c] + beta * c_fp32[row*n+col];
        }
    }
}
#include "cuda_gemm.h"
using namespace nvcuda;

__global__ 
void gemm_wmma(
    const half *a, 
    const half *b, 
    float *c, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    int lda = k;
    int ldb = n;
    int ldc = n;

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int i = 0; i < k; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * WMMA_N;

        if (aRow < m && aCol < k && bRow < k && bCol < n) {
            wmma::load_matrix_sync(a_frag, a + aRow * lda + aCol, lda);
            wmma::load_matrix_sync(b_frag, b + bRow * ldb + bCol, ldb);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < m && cCol < n) {
        wmma::load_matrix_sync(c_frag, c + cRow * ldc + cCol, ldc, wmma::mem_row_major);

        #pragma unroll
        for(int i=0; i < c_frag.num_elements; i++) c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];

        wmma::store_matrix_sync(c + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
    }
}


__global__ 
void gemm_wmma_shmm(
    const half *a, 
    const half *b, 
    float *c, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    __shared__ half Mds[TILE_WIDTH_WMMA*TILE_WIDTH_WMMA];
    __shared__ half Nds[TILE_WIDTH_WMMA*TILE_WIDTH_WMMA];

    int lda = k;
    int ldb = n;
    int ldc = n;

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = 0; i < k; i += TILE_WIDTH_WMMA) {
        int a_block_row = blockIdx.y * TILE_WIDTH_WMMA;
        int a_block_col = i;

        for (int j = idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
            Mds[j] = a[(a_block_row + j/TILE_WIDTH_WMMA) * k + a_block_col + (j % TILE_WIDTH_WMMA)];
        }

        int b_block_row = i;
        int b_block_col = blockIdx.x * TILE_WIDTH_WMMA;

        for (int j = idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
            Nds[j] = b[(b_block_row + j/TILE_WIDTH_WMMA) * n + b_block_col + (j % TILE_WIDTH_WMMA)];
        }

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < TILE_WIDTH_WMMA; j += WMMA_K) {
            int a_warp_row = threadIdx.y * WMMA_M;
            int a_warp_col = j;

            int b_warp_row = j;
            int b_warp_col = (threadIdx.x / 32) * WMMA_N;

            if (a_warp_row < TILE_WIDTH_WMMA && a_warp_col < TILE_WIDTH_WMMA && b_warp_row < TILE_WIDTH_WMMA && b_warp_col < TILE_WIDTH_WMMA) {
                wmma::load_matrix_sync(a_frag, Mds + a_warp_row * TILE_WIDTH_WMMA + a_warp_col, TILE_WIDTH_WMMA);
                wmma::load_matrix_sync(b_frag, Nds + b_warp_row * TILE_WIDTH_WMMA + b_warp_col, TILE_WIDTH_WMMA);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }

        __syncthreads();
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < m && cCol < n) {
        wmma::load_matrix_sync(c_frag, c + cRow * ldc + cCol, ldc, wmma::mem_row_major);

        #pragma unroll
        for(int i=0; i < c_frag.num_elements; i++) c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];

        wmma::store_matrix_sync(c + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
    }
}
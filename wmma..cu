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
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    __shared__ half Mds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH_WMMA*TILE_WIDTH_WMMA];
    __shared__ half Nds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH_WMMA*TILE_WIDTH_WMMA];

    int ldc = n;

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    int a_block_row = by * TILE_WIDTH_WMMA;
    int b_block_col = bx * TILE_WIDTH_WMMA;

    for (int s = 0; s < NUM_STAGES_ASYNC_PIPELINE; s++) {
        int h = s*TILE_WIDTH_WMMA;

        pipeline.producer_acquire();
        #pragma unroll
        for (int j = idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
            cuda::memcpy_async(Mds[s] + j, a + (a_block_row + j/TILE_WIDTH_WMMA)*k + h + (j % TILE_WIDTH_WMMA), cuda::aligned_size_t<2>(sizeof(half)), pipeline);
            cuda::memcpy_async(Nds[s] + j, b + (h + j/TILE_WIDTH_WMMA)*n + b_block_col + (j % TILE_WIDTH_WMMA), cuda::aligned_size_t<2>(sizeof(half)), pipeline);
        }
        pipeline.producer_commit();
    }

    int s = NUM_STAGES_ASYNC_PIPELINE;

    for (int i = 0; i < k; i += TILE_WIDTH_WMMA) {
        int stage = s % NUM_STAGES_ASYNC_PIPELINE;

        constexpr size_t pending_batches = NUM_STAGES_ASYNC_PIPELINE - 1;
        cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline);
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < TILE_WIDTH_WMMA; j += WMMA_K) {
            int a_warp_row = threadIdx.y * WMMA_M;
            int a_warp_col = j;

            int b_warp_row = j;
            int b_warp_col = (threadIdx.x / 32) * WMMA_N;

            if (a_warp_row < TILE_WIDTH_WMMA && a_warp_col < TILE_WIDTH_WMMA && b_warp_row < TILE_WIDTH_WMMA && b_warp_col < TILE_WIDTH_WMMA) {
                wmma::load_matrix_sync(a_frag, Mds[stage] + a_warp_row * TILE_WIDTH_WMMA + a_warp_col, TILE_WIDTH_WMMA);
                wmma::load_matrix_sync(b_frag, Nds[stage] + b_warp_row * TILE_WIDTH_WMMA + b_warp_col, TILE_WIDTH_WMMA);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }

        pipeline.consumer_release();
        __syncthreads();

        pipeline.producer_acquire();
        int h = s*TILE_WIDTH_WMMA;
        if (h < k) {
            #pragma unroll
            for (int j = idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
                cuda::memcpy_async(Mds[stage] + j, a + (a_block_row + j/TILE_WIDTH_WMMA)*k + h + (j % TILE_WIDTH_WMMA), cuda::aligned_size_t<2>(sizeof(half)), pipeline);
                cuda::memcpy_async(Nds[stage] + j, b + (h + j/TILE_WIDTH_WMMA)*n + b_block_col + (j % TILE_WIDTH_WMMA), cuda::aligned_size_t<2>(sizeof(half)), pipeline);
            }
        }
        pipeline.producer_commit();

        s += 1;
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
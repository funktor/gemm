#include "cuda_gemm.h"
using namespace nvcuda;

__global__
void gemm_fp32_cuda_tiled_2D_async(
    const float *a_fp32, 
    const float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    __shared__ alignas(16) float Mds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH*TILE_WIDTH];
    __shared__ alignas(16) float Nds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = by*TILE_WIDTH*COARSE_FACTOR_2D + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR_2D + tx*4;

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;
        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;
            
            for (int s = 0; s < NUM_STAGES_ASYNC_PIPELINE; s++) {
                pipeline.producer_acquire();
                cuda::memcpy_async(Mds[s] + ty*TILE_WIDTH + tx*4, a_fp32 + row*k + s*TILE_WIDTH + tx*4, cuda::aligned_size_t<4>(sizeof(float)*4), pipeline);
                cuda::memcpy_async(Nds[s] + ty*TILE_WIDTH + tx*4, b_fp32 + (s*TILE_WIDTH + ty)*n + col, cuda::aligned_size_t<4>(sizeof(float)*4), pipeline);
                pipeline.producer_commit();
            }

            int s = NUM_STAGES_ASYNC_PIPELINE;
            float res[4] = {0.0f};

            for (int ph = 0; ph < k; ph += TILE_WIDTH) {
                int stage = s % NUM_STAGES_ASYNC_PIPELINE;

                constexpr size_t pending_batches = NUM_STAGES_ASYNC_PIPELINE - 1;
                cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline);
                __syncthreads();

                for (int i = 0; i < TILE_WIDTH; i++) {
                    res[0] += Mds[stage][ty*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+0];
                    res[1] += Mds[stage][ty*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+1];
                    res[2] += Mds[stage][ty*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+2];
                    res[3] += Mds[stage][ty*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+3];
                }

                pipeline.consumer_release();
                __syncthreads();

                pipeline.producer_acquire();
                if (s*TILE_WIDTH < k) {
                    cuda::memcpy_async(Mds[stage] + ty*TILE_WIDTH + tx*4, a_fp32 + row*k + s*TILE_WIDTH + tx*4, cuda::aligned_size_t<4>(sizeof(float)*4), pipeline);
                    cuda::memcpy_async(Nds[stage] + ty*TILE_WIDTH + tx*4, b_fp32 + (s*TILE_WIDTH + ty)*n + col, cuda::aligned_size_t<4>(sizeof(float)*4), pipeline);
                }
                pipeline.producer_commit();

                s += 1;
            }

            c_fp32[row*n+col+0] = alpha * res[0] + beta * c_fp32[row*n+col+0];
            c_fp32[row*n+col+1] = alpha * res[1] + beta * c_fp32[row*n+col+1];
            c_fp32[row*n+col+2] = alpha * res[2] + beta * c_fp32[row*n+col+2];
            c_fp32[row*n+col+3] = alpha * res[3] + beta * c_fp32[row*n+col+3];
        }
    }
}

__global__
void gemm_fp32_cuda_tiled_2D_async_block(
    const float *a_fp32, 
    const float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, NUM_STAGES_ASYNC_PIPELINE> shared_state;
    cuda::pipeline<cuda::thread_scope_block> pipeline = cuda::make_pipeline(block, &shared_state);

    __shared__ alignas(16) float Mds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH*TILE_WIDTH];
    __shared__ alignas(16) float Nds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = by*TILE_WIDTH*COARSE_FACTOR_2D + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR_2D + tx*4;

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;
        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;
            
            for (int s = 0; s < NUM_STAGES_ASYNC_PIPELINE; s++) {
                pipeline.producer_acquire();
                cuda::memcpy_async(Mds[s] + ty*TILE_WIDTH + tx*4, a_fp32 + row*k + s*TILE_WIDTH + tx*4, cuda::aligned_size_t<4>(sizeof(float)*4), pipeline);
                cuda::memcpy_async(Nds[s] + ty*TILE_WIDTH + tx*4, b_fp32 + (s*TILE_WIDTH + ty)*n + col, cuda::aligned_size_t<4>(sizeof(float)*4), pipeline);
                pipeline.producer_commit();
            }

            int s = NUM_STAGES_ASYNC_PIPELINE;
            float res[4] = {0.0f};

            for (int ph = 0; ph < k; ph += TILE_WIDTH) {
                int stage = s % NUM_STAGES_ASYNC_PIPELINE;

                pipeline.consumer_wait();
                for (int i = 0; i < TILE_WIDTH; i++) {
                    res[0] += Mds[stage][ty*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+0];
                    res[1] += Mds[stage][ty*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+1];
                    res[2] += Mds[stage][ty*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+2];
                    res[3] += Mds[stage][ty*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+3];
                }
                pipeline.consumer_release();
                
                if (s*TILE_WIDTH < k) {
                    pipeline.producer_acquire();
                    cuda::memcpy_async(Mds[stage] + ty*TILE_WIDTH + tx*4, a_fp32 + row*k + s*TILE_WIDTH + tx*4, cuda::aligned_size_t<4>(sizeof(float)*4), pipeline);
                    cuda::memcpy_async(Nds[stage] + ty*TILE_WIDTH + tx*4, b_fp32 + (s*TILE_WIDTH + ty)*n + col, cuda::aligned_size_t<4>(sizeof(float)*4), pipeline);     
                    pipeline.producer_commit();
                }
                
                s += 1;
            }

            c_fp32[row*n+col+0] = alpha * res[0] + beta * c_fp32[row*n+col+0];
            c_fp32[row*n+col+1] = alpha * res[1] + beta * c_fp32[row*n+col+1];
            c_fp32[row*n+col+2] = alpha * res[2] + beta * c_fp32[row*n+col+2];
            c_fp32[row*n+col+3] = alpha * res[3] + beta * c_fp32[row*n+col+3];
        }
    }
}
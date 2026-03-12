#include "cuda_gemm.h"
using namespace nvcuda;

__global__
void gemm_fp32_cuda_tiled_2D_async_warp_spl(
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
    cuda::pipeline<cuda::thread_scope_block> pipe = cuda::make_pipeline(block, &shared_state, 32);

    __shared__ alignas(16) float Mds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH*TILE_WIDTH];
    __shared__ alignas(16) float Nds[NUM_STAGES_ASYNC_PIPELINE][TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = by*TILE_WIDTH*COARSE_FACTOR_2D + ty;
    int col_start = bx*TILE_WIDTH*COARSE_FACTOR_2D + tx*4;
    int tid = block.thread_rank();
    int warp_id = tid/32;

    for (int r = 0; r < COARSE_FACTOR_2D; r++) {
        int row = row_start + r*TILE_WIDTH;
        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = col_start + c*TILE_WIDTH;
            float res[4] = {0.0f};

            if (warp_id == 0) {
                int s = 0;
                int row_off = by*TILE_WIDTH*COARSE_FACTOR_2D + r*TILE_WIDTH + tid;
                int col_off = bx*TILE_WIDTH*COARSE_FACTOR_2D + c*TILE_WIDTH;

                for (int ph = 0; ph < k; ph += TILE_WIDTH) {
                    int stage = s % NUM_STAGES_ASYNC_PIPELINE;
                    pipe.producer_acquire();
                    cuda::memcpy_async(Mds[stage] + tid*TILE_WIDTH, a_fp32 + row_off*k + ph, cuda::aligned_size_t<4>(sizeof(float)*32), pipe);
                    cuda::memcpy_async(Nds[stage] + tid*TILE_WIDTH, b_fp32 + (ph + tid)*n + col_off, cuda::aligned_size_t<4>(sizeof(float)*32), pipe);
                    pipe.producer_commit();
                    s += 1;
                }
            }
            else {
                auto consumer_group = cooperative_groups::tiled_partition<32>(block);
                int s = 0;
                int row_off = ty-4;

                for (int ph = 0; ph < k; ph += TILE_WIDTH) {
                    int stage = s % NUM_STAGES_ASYNC_PIPELINE;
                    pipe.consumer_wait();
                    for (int i = 0; i < TILE_WIDTH; i++) {
                        res[0] += Mds[stage][row_off*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+0];
                        res[1] += Mds[stage][row_off*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+1];
                        res[2] += Mds[stage][row_off*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+2];
                        res[3] += Mds[stage][row_off*TILE_WIDTH+i]*Nds[stage][i*TILE_WIDTH+tx*4+3];
                    }
                    cooperative_groups::sync(consumer_group);
                    pipe.consumer_release();
                    s += 1;
                }

                row_off = row - 4;
                c_fp32[row_off*n+col+0] = alpha * res[0] + beta * c_fp32[row_off*n+col+0];
                c_fp32[row_off*n+col+1] = alpha * res[1] + beta * c_fp32[row_off*n+col+1];
                c_fp32[row_off*n+col+2] = alpha * res[2] + beta * c_fp32[row_off*n+col+2];
                c_fp32[row_off*n+col+3] = alpha * res[3] + beta * c_fp32[row_off*n+col+3];
            }
        }
    }
}
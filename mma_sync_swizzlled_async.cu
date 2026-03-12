#include "cuda_gemm.h"
using namespace nvcuda;

__global__ 
void gemm_mma_sync_fp16_2d_tiled_swizzled_async(
    half *a, 
    half *b, 
    float *c, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    __shared__ alignas(16) half Mds[NUM_STAGES_ASYNC_PIPELINE][32*32];
    __shared__ alignas(16) half Nds[NUM_STAGES_ASYNC_PIPELINE][32*32];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    int warp_row_id = idx/blockDim.x;
    int warp_col_id = (idx % blockDim.x)/32;
    int thread_id_in_warp = idx % 32;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {

            float regs_c_1[4] = {0.0f};
            float regs_c_2[4] = {0.0f};

            for (int s = 0; s < NUM_STAGES_ASYNC_PIPELINE; s++) {
                int a_row = (8 * blockIdx.y + i) * 32;
                int a_col = s*32;

                int b_row = s*32;
                int b_col = (8 * blockIdx.x + j) * 32;

                pipeline.producer_acquire();
                #pragma unroll
                for (int j1 = idx; j1 < 32*32; j1 += blockDim.x * blockDim.y) {
                    int row = j1/32;
                    int col = j1 % 32;
                    int s_col = get_swizzled_index(row, col, 32, 2, 8);
                    cuda::memcpy_async(Mds[s] + row*32 + s_col, a + (a_row + row) * k + (a_col + col), cuda::aligned_size_t<2>(sizeof(half)), pipeline);
                    cuda::memcpy_async(Nds[s] + row*32 + s_col, b + (b_row + row) * n + (b_col + col), cuda::aligned_size_t<2>(sizeof(half)), pipeline);
                }
                pipeline.producer_commit();
            }

            int s = NUM_STAGES_ASYNC_PIPELINE;

            for (int k1 = 0; k1 < k; k1 += 32) {
                int stage = s % NUM_STAGES_ASYNC_PIPELINE;

                int a_row = (8 * blockIdx.y + i) * 32;
                int a_col = s*32;

                int b_row = s*32;
                int b_col = (8 * blockIdx.x + j) * 32;

                constexpr size_t pending_batches = NUM_STAGES_ASYNC_PIPELINE - 1;
                cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline);
                __syncthreads();

                for (int k2 = 0; k2 < 32; k2 += 16) {
                    uint32_t regs_a[4];

                    uint32_t regs_b_1[2];
                    uint32_t regs_b_2[2];

                    int m_row = warp_row_id * 16;
                    int m_col = k2;

                    int n_row = k2;
                    int n_col_1 = warp_col_id * 16;
                    int n_col_2 = n_col_1 + 8;

                    int x = (thread_id_in_warp/16) * 8 + m_col;
                    int y = n_col_1;
                    int z = n_col_2;

                    x = get_swizzled_index(m_row + thread_id_in_warp % 16, x, 32, 2, 8);
                    y = get_swizzled_index(n_row + thread_id_in_warp % 16, y, 32, 2, 8);
                    z = get_swizzled_index(n_row + thread_id_in_warp % 16, z, 32, 2, 8);

                    uint32_t addr_a   = __cvta_generic_to_shared(&Mds[stage][(m_row + thread_id_in_warp % 16) * 32 + x]);
                    uint32_t addr_b_1 = __cvta_generic_to_shared(&Nds[stage][(n_row + thread_id_in_warp % 16) * 32 + y]);
                    uint32_t addr_b_2 = __cvta_generic_to_shared(&Nds[stage][(n_row + thread_id_in_warp % 16) * 32 + z]);

                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                        "{%0, %1, %2, %3}, [%4];"
                        : "=r"(regs_a[0]), "=r"(regs_a[1]), "=r"(regs_a[2]), "=r"(regs_a[3])
                        : "r"(addr_a)
                    );

                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x2.shared.trans.b16 "
                        "{%0, %1}, [%2];"
                        : "=r"(regs_b_1[0]), "=r"(regs_b_1[1])
                        : "r"(addr_b_1)
                    );

                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x2.shared.trans.b16 "
                        "{%0, %1}, [%2];"
                        : "=r"(regs_b_2[0]), "=r"(regs_b_2[1])
                        : "r"(addr_b_2)
                    );

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+f"(regs_c_1[0]), "+f"(regs_c_1[1]), "+f"(regs_c_1[2]),"+f"(regs_c_1[3])
                        : "r"(regs_a[0]), "r"(regs_a[1]), "r"(regs_a[2]), "r"(regs_a[3]), "r"(regs_b_1[0]), "r"(regs_b_1[1])
                    );

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+f"(regs_c_2[0]), "+f"(regs_c_2[1]), "+f"(regs_c_2[2]),"+f"(regs_c_2[3])
                        : "r"(regs_a[0]), "r"(regs_a[1]), "r"(regs_a[2]), "r"(regs_a[3]), "r"(regs_b_2[0]), "r"(regs_b_2[1])
                    );
                }

                pipeline.consumer_release();
                __syncthreads();

                pipeline.producer_acquire();
                if (a_col < k) {
                    #pragma unroll
                    for (int j1 = idx; j1 < 32*32; j1 += blockDim.x * blockDim.y) {
                        int row = j1/32;
                        int col = j1 % 32;
                        int s_col = get_swizzled_index(row, col, 32, 2, 8);
                        cuda::memcpy_async(Mds[stage] + row*32 + s_col, a + (a_row + row) * k + (a_col + col), cuda::aligned_size_t<2>(sizeof(half)), pipeline);
                        cuda::memcpy_async(Nds[stage] + row*32 + s_col, b + (b_row + row) * n + (b_col + col), cuda::aligned_size_t<2>(sizeof(half)), pipeline);
                    }
                }
                pipeline.producer_commit();
                
                s += 1;
            }

            int a_row = (8 * blockIdx.y + i) * 32;
            int b_col = (8 * blockIdx.x + j) * 32;

            int m_row   = warp_row_id * 16;
            int n_col_1 = warp_col_id * 16;
            int n_col_2 = n_col_1 + 8;

            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int rw = (thread_id_in_warp >> 2) + 8 * (q / 2);
                int cl = 2 * (thread_id_in_warp % 4) + (q % 2);
                c[(a_row + m_row + rw) * n + (b_col + n_col_1 + cl)] += regs_c_1[q];
                c[(a_row + m_row + rw) * n + (b_col + n_col_2 + cl)] += regs_c_2[q];
            }
        }
    }
}
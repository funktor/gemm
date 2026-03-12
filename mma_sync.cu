#include "cuda_gemm.h"
using namespace nvcuda;

__global__ 
void gemm_mma_sync_fp16(
    const half *a, 
    const half *b, 
    float *c, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    __shared__ alignas(16) half Mds[TILE_WIDTH_WMMA*TILE_WIDTH_WMMA];
    __shared__ alignas(16) half Nds[TILE_WIDTH_WMMA*TILE_WIDTH_WMMA];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    int warp_row_id = idx/blockDim.x;
    int warp_col_id = (idx % blockDim.x)/32;
    int thread_id_in_warp = idx % 32;

    for (int i = 0; i < k; i += TILE_WIDTH_WMMA) {
        int a_row = blockIdx.y * TILE_WIDTH_WMMA;
        int a_col = i;

        for (int j = idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
            Mds[j] = a[(a_row + j/TILE_WIDTH_WMMA) * k + (a_col + j % TILE_WIDTH_WMMA)];
        }

        int b_row = i;
        int b_col = blockIdx.x * TILE_WIDTH_WMMA;

        for (int j = idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
            Nds[j] = b[(b_row + j/TILE_WIDTH_WMMA) * n + (b_col + j % TILE_WIDTH_WMMA)];
        }

        __syncthreads();

        for (int j = 0; j < TILE_WIDTH_WMMA; j += 16) {
            uint32_t regs_a[4];

            uint32_t regs_b_1[2];
            uint32_t regs_b_2[2];

            float regs_c_1[4] = {0.0f};
            float regs_c_2[4] = {0.0f};

            int m_row = warp_row_id * 16;
            int m_col = j;

            int n_row = j;
            int n_col_1 = warp_col_id * 16;
            int n_col_2 = n_col_1 + 8;

            uint32_t addr_a   = __cvta_generic_to_shared(&Mds[(m_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + (thread_id_in_warp/16) * 8 + m_col]);
            uint32_t addr_b_1 = __cvta_generic_to_shared(&Nds[(n_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + n_col_1]);
            uint32_t addr_b_2 = __cvta_generic_to_shared(&Nds[(n_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + n_col_2]);

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

            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int rw = (thread_id_in_warp >> 2) + 8 * (q / 2);
                int cl = 2 * (thread_id_in_warp % 4) + (q % 2);
                c[(a_row + m_row + rw) * n + (b_col + n_col_1 + cl)] += regs_c_1[q];
                c[(a_row + m_row + rw) * n + (b_col + n_col_2 + cl)] += regs_c_2[q];
            }
        }

        __syncthreads();
    }
}
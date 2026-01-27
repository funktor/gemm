#include "cuda_gemm.h"
using namespace nvcuda;

#define TILE_WIDTH 32
#define TILE_WIDTH_WMMA 64
#define COARSE_FACTOR 4
#define COARSE_FACTOR_2D 4
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


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

void gemm_fp16_cublas(
    const __half *a_fp16, 
    const __half *b_fp16, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    cublasHandle_t handle;
    cublasErrCheck(cublasCreate(&handle));
    // Use tensor cores
    cublasErrCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    cublasErrCheck(
        cublasGemmEx(
            handle, 
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            b_fp16, CUDA_R_16F, n,
            a_fp16, CUDA_R_16F, k,
            &beta,
            c_fp32, CUDA_R_32F, n,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
    );

    cublasDestroy(handle);
}

void gemm_fp32_cublas(
    const float *a_fp32, 
    const float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    cublasHandle_t handle;
    cublasErrCheck(cublasCreate(&handle));
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasErrCheck(
        cublasSgemm(
            handle, 
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            b_fp32, n,
            a_fp32, k,
            &beta,
            c_fp32, n
        )
    );

    cublasDestroy(handle);
}

__global__
void gemm_fp32_cuda(
    const float *a_fp32, 
    const float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float res = 0.0f;
        for (int i = 0; i < k; i++) res += a_fp32[row*k+i]*b_fp32[i*n+col];
        c_fp32[row*n+col] = alpha*res + beta*c_fp32[row*n+col];
    }
}

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

    for (int i = 0; i < k; i += TILE_WIDTH_WMMA) {
        int a_block_row = blockIdx.y * TILE_WIDTH_WMMA;
        int a_block_col = i;
        int a_idx = threadIdx.y * blockDim.x + threadIdx.x;

        for (int j = a_idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
            Mds[j] = a[(a_block_row + j/TILE_WIDTH_WMMA) * k + a_block_col + (j % TILE_WIDTH_WMMA)];
        }

        int b_block_row = i;
        int b_block_col = blockIdx.x * TILE_WIDTH_WMMA;
        int b_idx = threadIdx.y * blockDim.x + threadIdx.x;

        for (int j = b_idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
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
            int n_col_1 = 2 * warp_col_id * 8;
            int n_col_2 = n_col_1 + 8;

            uint32_t addr_a = __cvta_generic_to_shared(&Mds[(m_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + (thread_id_in_warp/16) * 8 + m_col]);
            uint32_t addr_b_1 = __cvta_generic_to_shared(&Nds[(n_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + n_col_1]);
            uint32_t addr_b_2 = __cvta_generic_to_shared(&Nds[(n_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + n_col_2]);

            __syncthreads();

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
    }

    // for (int i = 0; i < k; i += 16) {
    //     int a_row = blockIdx.y * 16;
    //     int a_col = i;
    //     int idx = threadIdx.y * blockDim.x + threadIdx.x;

    //     for (int j = idx; j < 256; j += blockDim.x * blockDim.y) {
    //         Mds[j] = a[(a_row + j/16) * k + (a_col + j % 16)];
    //     }

    //     int b_row = i;
    //     int b_col = blockIdx.x * 8;

    //     for (int j = idx; j < 128; j += blockDim.x * blockDim.y) {
    //         Nds[j] = b[(b_row + j/8) * n + (b_col + j % 8)];
    //     }

    //     uint32_t regs_a[4];
    //     uint32_t regs_b[2];
    //     float regs_c[4] = {0.0f};
        
    //     uint32_t addr_a = __cvta_generic_to_shared(&Mds[(idx % 16) * 16 + (idx/16) * 8]);
    //     uint32_t addr_b = __cvta_generic_to_shared(&Nds[(idx % 16) * 8]);

    //     asm volatile(
    //         "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    //         "{%0, %1, %2, %3}, [%4];"
    //         : "=r"(regs_a[0]), "=r"(regs_a[1]), "=r"(regs_a[2]), "=r"(regs_a[3])
    //         : "r"(addr_a)
    //     );

    //     asm volatile(
    //         "ldmatrix.sync.aligned.m8n8.x2.shared.trans.b16 "
    //         "{%0, %1}, [%2];"
    //         : "=r"(regs_b[0]), "=r"(regs_b[1])
    //         : "r"(addr_b)
    //     );

    //     asm volatile(
    //         "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    //         "{%0, %1, %2, %3}, "
    //         "{%4, %5, %6, %7}, "
    //         "{%8, %9}, "
    //         "{%0, %1, %2, %3};\n"
    //         : "+f"(regs_c[0]), "+f"(regs_c[1]), "+f"(regs_c[2]),"+f"(regs_c[3])
    //         : "r"(regs_a[0]), "r"(regs_a[1]), "r"(regs_a[2]), "r"(regs_a[3]), "r"(regs_b[0]), "r"(regs_b[1])
    //     );

    //     #pragma unroll
    //     for (int i = 0; i < 4; i++) {
    //         int rw = (idx >> 2) + 8 * (i / 2);
    //         int cl = 2 * (idx % 4) + (i % 2);
    //         c[(a_row + rw) * n + (b_col + cl)] += regs_c[i];
    //     }
    // }
}

__global__ 
void gemm_mma_sync_fp16_vectorized(
    half *a, 
    half *b, 
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
            int row = j/TILE_WIDTH_WMMA;
            int col = j % TILE_WIDTH_WMMA;
            Mds[j] = a[(a_row + row) * k + (a_col + col)];
        }

        int b_row = i;
        int b_col = blockIdx.x * TILE_WIDTH_WMMA;

        for (int j = idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
            int row = j/TILE_WIDTH_WMMA;
            int col = j % TILE_WIDTH_WMMA;
            Nds[j] = b[(b_row + row) * n + (b_col + col)];
        }

        __syncthreads();

        for (int j = 0; j < TILE_WIDTH_WMMA; j += 32) {
            int a0_row = warp_row_id * 32;
            int a0_col = j;

            int b0_row = j;
            int b0_col = warp_col_id * 32;

            #pragma unroll
            for (int r1 = 0; r1 < 32; r1 += 16) {
                for (int c1 = 0; c1 < 32; c1 += 16) {
                    for (int k1 = 0; k1 < 32; k1 += 16) {
                        uint32_t regs_a[4];

                        uint32_t regs_b_1[2];
                        uint32_t regs_b_2[2];

                        float regs_c_1[4] = {0.0f};
                        float regs_c_2[4] = {0.0f};

                        int m_row = a0_row + r1;
                        int m_col = a0_col + k1;

                        int n_row = b0_row + k1;
                        int n_col_1 = b0_col + c1;
                        int n_col_2 = n_col_1 + 8;

                        uint32_t addr_a   = __cvta_generic_to_shared(&Mds[(m_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + (thread_id_in_warp/16) * 8 + m_col]);
                        uint32_t addr_b_1 = __cvta_generic_to_shared(&Nds[(n_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + n_col_1]);
                        uint32_t addr_b_2 = __cvta_generic_to_shared(&Nds[(n_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + n_col_2]);

                        __syncthreads();

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
                }
            }
        }
    }
}


__global__ 
void gemm_mma_sync_fp16_swizzled(
    half *a, 
    half *b, 
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
            int row = j/TILE_WIDTH_WMMA;
            int col = j % TILE_WIDTH_WMMA;
            int s_col = (col/8)*8 + (row % 8)^(col % 8);

            Mds[row*TILE_WIDTH_WMMA + s_col] = a[(a_row + row) * k + (a_col + col)];
        }

        int b_row = i;
        int b_col = blockIdx.x * TILE_WIDTH_WMMA;

        for (int j = idx; j < TILE_WIDTH_WMMA*TILE_WIDTH_WMMA; j += blockDim.x * blockDim.y) {
            int row = j/TILE_WIDTH_WMMA;
            int col = j % TILE_WIDTH_WMMA;
            int s_col = (col/8)*8 + (row % 8)^(col % 8);

            Nds[row*TILE_WIDTH_WMMA + s_col] = b[(b_row + row) * n + (b_col + col)];
        }

        __syncthreads();

        for (int j = 0; j < TILE_WIDTH_WMMA; j += 32) {
            int a0_row = warp_row_id * 32;
            int a0_col = j;

            int b0_row = j;
            int b0_col = warp_col_id * 32;

            #pragma unroll
            for (int r1 = 0; r1 < 32; r1 += 16) {
                for (int c1 = 0; c1 < 32; c1 += 16) {
                    for (int k1 = 0; k1 < 32; k1 += 16) {
                        half a_tile[8] = {};
                        half b_tile_1[4] = {};
                        half b_tile_2[4] = {};

                        // uint32_t regs_a[4];

                        // uint32_t regs_b_1[2];
                        // uint32_t regs_b_2[2];

                        float regs_c_1[4] = {0.0f};
                        float regs_c_2[4] = {0.0f};

                        int m_row = a0_row + r1;
                        int m_col = a0_col + k1;

                        int n_row = b0_row + k1;
                        int n_col_1 = b0_col + c1;
                        int n_col_2 = n_col_1 + 8;

                        for (int q = 0; q < 8; q++) {
                            int row = (thread_id_in_warp >> 2) + 8 * ((q / 2) % 2);
                            int col = 2 * (thread_id_in_warp % 4) + (q % 2) + 8 * (q / 4);
                            int s_col = (col/8)*8 + (row % 8) ^ (col % 8);
                            a_tile[q] = Mds[(m_row + row)*TILE_WIDTH_WMMA + m_col + s_col];
                        }

                        for (int q = 0; q < 4; q++) {
                            int row = (thread_id_in_warp % 4) * 2 + (q % 2) + 8 * (q / 2);
                            int col = thread_id_in_warp >> 2;
                            int s_col = (col/8)*8 + (row % 8) ^ (col % 8);
                            b_tile_1[q] = Nds[(n_row + row)*TILE_WIDTH_WMMA + n_col_1 + s_col];
                            b_tile_2[q] = Nds[(n_row + row)*TILE_WIDTH_WMMA + n_col_2 + s_col];
                        }

                        __syncthreads();

                        const int *regs_a = (const int *)a_tile;
                        const int *regs_b_1 = (const int *)b_tile_1;
                        const int *regs_b_2 = (const int *)b_tile_2;

                        // uint32_t addr_a   = __cvta_generic_to_shared(&Mds[(m_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + (thread_id_in_warp/16) * 8 + m_col]);
                        // uint32_t addr_b_1 = __cvta_generic_to_shared(&Nds[(n_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + n_col_1]);
                        // uint32_t addr_b_2 = __cvta_generic_to_shared(&Nds[(n_row + thread_id_in_warp % 16) * TILE_WIDTH_WMMA + n_col_2]);

                        

                        // asm volatile(
                        //     "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                        //     "{%0, %1, %2, %3}, [%4];"
                        //     : "=r"(regs_a[0]), "=r"(regs_a[1]), "=r"(regs_a[2]), "=r"(regs_a[3])
                        //     : "r"(addr_a)
                        // );

                        // asm volatile(
                        //     "ldmatrix.sync.aligned.m8n8.x2.shared.trans.b16 "
                        //     "{%0, %1}, [%2];"
                        //     : "=r"(regs_b_1[0]), "=r"(regs_b_1[1])
                        //     : "r"(addr_b_1)
                        // );

                        // asm volatile(
                        //     "ldmatrix.sync.aligned.m8n8.x2.shared.trans.b16 "
                        //     "{%0, %1}, [%2];"
                        //     : "=r"(regs_b_2[0]), "=r"(regs_b_2[1])
                        //     : "r"(addr_b_2)
                        // );

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
                }
            }
        }
    }
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

int main(){
    int m = 2048;
    int n = 2048;
    int k = 128;

    float *a_fp32;
    float *b_fp32;

    cudaErrCheck(cudaMallocManaged(&a_fp32, m * k * sizeof(float)));
    cudaErrCheck(cudaMallocManaged(&b_fp32, k * n * sizeof(float)));

    generate_data(a_fp32, m*k);
    generate_data(b_fp32, k*n);

    float cublasTime;
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;
    
    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));



    float *c_cpu_fp32;
    cudaErrCheck(cudaMallocManaged(&c_cpu_fp32, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_cpu_fp32[i] = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();
    gemm_cpu(a_fp32, b_fp32, c_cpu_fp32, 1.0, 0.0, m, n, k);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "CPU GEMM Duration = " << duration.count() << " ms" << std::endl;



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



    half *a_fp16;
    half *b_fp16;
    float *c_gpu_fp32_wmma;

    cudaErrCheck(cudaMallocManaged(&a_fp16, m * k * sizeof(half)));
    cudaErrCheck(cudaMallocManaged(&b_fp16, k * n * sizeof(half)));
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_wmma, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32_wmma[i] = 0.0f;

    convertFp32ToFp16 <<< (m * k + 255) / 256, 256 >>> (a_fp16, a_fp32, m * k);
    convertFp32ToFp16 <<< (k * n + 255) / 256, 256 >>> (b_fp16, b_fp32, k * n);
    cudaDeviceSynchronize();

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




    float *c_gpu_mma_sync_fp16_vec;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16_vec, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16_vec[i] = 0.0f;

    dim3 bd7(64, 2, 1);
    dim3 gd7((n+TILE_WIDTH_WMMA-1)/TILE_WIDTH_WMMA, (m+TILE_WIDTH_WMMA-1)/TILE_WIDTH_WMMA, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16_vectorized<<<gd7, bd7>>>(a_fp16, b_fp16, c_gpu_mma_sync_fp16_vec, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 VEC GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16_vec, m*n) << std::endl;



    float *c_gpu_mma_sync_fp16_swz;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16_swz, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16_swz[i] = 0.0f;

    dim3 bd8(64, 2, 1);
    dim3 gd8((n+TILE_WIDTH_WMMA-1)/TILE_WIDTH_WMMA, (m+TILE_WIDTH_WMMA-1)/TILE_WIDTH_WMMA, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16_swizzled<<<gd8, bd8>>>(a_fp16, b_fp16, c_gpu_mma_sync_fp16_swz, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 SWIZZLED GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16_swz, m*n) << std::endl;




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

    cudaErrCheck(cudaEventDestroy(startcublas));             
    cudaErrCheck(cudaEventDestroy(stopcublas));
    
    cudaErrCheck(cudaFree(a_fp32));
    cudaErrCheck(cudaFree(b_fp32));
    cudaErrCheck(cudaFree(c_cpu_fp32));
    cudaErrCheck(cudaFree(a_fp16));
    cudaErrCheck(cudaFree(b_fp16));
    cudaErrCheck(cudaFree(c_gpu_fp32));
    cudaErrCheck(cudaFree(d_gpu_fp32));

    cudaErrCheck(cudaDeviceReset());
    return 0;
}
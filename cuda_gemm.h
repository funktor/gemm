#ifndef CUDA_GEMM_H
#define CUDA_GEMM_H

#include <tbb/tbb.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <tuple>
#include <map>
#include <fcntl.h>
#include <functional>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <ctime> 
#include <stdbool.h>    // bool type
#include <fstream>
#include <cmath>
#include <variant>
#include <assert.h>
#include <initializer_list>

#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <cuda/pipeline>

using namespace nvcuda;

#define TILE_WIDTH 32
#define TILE_WIDTH_WMMA 64
#define COARSE_FACTOR 4
#define COARSE_FACTOR_2D 4
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define NUM_STAGES_ASYNC_PIPELINE 4
#define MAX_NUM_TILES 128

struct __align__(8) half4 {
    half2 a;
    half2 b;
};

template <typename T> struct Afrag_16x16 {
    static constexpr size_t ne = 8; // num of elements per thread
    T x[ne];

    static __device__ size_t get_row(int tid, int l) {
        int group_id = tid >> 2;
        return group_id + 8 * ((l / 2) % 2);
    }

    static __device__ size_t get_col(int tid, int l) {
        return 2 * (tid % 4) + (l % 2) + 8 * (l / 4);
    }
};

template <typename T> struct Bfrag_16x8 {
    static constexpr size_t ne = 4;
    T x[ne] = {};
    static __device__ size_t get_row(int tid, int l) {
        return (tid % 4) * 2 + (l % 2) + 8 * (l / 2);
    }

    static __device__ size_t get_col(int tid, int l) { return tid >> 2; }
};

template <typename T> struct CFrag_16x8 {
    static constexpr size_t ne = 4;
    T x[ne] = {};

    static __device__ size_t get_row(int tid, int l) {
        return (tid >> 2) + 8 * (l / 2);
    }

    static __device__ size_t get_col(int tid, int l) {
        assert(l < ne);
        return 2 * (tid % 4) + (l % 2);
    }
};

void cudaErrCheck_(cudaError_t stat, const char *file, int line);
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line);
void curandErrCheck_(curandStatus_t stat, const char *file, int line);
void generate_data(float *x, const long n);
bool compare_matrices(const float *x, const float *y, const long n);
void transpose(const float *a, float *out, const int n, const int m);
void print_arr(const float *x, const long n);
__global__ void convertFp32ToFp16 (half *out, const float *in, const long n);

void gemm_cpu(
    const float *a, 
    const float *b, 
    float *c, 
    const float alpha,
    const float beta,
    const unsigned int m, 
    const unsigned int n, 
    const unsigned int k
);

void gemm_cpu_b_trans(
    const float *a, 
    const float *b, 
    float *c, 
    const float alpha,
    const float beta,
    const unsigned int m, 
    const unsigned int n, 
    const unsigned int k
);

void softmax(
    const float *inp, 
    float *out, 
    const unsigned int n, 
    const unsigned int m
);

void attention_cpu(
    const float *q, 
    const float *k, 
    const float *v, 
    float *out, 
    const unsigned int m, 
    const unsigned int n
);

void gemm_fp16_cublas(
    const __half *a_fp16, 
    const __half *b_fp16, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
);

void gemm_fp32_cublas(
    const float *a_fp32, 
    const float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

__global__
void gemm_fp32_cuda_tiled_2D_vectorize_b_trans(
    float *a_fp32, 
    float *b_fp32, 
    float *c_fp32, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
);

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
);

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
);

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
);

__global__ 
void gemm_mma_sync_fp16_2d_tiled(
    half *a, 
    half *b, 
    float *c, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
);

__global__ 
void gemm_mma_sync_fp16_2d_tiled_b_trans(
    half *a, 
    half *b, 
    float *c, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
);

__device__
int get_swizzled_index(int row, int col, int k, int u, int v);

__global__ 
void gemm_mma_sync_fp16_2d_tiled_swizzled(
    half *a, 
    half *b, 
    float *c, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
);

__global__ 
void gemm_mma_sync_fp16_2d_tiled_swizzled_explicit(
    half *a, 
    half *b, 
    float *c, 
    const float alpha, 
    const float beta, 
    const int m, 
    const int n, 
    const int k
);

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
);

__device__ __forceinline__ float atomicMaxF32(float *address, float val);

__global__
void attn_sftmax_dot_v(
    float *a_fp32, 
    float *b_fp32, 
    float *c_fp32, 
    const int m, 
    const int n, 
    const int k
);

__global__
void attn_qk_t(
    float *a_fp32, 
    float *b_fp32, 
    float *c_fp32, 
    const float alpha,
    const int m, 
    const int n, 
    const int k
);

void attention_gpu(
    float *q_fp32, 
    float *k_fp32, 
    float *v_fp32, 
    float *out, 
    const int m, 
    const int k
);

#endif
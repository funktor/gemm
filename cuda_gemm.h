#ifndef CUDA_GEMM_H
#define CUDA_GEMM_H

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <chrono>
#include <random>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <cuda/pipeline>

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

void generate_data(float *x, const long n);

void gemm_cpu(
    const float *a, 
    const float *b, 
    float *c, 
    const float alpha,
    const float beta,
    const int m, 
    const int n, 
    const int k
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

bool compare_matrices(const float *x, const float *y, const long n);
__global__ void convertFp32ToFp16 (half *out, const float *in, const long n);

#endif
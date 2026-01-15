#ifndef CUDA_GEMM_H
#define CUDA_GEMM_H

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>

void gemm_cpu(
    const float *a, 
    const float *b, 
    float *c, 
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
void convertFp32ToFp16 (half *out, const float *in, const long n);

#endif
#include "cuda_gemm.h"
using namespace std;

void gemm_cpu(
    const float *a, 
    const float *b, 
    float *c, 
    const float alpha,
    const float beta,
    const unsigned int m, 
    const unsigned int n, 
    const unsigned int k
) {
    omp_set_num_threads(8);
    #pragma omp parallel for shared(a, b, c)
    for(auto i = 0; i < m; i++) {
        for (auto j = 0; j < n; j++) {
            float r = 0.0f;
            for (auto q = 0; q < k; q++) r += a[i*k+q]*b[q*n+j];
            c[i*n+j] = alpha*r + beta*c[i*n+j];
        }
    }
}

void gemm_cpu_b_trans(
    const float *a, 
    const float *b, 
    float *c, 
    const float alpha,
    const float beta,
    const unsigned int m, 
    const unsigned int n, 
    const unsigned int k
) {
    omp_set_num_threads(8);
    #pragma omp parallel for shared(a, b, c)
    for(auto i = 0; i < m; i++) {
        for (auto j = 0; j < n; j++) {
            float r = 0.0f;
            for (auto q = 0; q < k; q++) r += a[i*k+q]*b[j*k+q];
            c[i*n+j] = alpha*r + beta*c[i*n+j];
        }
    }
}

void softmax(
    const float *inp, 
    float *out, 
    const unsigned int n, 
    const unsigned int m
) {

    float *max_per_row = new float[n];
    float *sum_per_row = new float[n];

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n), 
        [&max_per_row, &sum_per_row](tbb::blocked_range<size_t> r) {
        for (auto i = r.begin(); i < r.end(); i++) {
            max_per_row[i] = -MAXFLOAT;
            sum_per_row[i] = 0.0;
        }
    });

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n), 
        [&max_per_row, &inp, m](tbb::blocked_range<size_t> r) {
        for (auto i = r.begin(); i < r.end(); i++) {
            for (unsigned long j = 0; j < m; j++) {
                max_per_row[i] = std::max(max_per_row[i], inp[i*m+j]);
            }
        }
    });

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n), 
        [&sum_per_row, &max_per_row, &inp, m](tbb::blocked_range<size_t> r) {
        for (auto i = r.begin(); i < r.end(); i++) {
            for (unsigned long j = 0; j < m; j++) {
                sum_per_row[i] += exp(inp[i*m+j]-max_per_row[i]);
            }
        }
    });

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n), 
        [&sum_per_row, &max_per_row, &inp, &out, m](tbb::blocked_range<size_t> r) {
        for (auto i = r.begin(); i < r.end(); i++) {
            for (unsigned long j = 0; j < m; j++) {
                out[i*m+j] = exp(inp[i*m+j]-max_per_row[i])/sum_per_row[i];
            }
        }
    });
}

void attention_cpu(
    const float *q, 
    const float *k, 
    const float *v, 
    float *out, 
    const unsigned int m, 
    const unsigned int n
) {
    float *qk_t = new float[m*m];
    gemm_cpu_b_trans(q, k, qk_t, 1.0, 0.0, m, m, n);

    for (auto i = 0; i < m*m; i++) qk_t[i] /= sqrt(n);

    float *sftmax = new float[m*m];
    softmax(qk_t, sftmax, m, m);

    gemm_cpu(sftmax, v, out, 1.0, 0.0, m, n, m);
}
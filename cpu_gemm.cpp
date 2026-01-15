#include "cuda_gemm.h"
using namespace std;

void gemm_cpu(
    const float *a, 
    const float *b, 
    float *c, 
    const float alpha,
    const float beta,
    const int m, 
    const int n, 
    const int k
) 
{
    for (auto i = 0; i < m*n; i++) c[i] *= beta;

    omp_set_num_threads(8);
    #pragma omp parallel for shared(a, b, c)
    for(auto i = 0; i < m; i++) {
        for (auto j = 0; j < n; j++) {
            float r = 0.0f;
            for (auto q = 0; q < k; q++) r += alpha*a[i*k+q]*b[q*n+j];
            c[i*n+j] = r;
        }
    }
}
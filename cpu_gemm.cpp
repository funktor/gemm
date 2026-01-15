#include "cuda_gemm.h"
using namespace std;

void gemm_cpu(
    const float *a, 
    const float *b, 
    float *c, 
    const int m, 
    const int n, 
    const int k
) 
{
    for (int i = 0; i < m*n; i++) c[i] = 0.0;

    omp_set_num_threads(8);
    #pragma omp parallel for shared(a, b, c)
    for(int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int q = 0; q < k; q++) c[i*n+j] += a[i*k+q]*b[q*n+j];
        }
    }
}
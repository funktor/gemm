# gemm
CUDA GEMM kernels

```
nvcc cuda_gemm.h cpu_gemm.cpp cublas_gemm.cu -o my_gemm -O3 -Xcompiler -O3 --gpu-code=sm_90 -arch=compute_90
```

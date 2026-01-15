# gemm
CUDA GEMM kernels

```
nvcc cpu_gemm.cpp cublas_gemm.cu -Xcompiler -fopenmp -o my_gemm -O3 -Xcompiler -O3 --gpu-code=sm_90 -arch=compute_90 -lcublas -lcurand
```

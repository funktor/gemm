# gemm
CUDA GEMM kernels

```
nvcc cpu_gemm.cu cublas_gemm.cu -Xcompiler -fopenmp -o my_gemm -O3 -Xcompiler -O3 --gpu-code=sm_89 -arch=compute_89 -lcublas -lcurand
nvcc cpu_gemm.cu cublas_gemm.cu -Xcompiler -fopenmp -o my_gemm -O3 -Xcompiler -O3 --gpu-code=sm_80 -arch=compute_80 -lcublas -lcurand
nvcc cpu_gemm.cu cublas_gemm.cu -Xcompiler -fopenmp -o my_gemm -O3 -Xcompiler -O3 --gpu-code=sm_75 -arch=compute_75 -lcublas -lcurand

```

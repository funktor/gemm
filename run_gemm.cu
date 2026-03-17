#include "cuda_gemm.h"
using namespace nvcuda;

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }

int main(){
    int m = 1024;
    int n = 1024;
    int k = 1024;

    float *a_fp32;
    float *b_fp32;
    float *b_fp32_t;

    cudaErrCheck(cudaMallocManaged(&a_fp32, m * k * sizeof(float)));
    cudaErrCheck(cudaMallocManaged(&b_fp32, k * n * sizeof(float)));
    cudaErrCheck(cudaMallocManaged(&b_fp32_t, n * k * sizeof(float)));

    generate_data(a_fp32, m*k);
    generate_data(b_fp32, k*n);
    transpose(b_fp32, b_fp32_t, k, n);

    half *a_fp16;
    half *b_fp16;
    half *b_fp16_t;

    cudaErrCheck(cudaMallocManaged(&a_fp16, m * k * sizeof(half)));
    cudaErrCheck(cudaMallocManaged(&b_fp16, k * n * sizeof(half)));
    cudaErrCheck(cudaMallocManaged(&b_fp16_t, k * n * sizeof(half)));

    float cublasTime;
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;
    
    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));

    convertFp32ToFp16 <<< (m * k + 255) / 256, 256 >>> (a_fp16, a_fp32, m * k);
    convertFp32ToFp16 <<< (k * n + 255) / 256, 256 >>> (b_fp16, b_fp32, k * n);
    convertFp32ToFp16 <<< (k * n + 255) / 256, 256 >>> (b_fp16_t, b_fp32_t, k * n);
    cudaDeviceSynchronize();



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
    cudaErrCheck(cudaFree(c_gpu_fp32_ccores));



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
    cudaErrCheck(cudaFree(c_gpu_fp32_tiled));



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
    cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d));



    float *c_gpu_fp32_tiled_2d_async;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled_2d_async, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32_tiled_2d_async[i] = 0.0f;

    dim3 bd21(8, 32, 1);
    dim3 gd21((n+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp32_cuda_tiled_2D_async<<<gd21, bd21>>>(a_fp32, b_fp32, c_gpu_fp32_tiled_2d_async, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUDA TILED 2D ASYNC FP32 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32_tiled_2d_async, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d_async));




    float *c_gpu_fp32_tiled_2d_async_warp_spl;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled_2d_async_warp_spl, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32_tiled_2d_async_warp_spl[i] = 0.0f;

    dim3 bd22(8, 32, 1);
    dim3 gd22((n+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp32_cuda_tiled_2D_async_warp_spl<<<gd22, bd22>>>(a_fp32, b_fp32, c_gpu_fp32_tiled_2d_async_warp_spl, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUDA TILED 2D ASYNC WARP SPL FP32 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32_tiled_2d_async_warp_spl, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d_async_warp_spl));


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
    cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d_vec));



    float *c_gpu_fp32_tiled_2d_vec_b_trans;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_tiled_2d_vec_b_trans, m * n * sizeof(float)));

    dim3 bd31(8, 32, 1);
    dim3 gd31((n+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), (m+32*COARSE_FACTOR_2D-1)/(32*COARSE_FACTOR_2D), 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_fp32_cuda_tiled_2D_vectorize_b_trans<<<gd31, bd31>>>(a_fp32, b_fp32_t, c_gpu_fp32_tiled_2d_vec_b_trans, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU CUDA TILED 2D VEC B TRANS FP32 GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_fp32_tiled_2d_vec_b_trans, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_fp32_tiled_2d_vec_b_trans));



    float *c_gpu_fp32_wmma;
    cudaErrCheck(cudaMallocManaged(&c_gpu_fp32_wmma, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_fp32_wmma[i] = 0.0f;

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
    cudaErrCheck(cudaFree(c_gpu_fp32_wmma));



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
    cudaErrCheck(cudaFree(c_gpu_fp32_wmma_shmm));



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
    cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16));




    float *c_gpu_mma_sync_fp16_2d_tiled;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16_2d_tiled, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16_2d_tiled[i] = 0.0f;

    dim3 bd7(64, 2, 1);
    dim3 gd7((n+256-1)/256, (m+256-1)/256, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16_2d_tiled<<<gd7, bd7>>>(a_fp16, b_fp16, c_gpu_mma_sync_fp16_2d_tiled, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 2D TILED GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16_2d_tiled, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16_2d_tiled));


    float *c_gpu_mma_sync_fp16_2d_tiled_b_trans;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16_2d_tiled_b_trans, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16_2d_tiled_b_trans[i] = 0.0f;

    dim3 bd71(64, 2, 1);
    dim3 gd71((n+256-1)/256, (m+256-1)/256, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16_2d_tiled_b_trans<<<gd71, bd71>>>(a_fp16, b_fp16_t, c_gpu_mma_sync_fp16_2d_tiled_b_trans, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 2D B TRANS TILED GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16_2d_tiled_b_trans, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16_2d_tiled_b_trans));




    float *c_gpu_mma_sync_fp16_2d_tiled_swz;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16_2d_tiled_swz, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16_2d_tiled_swz[i] = 0.0f;

    dim3 bd8(64, 2, 1);
    dim3 gd8((n+256-1)/256, (m+256-1)/256, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16_2d_tiled_swizzled<<<gd8, bd8>>>(a_fp16, b_fp16, c_gpu_mma_sync_fp16_2d_tiled_swz, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 2D TILED SWIZZLED GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16_2d_tiled_swz, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16_2d_tiled_swz));




    float *c_gpu_mma_sync_fp16_2d_tiled_swz_async;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16_2d_tiled_swz_async, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16_2d_tiled_swz_async[i] = 0.0f;

    dim3 bd81(64, 2, 1);
    dim3 gd81((n+256-1)/256, (m+256-1)/256, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16_2d_tiled_swizzled_async<<<gd81, bd81>>>(a_fp16, b_fp16, c_gpu_mma_sync_fp16_2d_tiled_swz_async, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 2D TILED SWIZZLED ASYNC GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16_2d_tiled_swz_async, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16_2d_tiled_swz_async));



    float *c_gpu_mma_sync_fp16_2d_tiled_swz_exp;
    cudaErrCheck(cudaMallocManaged(&c_gpu_mma_sync_fp16_2d_tiled_swz_exp, m * n * sizeof(float)));

    for (auto i = 0; i < m*n; i++) c_gpu_mma_sync_fp16_2d_tiled_swz_exp[i] = 0.0f;

    dim3 bd9(64, 2, 1);
    dim3 gd9((n+256-1)/256, (m+256-1)/256, 1);

    cudaErrCheck(cudaEventRecord(startcublas));
    gemm_mma_sync_fp16_2d_tiled_swizzled_explicit<<<gd9, bd9>>>(a_fp16, b_fp16, c_gpu_mma_sync_fp16_2d_tiled_swz_exp, 1.0, 0.0, m, n, k);
    cudaDeviceSynchronize();
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    std::cout << "GPU MMA SYNC FP16 2D TILED SWIZZLED EXPLICIT GEMM Duration = " << cublasTime << " ms" << std::endl;
    std::cout << "Matrices matching = " << compare_matrices(c_cpu_fp32, c_gpu_mma_sync_fp16_2d_tiled_swz_exp, m*n) << std::endl;
    cudaErrCheck(cudaFree(c_gpu_mma_sync_fp16_2d_tiled_swz_exp));



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
    cudaErrCheck(cudaFree(c_gpu_fp32));


    
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
    cudaErrCheck(cudaFree(d_gpu_fp32));


    cudaErrCheck(cudaEventDestroy(startcublas));             
    cudaErrCheck(cudaEventDestroy(stopcublas));
    
    cudaErrCheck(cudaFree(a_fp32));
    cudaErrCheck(cudaFree(b_fp32));
    cudaErrCheck(cudaFree(c_cpu_fp32));
    cudaErrCheck(cudaFree(a_fp16));
    cudaErrCheck(cudaFree(b_fp16));
    cudaErrCheck(cudaDeviceReset());
    return 0;
}
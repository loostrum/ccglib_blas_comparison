#include <iostream>
#include <complex>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <ccglib/ccglib.hpp>

#include "config.h"

inline void cublas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(cublasGetStatusString(status));
  }
}

inline void cuda_check(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

int main() {
  Params<ccglib::float16, ccglib::float32, large> params;

  cublasHandle_t handle;
  cublas_check(cublasCreate(&handle));

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  cudaDataType_t Atype = CUDA_C_32F;
  cudaDataType_t Btype = CUDA_C_32F;
  cudaDataType_t Ctype = CUDA_C_32F;
  cublasComputeType_t Computetype = CUBLAS_COMPUTE_32F_FAST_16F;

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

  int lda = params.K;
  int ldb = params.K;
  int ldc = params.N;

  std::complex<float> alpha = {1, 0};
  std::complex<float> beta = {1, 0};

  size_t bytes_a = params.M * params.K * sizeof(float) * 2;
  size_t bytes_b = params.N * params.K * sizeof(float) * 2;
  size_t bytes_c = params.M * params.N * sizeof(float) * 2;

  void *d_a;
  void *d_b;
  void *d_c;

  cuda_check(cudaMalloc(&d_a, bytes_a));
  cuda_check(cudaMalloc(&d_b, bytes_b));
  cuda_check(cudaMalloc(&d_c, bytes_c));

  // should try _batched variant where the device pointers to a,b,c are arrays
  cudaEvent_t start, end;
  cuda_check(cudaEventCreate(&start));
  cuda_check(cudaEventCreate(&end));

  cuda_check(cudaEventRecord(start));
  cublas_check(cublasGemmEx(handle, transa, transb, params.M, params.N, params.K,
                            &alpha, d_a, Atype, lda, d_b, Btype, ldb, &beta,
                            d_c, Ctype, ldc, Computetype, algo));

  cuda_check(cudaEventRecord(end));
  cuda_check(cudaEventSynchronize(end));
  cuda_check(cudaDeviceSynchronize());

  float runtime;
  cuda_check(cudaEventElapsedTime(&runtime, start, end));
  const double tflops = 8ULL * 1e-9 * params.M * params.N * params.K / runtime;

  std::cout << "Runtime: " << runtime << " ms" << std::endl;
  std::cout << "TFLOPS: " << tflops << std::endl;
  

  cublas_check(cublasDestroy(handle));

  cuda_check(cudaFree(d_a));
  cuda_check(cudaFree(d_b));
  cuda_check(cudaFree(d_c));
}


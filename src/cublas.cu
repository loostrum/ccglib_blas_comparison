#include <iostream>
#include <complex>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <ccglib/ccglib.hpp>

#include "config.h"

#define cublas_check(status) {cublas_assert((status), __FILE__, __LINE__);}
inline void cublas_assert(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::string err = std::string(file) + " line " + std::to_string(line) + " " + std::string(cublasLtGetStatusString(status));
    throw std::runtime_error(err);
  }
}

inline void cuda_check(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

int main() {
  Params<ccglib::float16, ccglib::float32, large> params;

  cublasLtHandle_t handle;
  cublas_check(cublasLtCreate(&handle));

  cuComplex alpha = {1, 0};
  cuComplex beta = {0, 0};

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;

  // create matrix multiplication descriptor
  cublasLtMatmulDesc_t desc;
  // compute type (= effectively if/which tensor cores are used), scale type (=alpha,beta)
  cublas_check(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_C_32F));
  cublas_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  cublas_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));


  // planar layout, cublas needs offset between real and imag part
  long offset_a = params.M * params.K * sizeof(half);
  long offset_b = params.N * params.K * sizeof(half);
  long offset_c = params.M * params.N * sizeof(float);

  // create layout descriptors
  cublasLtMatrixLayout_t layout_a, layout_b, layout_c;
  // type, rows, cols, leading dimension (col major: nr elements from one col to the next)
  cublas_check(cublasLtMatrixLayoutCreate(&layout_a, CUDA_C_16F, params.M, params.K, params.K));
  cublas_check(cublasLtMatrixLayoutCreate(&layout_b, CUDA_C_16F, params.K, params.N, params.K));
  cublas_check(cublasLtMatrixLayoutCreate(&layout_c, CUDA_C_32F, params.M, params.N, params.N));

  // setting the plane offset to a nonzero value makes this a planar mode (vs interleaved real,imag)
  cublas_check(cublasLtMatrixLayoutSetAttribute(layout_a, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &offset_a, sizeof(offset_a)));
  cublas_check(cublasLtMatrixLayoutSetAttribute(layout_b, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &offset_b, sizeof(offset_b)));
  cublas_check(cublasLtMatrixLayoutSetAttribute(layout_c, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &offset_c, sizeof(offset_c)));

  // create matrices
  size_t bytes_a = params.M * params.K * sizeof(half) * 2;
  size_t bytes_b = params.N * params.K * sizeof(half) * 2;
  size_t bytes_c = params.M * params.N * sizeof(float) * 2;

  __half *d_a;
  __half *d_b;
  float *d_c;

  cuda_check(cudaMalloc(&d_a, bytes_a));
  cuda_check(cudaMalloc(&d_b, bytes_b));
  cuda_check(cudaMalloc(&d_c, bytes_c));


  // launch
  cudaEvent_t start, end;
  cuda_check(cudaEventCreate(&start));
  cuda_check(cudaEventCreate(&end));

  cuda_check(cudaEventRecord(start));
  cublas_check(cublasLtMatmul(handle,
                              desc,
                              &alpha,
                              d_a,
                              layout_a,
                              d_b,
                              layout_b,
                              &beta,
                              d_c,
                              layout_c,
                              d_c,
                              layout_c,
                              NULL, // algo
                              NULL, // workspace
                              0, // workspace size in bytes
                              0 // stream
                              ));
  cuda_check(cudaEventRecord(end));
  cuda_check(cudaEventSynchronize(end));
  cuda_check(cudaDeviceSynchronize());

  float runtime;
  cuda_check(cudaEventElapsedTime(&runtime, start, end));
  const double tflops = 8ULL * 1e-9 * params.M * params.N * params.K / runtime;

  std::cout << "Runtime: " << runtime << " ms" << std::endl;
  std::cout << "TFLOPS: " << tflops << std::endl;
  
  cublas_check(cublasLtMatrixLayoutDestroy(layout_a));
  cublas_check(cublasLtMatrixLayoutDestroy(layout_b));
  cublas_check(cublasLtMatrixLayoutDestroy(layout_c));
  cublas_check(cublasLtMatmulDescDestroy(desc));

  cublas_check(cublasLtDestroy(handle));

  cuda_check(cudaFree(d_a));
  cuda_check(cudaFree(d_b));
  cuda_check(cudaFree(d_c));
}


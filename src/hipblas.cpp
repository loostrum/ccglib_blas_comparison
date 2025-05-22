#include <iostream>
#include <complex>

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <ccglib/ccglib.hpp>

#include "config.h"

#define hipblas_check(status) {hipblas_assert((status), __FILE__, __LINE__);}
inline void hipblas_assert(hipblasStatus_t status, const char *file, int line) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    std::string err = std::string(file) + " line " + std::to_string(line) + " " + std::string(hipblasStatusToString(status));
    throw std::runtime_error(err);
  }
}

inline void hip_check(hipError_t err) {
  if (err != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(err));
  }
}

int main() {
  Params<ccglib::float16, ccglib::float32, large> params;

  hipblasLtHandle_t handle;
  hipblas_check(hipblasLtCreate(&handle));

  hipComplex alpha = {1, 0};
  hipComplex beta = {0, 0};

  hipblasOperation_t transa = HIPBLAS_OP_T;
  hipblasOperation_t transb = HIPBLAS_OP_N;

  // create matrix multiplication descriptor
  hipblasLtMatmulDesc_t desc;
  // compute type (= effectively if/which tensor cores are used), scale type (=alpha,beta)

  hipblas_check(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblas_check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  hipblas_check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  // planar layout, hipblas needs offset between real and imag part
  long offset_a = params.M * params.K * sizeof(half);
  long offset_b = params.N * params.K * sizeof(half);
  long offset_c = params.M * params.N * sizeof(float);

  // create layout descriptors
  hipblasLtMatrixLayout_t layout_a, layout_b, layout_c;
  // type, rows, cols, leading dimension (col major: nr elements from one col to the next)
  hipblas_check(hipblasLtMatrixLayoutCreate(&layout_a, HIP_R_16F, params.M, params.K, params.K));
  hipblas_check(hipblasLtMatrixLayoutCreate(&layout_b, HIP_R_16F, params.K, params.N, params.K));
  hipblas_check(hipblasLtMatrixLayoutCreate(&layout_c, HIP_R_32F, params.M, params.N, params.N));

  // create matrices
  size_t bytes_a = params.M * params.K * sizeof(half);
  size_t bytes_b = params.N * params.K * sizeof(half);
  size_t bytes_c = params.M * params.N * sizeof(float);

  half *d_a;
  half *d_b;
  float *d_c;

  hip_check(hipMalloc(&d_a, bytes_a));
  hip_check(hipMalloc(&d_b, bytes_b));
  hip_check(hipMalloc(&d_c, bytes_c));


  // launch
  hipEvent_t start, end;
  hip_check(hipEventCreate(&start));
  hip_check(hipEventCreate(&end));

  hip_check(hipEventRecord(start));
  hipblas_check(hipblasLtMatmul(handle,
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
  hip_check(hipEventRecord(end));
  hip_check(hipEventSynchronize(end));
  hip_check(hipDeviceSynchronize());

  float runtime;
  hip_check(hipEventElapsedTime(&runtime, start, end));
  const double tflops = 2ULL * 1e-9 * params.M * params.N * params.K / runtime;

  std::cout << "Runtime: " << runtime << " ms" << std::endl;
  std::cout << "TFLOPS: " << tflops << std::endl;
  
  hipblas_check(hipblasLtMatrixLayoutDestroy(layout_a));
  hipblas_check(hipblasLtMatrixLayoutDestroy(layout_b));
  hipblas_check(hipblasLtMatrixLayoutDestroy(layout_c));
  hipblas_check(hipblasLtMatmulDescDestroy(desc));

  hipblas_check(hipblasLtDestroy(handle));

  hip_check(hipFree(d_a));
  hip_check(hipFree(d_b));
  hip_check(hipFree(d_c));
}


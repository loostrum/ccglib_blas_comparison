#include <iostream>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <ccglib/ccglib.hpp>

#include "config.h"

#define rocblas_check(status) {rocblas_assert((status), __FILE__, __LINE__);}
inline void rocblas_assert(rocblas_status status, const char *file, int line) {
  if (status != rocblas_status_success) {
    std::string err = std::string(file) + " line " + std::to_string(line) + " " + std::string(rocblas_status_to_string(status));
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

  rocblas_handle handle;
  rocblas_check(rocblas_create_handle(&handle));

  rocblas_operation trans_a = rocblas_operation_transpose;
  rocblas_operation trans_b = rocblas_operation_none;

  rocblas_datatype type_a = rocblas_datatype_f16_r;
  rocblas_datatype type_b = rocblas_datatype_f16_r;
  rocblas_datatype type_c = rocblas_datatype_f16_r;
  rocblas_datatype type_compute = rocblas_datatype_f16_r;

  rocblas_gemm_algo algo = rocblas_gemm_algo_standard;

  int solution_index = 0;
  unsigned flags = 0;

  rocblas_int m = params.M;
  rocblas_int n = params.N;
  rocblas_int k = params.K;

  rocblas_int lda = params.K;
  rocblas_int ldb = params.K;
  rocblas_int ldc = params.N;

  rocblas_float alpha{1};
  rocblas_float beta{0};

  rocblas_half *d_a;
  rocblas_half *d_b;
  rocblas_half *d_c;

  size_t bytes_a = params.M * params.K * sizeof(half);
  size_t bytes_b = params.N * params.K * sizeof(half);
  size_t bytes_c = params.M * params.N * sizeof(half);

  hip_check(hipMalloc(&d_a, bytes_a));
  hip_check(hipMalloc(&d_b, bytes_b));
  hip_check(hipMalloc(&d_c, bytes_c));

  hipEvent_t start, end;
  hip_check(hipEventCreate(&start));
  hip_check(hipEventCreate(&end));

  hip_check(hipEventRecord(start));
  rocblas_check(rocblas_gemm_ex(handle,
                                trans_a,
                                trans_b,
                                m, n, k,
                                &alpha, d_a, type_a, lda,
                                d_b, type_b, ldb,
                                &beta, d_c, type_c, ldc,
                                d_c, type_c, ldc,
                                type_compute, algo, solution_index, flags
                                ));
  hip_check(hipEventRecord(end));
  hip_check(hipEventSynchronize(end));
  hip_check(hipDeviceSynchronize());

  float runtime;
  hip_check(hipEventElapsedTime(&runtime, start, end));
  const double tflops = 2ULL * 1e-9 * params.M * params.N * params.K / runtime;

  std::cout << "Runtime: " << runtime << " ms" << std::endl;
  std::cout << "TFLOPS: " << tflops << std::endl;

  rocblas_check(rocblas_destroy_handle(handle));
  
  hip_check(hipFree(d_a));
  hip_check(hipFree(d_b));
  hip_check(hipFree(d_c));
}

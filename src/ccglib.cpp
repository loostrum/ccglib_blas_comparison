#include <iostream>

#include <cudawrappers/cu.hpp>
#include <ccglib/ccglib.hpp>

#include "config.h"

int main() {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  std::cout << "Running on " << device.getName() << " (" << device.getArch() << ")" << std::endl;

  constexpr size_t kComplex = 2;

  Params<ccglib::float16, ccglib::float32, large> params;

  constexpr size_t bytes_a = kComplex * params.B * params.M * params.K * sizeof(half);
  constexpr size_t bytes_b = kComplex * params.B * params.N * params.K * sizeof(half);
  constexpr size_t bytes_c = kComplex * params.B * params.M * params.N * sizeof(float);

  cu::HostMemory h_a(bytes_a);
  cu::HostMemory h_b(bytes_b);
  cu::HostMemory h_c(bytes_c);

  cu::DeviceMemory d_a(bytes_a);
  cu::DeviceMemory d_b(bytes_b);
  cu::DeviceMemory d_c(bytes_c);

  stream.memcpyHtoDAsync(d_a, h_a, bytes_a);
  stream.memcpyHtoDAsync(d_b, h_b, bytes_b);
  d_c.zero(bytes_c);

  cu::DeviceMemory d_a_trans(bytes_a);
  cu::DeviceMemory d_b_trans(bytes_b);

  dim3 dims = ccglib::mma::GEMM::GetDimensions(ccglib::float16, ccglib::mma::opt);

  ccglib::transpose::Transpose transpose_a(params.B, params.M, params.K, dims.x, dims.z, 
                                           ccglib::ValuePrecision{ccglib::float16}.GetBitWidth(), device, stream,
                                           ccglib::transpose::complex_middle);

  ccglib::transpose::Transpose transpose_b(params.B, params.N, params.K, dims.y, dims.z, 
                                           ccglib::ValuePrecision{ccglib::float16}.GetBitWidth(), device, stream,
                                           ccglib::transpose::complex_middle);

  ccglib::mma::GEMM gemm(params.B, params.M, params.N, params.K, device, stream,
                         ccglib::Precision(ccglib::float16, ccglib::float32),
                         ccglib::mma::opt, ccglib::mma::complex_middle,
                         ccglib::mma::row_major, ccglib::mma::row_major, ccglib::mma::col_major);

  cu::Event start, end_trans, end_gemm;
  stream.record(start);
  transpose_a.Run(d_a, d_a_trans);
  transpose_b.Run(d_b, d_b_trans);
  stream.record(end_trans);
  gemm.Run(d_a, d_b, d_c);
  stream.record(end_gemm);

  stream.memcpyDtoHAsync(h_c, d_c, bytes_c);
  stream.synchronize();

  const float runtime_trans = end_trans.elapsedTime(start);
  const float runtime_gemm = end_gemm.elapsedTime(end_trans);
  const float runtime_total = runtime_trans + runtime_gemm;

  const double tflops_gemm = 8ULL * 1e-9 * params.B * params.M * params.N * params.K / runtime_gemm;
  const double tflops_total = 8ULL * 1e-9 * params.B * params.M * params.N * params.K / runtime_total;

  std::cout << "Runtime: " << runtime_trans << " ms (transpose) + " << runtime_gemm << " ms (GEMM) = " << runtime_total << " ms (total)" << std::endl;
  std::cout << "TFLOPS: " << tflops_gemm << " (GEMM only), " << tflops_total << " (total)" << std::endl;
  
}

# ccglib vs (cu/roc)Blas

## cuBlas
GEMM is a cuBlas level 3 function.
the `cublas<t>gemm()` function requires all three matrices to have the same type, hence this is incompatible with ccglib.
The availble types are `float`, `double`, `cuComplex`, `cuDoubleComplex`, and `half`. I.e. no half precision complex.

`cublasGemmEx` is an extension of `cublas<t>gemm()`. This supports individual typing for A/B/C and compute. Supported type combinations listed at https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasGemmEx#cublasgemmex. The available complex types are `int8` (with `float` output), `float`, `double`. 

So cublas doesn't support the operations we want: half to float complex, or 1-bit to int complex.

`cuBlasLt` is the lightweight cuBlas focused on performant GEMMs. This interface does support the typing we want. The example in this repo uses the `cuBlasLt` interface. Even though the compute type is set to `float`, `ncu` shows that the half-precision tensor cores are used. It is not yet clear to me if any type casting happens.

## rocBlas
Like cuBlas, The rocblas level 3 functions required all matrices to have the same type. The available types are `float`, `double`, `half`, `complex float`, `complex double`. 
The `rocblas_gemm_ex` extension has more options, but no reduced precision complex types. There are beta functions in rocblas, but only for `float8`. Hence, `half` complex does not seem to be possible. The real `half` to `half` version is run to get at least a rough idea of the performance. 

## Results
Performance for `half` to `float` planar complex, batch size 1, matrices all 8192 by 8192 elements:

NVIDIA A100:
```
ccglib:
Runtime: 1.68474 ms (transpose) + 36.7135 ms (GEMM) = 38.3982 ms (total)
TFLOPS: 119.794 (GEMM only), 114.538 (total)

cublas:
Runtime: 37.5798 ms
TFLOPS: 117.032
```

AMD MI210:
```
ccglib:
Runtime: 1.34144 ms (transpose) + 29.5012 ms (GEMM) = 30.8427 ms (total)
TFLOPS: 149.08 (GEMM only), 142.596 (total)

rocblas (real half to real half):
Runtime: 23.5486 ms
TFLOPS: 46.6912
```

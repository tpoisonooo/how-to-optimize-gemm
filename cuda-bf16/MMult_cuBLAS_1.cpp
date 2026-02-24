#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA and CUBLAS functions
/* A: M x K, B: N x K, C: M x N */
/* Compute C = A * B^T */

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // cublasSgemm: C = alpha * op(A) * op(B) + beta * C
  // We need C = A * B^T
  // op(A) = A (M x K), op(B) = B^T (K x N), C (M x N)
  // cublas is column-major by default, but we use row-major interpretation
  // In row-major: C = A * B^T means:
  // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, ...)
  // Because cublas interprets matrices as column-major
  checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                              d_B, ldb, d_A, lda, &beta, d_C, ldc));
}

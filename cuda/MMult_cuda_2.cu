#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * naive 实现
 */
template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
  int _m = blockIdx.x * BLOCK + threadIdx.x;
  int _n = blockIdx.y * BLOCK + threadIdx.y;
  if (_m < m and _n < n) {
    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
      sum += a[_m * k + i] * b[i * n + _n];
    }
    c[_m * n + _n] = sum;
  }
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 16;
  // subm, subn, subk
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm<BLOCK><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}

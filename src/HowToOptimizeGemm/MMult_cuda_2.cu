#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper.h"

// CUDA and CUBLAS functions

__global__ void sgemm(int m, int n, int k, float* a, int lda, float* b, int ldb, float *c, int ldc) {
      int _m = blockIdx.x * 16 + threadIdx.x;
      int _n = blockIdx.y * 16 + threadIdx.y;
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

      // subm, subn, subk
      dim3 block(16, 16);
      dim3 grid((m+15)/16, (n+15)/16);

      sgemm<<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}

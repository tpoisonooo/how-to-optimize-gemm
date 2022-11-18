#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// a = mxk, b = kxn
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
  const int tx = (threadIdx.x % 16) * 2;
  const int ty = threadIdx.x / 16 * 2;
  const int bx = blockIdx.x * 64;
  const int by = blockIdx.y * 64;

  float *begin_a = a + by * k;
  float *begin_b = b + bx;
  float *end_a = begin_a + k;

  __shared__ float ashare[64][64];
  __shared__ float bshare[64][64];
  float sum0[2][2] = {0};
  float sum1[2][2] = {0};
  float sum2[2][2] = {0};
  float sum3[2][2] = {0};

  // bigger split
  for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += 64, b_ptr += 64 * n) {

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        ashare[ty + i][tx + j] = a_ptr[(ty + i) * k + tx + j];
        ashare[ty + i][tx + j + 32] = a_ptr[(ty + i) * k + tx + j + 32];
        ashare[ty + i + 32][tx + j] = a_ptr[(ty + 32 + i) * k + tx + j];
        ashare[ty + i + 32][tx + j + 32] =
            a_ptr[(ty + 32 + i) * k + tx + j + 32];

        bshare[ty + i][tx + j] = b_ptr[(ty + i) * n + tx + j];
        bshare[ty + i][tx + j + 32] = b_ptr[(ty + i) * n + tx + j + 32];
        bshare[ty + i + 32][tx + j] = b_ptr[(ty + i + 32) * n + tx + j];
        bshare[ty + i + 32][tx + j + 32] =
            b_ptr[(ty + i + 32) * n + tx + j + 32];
      }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int subk = 0; subk < 64; ++subk) {
          sum0[i][j] += ashare[ty + i][subk] * bshare[subk][tx + j];
          sum1[i][j] += ashare[ty + i][subk] * bshare[subk][tx + j + 32];
          sum2[i][j] += ashare[ty + i + 32][subk] * bshare[subk][tx + j];
          sum3[i][j] += ashare[ty + i + 32][subk] * bshare[subk][tx + j + 32];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      c[(by + ty + i) * n + bx + tx + j] = sum0[i][j];
      c[(by + ty + i) * n + bx + tx + 32 + j] = sum1[i][j];
      c[(by + ty + i + 32) * n + bx + tx + j] = sum2[i][j];
      c[(by + ty + i + 32) * n + bx + tx + 32 + j] = sum3[i][j];
    }
  }
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  dim3 block(256);
  dim3 grid(m / 64, n / 64);

  sgemm<<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}

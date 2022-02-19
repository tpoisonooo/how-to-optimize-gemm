#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// MY_MMult = [
// 1024 6467.51 7.247925e-05
// 2048 6693.74 1.525879e-04
// 3072 7096.70 2.288818e-04
// 4096 6677.67 4.425049e-04
// ];
/**
 * 和 version4 的区别：
 * 1. 修改了分块尺寸
 * 2. 每个 block 有 8x8 个线程，每个线程计算 4x4 个结果
 */
template <int BLOCK, int STRIDE>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
  // blockIdx control subpanel matrix
  constexpr int STEP = BLOCK * STRIDE;
  const int tx = threadIdx.x * STRIDE;
  const int ty = threadIdx.y * STRIDE;
  const int bx = blockIdx.x * STEP;
  const int by = blockIdx.y * STEP;

  float *begin_a = a + by * k;
  float *begin_b = b + bx;
  float *end_a = begin_a + k;

  float sum[STRIDE][STRIDE] = {0.f};
  for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += STEP, b_ptr += STEP * n) {
    __shared__ __align__(16 * 1024) float ashare[STEP][STEP];
    __shared__ __align__(16 * 1024) float bshare[STEP][STEP];

    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        ashare[ty + i][tx + j] = a_ptr[(ty + i) * k + tx + j];
        bshare[ty + i][tx + j] = b_ptr[(ty + i) * n + tx + j];
      }
    }
    __syncthreads();

    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        for (int kk = 0; kk < STEP; ++kk) {
          sum[i][j] += ashare[ty + i][kk] * bshare[kk][tx + j];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < STRIDE; ++i) {
    for (int j = 0; j < STRIDE; ++j) {
      c[(by + ty + i) * n + bx + tx + j] = sum[i][j];
    }
  }
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 8;
  constexpr int STRIDE = 4; // every thread calc STRIDExSTRIDE result
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);

  sgemm<BLOCK, STRIDE><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}

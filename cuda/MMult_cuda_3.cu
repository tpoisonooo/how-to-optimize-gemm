#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// a = mxk, b = kxn
template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, float *b, float *c) {
  // blockIdx control subpanel matrix

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  float *begin_a = a + bx * BLOCK * k;
  float *begin_b = b + by * BLOCK;
  float *end_a = begin_a + k;

  float sum = 0.f;

  for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += BLOCK, b_ptr += BLOCK * n) {

    __shared__ float ashare[BLOCK][BLOCK];
    __shared__ float bshare[BLOCK][BLOCK];

    ashare[ty][tx] = a_ptr[ty * k + tx];
    bshare[ty][tx] = b_ptr[ty * n + tx];
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BLOCK; ++kk) {
      sum += ashare[ty][kk] * bshare[kk][tx];
    }
    __syncthreads();
  }

  c[(BLOCK * bx + ty) * n + BLOCK * by + tx] = sum;
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 16;
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm<BLOCK><<<grid, block>>>(m, n, k, d_A, d_B, d_C);
}

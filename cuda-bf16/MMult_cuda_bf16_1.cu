#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float f) {
  return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 b) {
  return __bfloat162float(b);
}

// A: M x K, B: N x K, C: M x N
// Compute C = A * B^T
// Assume K and N are multiples of 8
template <int BLOCK>
__global__ void sgemm_bf16(int m, int n, int k, float *a, float *b, float *c) {
  // blockIdx control subpanel matrix

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  float *begin_a = a + bx * BLOCK * k;
  float *begin_b = b + by * BLOCK * k;
  float *end_a = begin_a + k;

  float sum = 0.f;

  for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += BLOCK, b_ptr += BLOCK) {

    __shared__ float ashare[BLOCK][BLOCK];
    __shared__ float bshare[BLOCK][BLOCK];

    ashare[ty][tx] = a_ptr[ty * k + tx];
    bshare[ty][tx] = b_ptr[ty * k + tx];
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BLOCK; ++kk) {
      // Convert to bf16 for computation
      __nv_bfloat16 a_bf16 = float_to_bf16(ashare[ty][kk]);
      __nv_bfloat16 b_bf16 = float_to_bf16(bshare[tx][kk]);
      sum += bf16_to_float(__hmul(a_bf16, b_bf16));
    }
    __syncthreads();
  }

  c[(BLOCK * bx + ty) * n + BLOCK * by + tx] = sum;
}

// A: M x K, B: N x K, C: M x N
// Compute C = A * B^T
void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 16;
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm_bf16<BLOCK><<<grid, block>>>(m, n, k, d_A, d_B, d_C);
}

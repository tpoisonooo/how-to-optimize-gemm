#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// a = mxk, b = kxn
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

  __shared__ __align__(16 * 1024) char smem[16 * 1024];
  float *ashare = reinterpret_cast<float *>(smem);
  float *bshare = reinterpret_cast<float *>(smem + 8 * 1024);

  float sum[2][STRIDE][STRIDE] = {0.f};
  for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += STEP, b_ptr += STEP * n) {

    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        ashare[(ty+i) * STEP + tx+j] = a_ptr[(ty+i) * k + tx + j];
        bshare[(ty+i) * STEP + tx+j] = b_ptr[(ty+i) * n + tx + j];
      }
    }
    __syncthreads();

    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        for (int kk = 0; kk < STEP; ++kk) {
          sum[i][j] += ashare[(ty+i) * STEP + kk] * bshare[kk * STEP + tx+j];
        }
      }
    }

    __syncthreads();
  }

    #pragma unroll
    for (int i = 0; i < STRIDE; ++i) {
      auto const addr = c + (by + ty + i) * n + bx + tx;
      asm volatile (
        "st.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(sum[i][0]), "f"(sum[i][1]), "f"(sum[i][2]), "f"(sum[i][3])
      );
    }
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 8;
  constexpr int STRIDE = 4; // every thread calc STRIDExSTRIDE result
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK /  STRIDE);

  sgemm<BLOCK, STRIDE><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}

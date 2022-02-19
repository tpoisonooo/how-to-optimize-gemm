#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define SMEM_LDA (128)
#define SMEM_LDB (128)

// GPU Device 0: "NVIDIA GeForce RTX 3080" with compute capability 8.6
// MY_MMult = [
// 1024 10968.46 7.247925e-05
// 2048 15003.43 1.525879e-04
// 3072 15738.94 2.288818e-04
// 4096 15814.25 4.425049e-04
// ];

/**
 * version 9 的特点是 gmem->smem 过程中用了 GPU 喜欢 interleave 的特性。
 *
 * 标准的 GEMM 里 matrixA 是要 transpose 的，thread 加载 gmem 的 4行1列
 * 个数据，放到 smem 里是 1x4，32x8 个线程加载 256x8 大小的 subA，变成  8x256
 *
 * matrixB 不 tranpose，就是单纯的加载。32 thread 合并访问。 thread i 访问  [i,
 * i+32, i+64, i+96]
 *
 */
__global__ __launch_bounds__(256, 2) void sgemm_128x128x8(int m, int n, int k,
                                                          const float *a,
                                                          const float *b,
                                                          float *c) {

  __shared__ __align__(
      16 * 1024) char smem[24 * 1024]; // 16KB shared memory for buffer
  float *ashare = reinterpret_cast<float *>(smem);
  float *bshare =
      reinterpret_cast<float *>(smem + 16 * 1024); // 8k shared mem for B
  float sum[8][8] = {0};
  float panelA[8] = {0}, panelB[8] = {0};

  int from_a = (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8;
  int from_b = (threadIdx.x / 32) * n + blockIdx.x * 128 + threadIdx.x % 32;

  for (int loop = 0; loop < k; loop += 8) {
    // part1: gmem to smem
    // load gmem to smem for ashare
    int to_a = (threadIdx.x % 8) * SMEM_LDA +
               (threadIdx.x / 8) * 4; // 连续的地址不能给同一个 thread 用
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ashare[to_a + i] = a[from_a + i * k];
    }

    // load gmem to smem for bshare
    int to_b = (threadIdx.x / 32) * SMEM_LDB + (threadIdx.x % 32);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      bshare[to_b + i * 32] =
          b[from_b + i * 32]; // 32 thread 合并访问。 thread i 访问  [i, i+32,
                              // i+64, i+96]
    }

    __syncthreads();
    from_a += 8;
    from_b += 8 * n;

    // part2: calculation
    // 计算 2x2 个 4x4
    int aidx0 = (threadIdx.x / 16) * 4;
    int bidx0 = (threadIdx.x % 16) * 4;
#pragma unroll
    for (int subk = 0; subk < 8; ++subk) {
      float *ptrA = ashare + aidx0 + subk * SMEM_LDA;

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        panelA[i] = ptrA[i];
        panelA[i + 4] = ptrA[i + 64];
      }

      const float *ptrB = bshare + bidx0 + subk * SMEM_LDB;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        panelB[i] = ptrB[i];
        panelB[i + 4] = ptrB[i + 64];
      }

#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          sum[i][j] += panelA[i] * panelB[j];
        }
      }
    }
    __syncthreads();
  }

  // part3: save to C
  int write_offset = (blockIdx.y * 128 + (threadIdx.x / 16) * 4) * n +
                     blockIdx.x * 128 + (threadIdx.x % 16) * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      c[write_offset + i * n + j] = sum[i][j];
      c[write_offset + i * n + j + 64] = sum[i][j + 4];
      c[write_offset + (i + 64) * n + j] = sum[i + 4][j];
      c[write_offset + (i + 64) * n + j + 64] = sum[i + 4][j + 4];
    }
  }
}

#undef SMEM_LDA
#undef SMEM_LDB

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 128;
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm_128x128x8<<<grid, 256>>>(m, n, k, d_A, d_B, d_C);
}

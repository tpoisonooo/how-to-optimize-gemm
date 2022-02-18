#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 256 threads  perblock, 2 blocks per multiprocessor
__launch_bounds__(256, 2)
__global__ void sgemm_128x128x8(int m, int n, int k, float *a, float *b, float *c) {

  #define SMEM_LDA (128)
  #define SMEM_LDB (128)
  __shared__  __align__(16 * 1024) char smem[16*1024]; // 16KB shared memory for buffer
  float* ashare = reinterpret_cast<float*>(smem);
  float* bshare = reinterpret_cast<float*>(smem + 8 * 1024);  // 8k shared mem for B

  float sum[8][8] = {0};
  const int block_offset_a = blockIdx.y * 128 * k + blockIdx.x * 128;
  int from_a = block_offset_a + (threadIdx.x / 8) * (4 * k) + (threadIdx.x % 8);

  const int block_offset_b = blockIdx.x * 128 * n + blockIdx.y * 128;
  int from_b = block_offset_b + (threadIdx.x / 32) * n + (threadIdx.x % 32);


  for (int loop = 0; loop < k; loop += 8) {
    // part1: gmem to smem
    // load gmem to smem for ashare
    const int to_a = (threadIdx.x % 8) * SMEM_LDA + (threadIdx.x / 8) * 4;  // 连续的地址不能给同一个 thread 用
    #pragma unroll
    for (int i =0; i < 4; ++i) {
      ashare[to_a+i] = a[from_a + i]; 
    }

    // load gmem to smem for bshare
    const int to_b = (threadIdx.x / 32) * SMEM_LDB + (threadIdx.x % 32);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      bshare[to_b + i * 32] = b[from_b + i * 32]; // 32 thread 合并访问。 thread i 访问  [i, i+32, i+64, i+96]
    }

    __syncthreads();
    from_a += 8;
    from_b += 8 * n;

    // part2: calculation
    // 计算 2x2 个 4x4
    int aidx0 = (threadIdx.x / 16) * 4;
    int bidx0 = (threadIdx.x % 16) * 4; 
    int aidx1 = aidx0 + 64;
    int bidx1 = bidx0 + 64;
    #pragma unroll
    for (int subk = 0; subk < 8; ++subk) {
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          sum[i][j] += ashare[aidx0 + i + subk * SMEM_LDA]* bshare[bidx0+j + subk * SMEM_LDB];
        }
      }
    }

    #pragma unroll
    for (int subk = 0; subk < 8; ++subk) {
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          sum[i][j+4] += ashare[aidx0 + i + subk * SMEM_LDA]* bshare[bidx1+j + subk * SMEM_LDB];
        }
      }
    }

    #pragma unroll
    for (int subk = 0; subk < 8; ++subk) {
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          sum[i+4][j] += ashare[aidx1 + i + subk * SMEM_LDA]* bshare[bidx0+j + subk * SMEM_LDB];
        }
      }
    }

    #pragma unroll
    for (int subk = 0; subk < 8; ++subk) {
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          sum[i+4][j+4] += ashare[aidx1 + i + subk * SMEM_LDA]* bshare[bidx1+j + subk * SMEM_LDB];
        }
      }
    }
  }

  #undef SMEM_LDA
  #undef SMEM_LDB

  // part3: save to C
  int write_offset = (blockIdx.y * 128 + (threadIdx.x % 16) * 4)* n + blockIdx.x * 128 + (threadIdx.x / 16) * 4;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      c[write_offset + i * n + j] = sum[i][j]; 
    }
  }
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      c[write_offset + i * n + j + 64] = sum[i][j+4]; 
    }
  }

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      c[write_offset + (i+64) * n + j] = sum[i+4][j]; 
    }
  }

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      c[write_offset + (i+64) * n + j + 64] = sum[i+4][j+4]; 
    }
  }
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 128;
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm_128x128x8<<<grid, 256>>>(m, n, k, d_A, d_B, d_C);
}

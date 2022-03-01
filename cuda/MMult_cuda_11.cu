#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
#include <vector>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define SMEM_LDA (132)
#define SMEM_LDB (128)
#define SMEM_LDC (64)

// remove original guard
__device__ __forceinline__ void ldg32_nc_0(float &reg, const void *ptr) {
  asm volatile("{.reg .pred p;\n"
               "mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 &&                 \
    __CUDA_ARCH__ >= 750
               "ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
               "ld.global.nc.f32 %0, [%1];}\n"
#endif
               : "=f"(reg)
               : "l"(ptr));
}

__device__ __forceinline__ uint32_t smem_u32addr(const void *smem_ptr) {
  uint32_t addr;
  asm("{.reg .u64 u64addr;\n"
      " cvta.to.shared.u64 u64addr, %1;\n"
      " cvt.u32.u64 %0, u64addr;}\n"
      : "=r"(addr)
      : "l"(smem_ptr));

  return addr;
}

__device__ __forceinline__ void lds128(float &reg0, float &reg1, float &reg2,
                                       float &reg3, const uint32_t &addr) {
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
               : "r"(addr));
}

__device__ __forceinline__ void sts128(const float &reg0, const float &reg1,
                                       const float &reg2, const float &reg3,
                                       const uint32_t &addr) {
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));
}

__device__ __forceinline__ void stg128(const float &reg0, const float &reg1,
                                       const float &reg2, const float &reg3,
                                       const float *addr) {
  asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};\n"
               :
               : "l"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));
}

__device__ __forceinline__ void stg32(const float &reg, const void *ptr) {
  asm volatile("{.reg .pred p;\n"
               " st.global.f32 [%0], %1;}\n"
               :
               : "l"(ptr), "f"(reg));
}

__device__ __forceinline__ void sts32(const float &reg, const uint32_t &addr) {
  asm volatile("st.shared.f32 [%0], %1;\n" : : "r"(addr), "f"(reg));
}

// MY_MMult = [
// 1024 12648.06 7.247925e-05
// 2048 16879.70 1.525879e-04
// 3072 17011.95 2.288818e-04
// 4096 16993.88 4.425049e-04
// ];
/**
 * version 11 相对于 version 10 的特点是
 * 1. 引入 gmem --- smem ping-pong， 没效果
 * 2. SMEM_LDA 改 132， 有效果.... **目前无法理解**
 * 3. 写回 C 矩阵时，引入 st128 方式，直接写入 global 地址，有效果。目前卡在
 * writeback 方式上
 * 4. 更进一步地，先借助  uint32_t 写入 smem，sync 一下，再写入  gmem.  无效果.
 * 5. ldgsts
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

  float a_ldg_reg[4], b_ldg_reg[4];

  uint32_t a_sts_addr = smem_u32addr(ashare + (threadIdx.x % 8) * SMEM_LDA +
                                     (threadIdx.x / 8) * 4);
  uint32_t b_sts_addr =
      smem_u32addr(bshare + (threadIdx.x / 32) * SMEM_LDB + (threadIdx.x % 32));

  uint32_t aptr_base = smem_u32addr(ashare + (threadIdx.x / 16) * 4);
  uint32_t bptr_base = smem_u32addr(bshare + (threadIdx.x % 16) * 4);

  {
// load first
// load gmem to smem for ashare
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ldg32_nc_0(a_ldg_reg[i],
                 (const char *)(a + from_a) + i * k * sizeof(float));
    }
    sts128(a_ldg_reg[0], a_ldg_reg[1], a_ldg_reg[2], a_ldg_reg[3], a_sts_addr);

// load gmem to smem for bshare
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ldg32_nc_0(b_ldg_reg[i],
                 (const char *)(b + from_b) + i * 32 * sizeof(float));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sts32(b_ldg_reg[i], b_sts_addr + i * 32 * sizeof(float));
    }
    __syncthreads();
    // add offset and flip flag
    from_a += 8;
    from_b += 8 * n;

    aptr_base ^= 0x2000;
    bptr_base ^= 0x1000;
    a_sts_addr ^= 0x2000;
    b_sts_addr ^= 0x1000;
  }

  for (int loop = 0; loop < k; loop += 8) {
    __syncthreads();
    if (loop < k - 8) {
      // if have more, load next
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        ldg32_nc_0(a_ldg_reg[i],
                   (const char *)(a + from_a) + i * k * sizeof(float));
      }
      sts128(a_ldg_reg[0], a_ldg_reg[1], a_ldg_reg[2], a_ldg_reg[3],
             a_sts_addr);
// load gmem to smem for bshare
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        ldg32_nc_0(b_ldg_reg[i],
                   (const char *)(b + from_b) + i * 32 * sizeof(float));
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        sts32(b_ldg_reg[i], b_sts_addr + i * 32 * sizeof(float));
      }

      from_a += 8;
      from_b += 8 * n;

      aptr_base ^= 0x2000;
      bptr_base ^= 0x1000;
      a_sts_addr ^= 0x2000;
      b_sts_addr ^= 0x1000;
    } else {
      aptr_base ^= 0x2000;
      bptr_base ^= 0x1000;
    }

    // calc
#pragma unroll
    for (int subk = 0; subk < 8; ++subk) {
      lds128(panelA[0], panelA[1], panelA[2], panelA[3],
             aptr_base + (subk * SMEM_LDA) * sizeof(float));
      lds128(panelA[4], panelA[5], panelA[6], panelA[7],
             aptr_base + (subk * SMEM_LDA + 64) * sizeof(float));

      lds128(panelB[0], panelB[1], panelB[2], panelB[3],
             bptr_base + (subk * SMEM_LDB) * sizeof(float));
      lds128(panelB[4], panelB[5], panelB[6], panelB[7],
             bptr_base + (subk * SMEM_LDB + 64) * sizeof(float));

#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          sum[i][j] += panelA[i] * panelB[j];
        }
      }
    }
  }

#if 1
  uint32_t c_sts_addr = smem_u32addr(reinterpret_cast<float *>(smem) +
                                     ((threadIdx.x / 16) * 4 * SMEM_LDC) +
                                     (threadIdx.x % 16) * 4);

  // 8x32
  float *C_lds_ptr =
      (float *)(smem) + (threadIdx.x / 32 * 8) * SMEM_LDC + (threadIdx.x % 32);

#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {

      __syncthreads();

#pragma unroll
      for (int p = 0; p < 4; ++p) {
        sts128(sum[i * 4 + p][j * 4], sum[i * 4 + p][j * 4 + 1],
               sum[i * 4 + p][j * 4 + 2], sum[i * 4 + p][j * 4 + 3],
               c_sts_addr + p * SMEM_LDC * sizeof(float));
      }
      __syncthreads();

      float *cptr = c + blockIdx.x * 128 + 64 * j +
                    (blockIdx.y * 128 + i * 64) * n +
                    (threadIdx.x / 32) * 8 * n + threadIdx.x % 32;

      for (int z = 0; z < 8; ++z) {
        stg32(C_lds_ptr[z * SMEM_LDC], cptr + z * n);
        stg32(C_lds_ptr[z * SMEM_LDC + 32], cptr + z * n + 32);
      }
    }
  }

#else

  int write_offset = (blockIdx.y * 128 + (threadIdx.x / 16) * 4) * n +
                     blockIdx.x * 128 + (threadIdx.x % 16) * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    stg128(sum[i][0], sum[i][1], sum[i][2], sum[i][3],
           c + write_offset + i * n);
    stg128(sum[i][4], sum[i][5], sum[i][6], sum[i][7],
           c + write_offset + i * n + 64);
    stg128(sum[i + 4][0], sum[i + 4][1], sum[i + 4][2], sum[i + 4][3],
           c + write_offset + (i + 64) * n);
    stg128(sum[i + 4][4], sum[i + 4][5], sum[i + 4][6], sum[i + 4][7],
           c + write_offset + (i + 64) * n + 64);
  }
#endif
}

#undef SMEM_LDA
#undef SMEM_LDB

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 128;
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm_128x128x8<<<grid, 256>>>(m, n, k, d_A, d_B, d_C);
}

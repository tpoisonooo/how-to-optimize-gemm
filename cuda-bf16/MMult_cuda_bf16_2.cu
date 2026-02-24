#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>

#define SMEM_LDA (132)
#define SMEM_LDB (128)

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float f) {
  return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 b) {
  return __bfloat162float(b);
}

// PTX inline assembly for optimized memory operations
__device__ __forceinline__ void ldg32_nc(float &reg, const void *ptr) {
  asm volatile("ld.global.nc.f32 %0, [%1];\n"
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
               : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));
}

__device__ __forceinline__ void stg128(const float &reg0, const float &reg1,
                                       const float &reg2, const float &reg3,
                                       const float *addr) {
  asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};\n"
               : : "l"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));
}

__device__ __forceinline__ void sts32(const float &reg, const uint32_t &addr) {
  asm volatile("st.shared.f32 [%0], %1;\n" : : "r"(addr), "f"(reg));
}

/**
 * BF16 SGEMM kernel optimized version (based on MMult_cuda_12.cu)
 * 
 * A (M x K), B (N x K), C = A * B^T
 * 
 * Assume K and N are multiples of 8
 * 
 * Tile size: 128x128
 * Each thread computes 8x8 results
 */
__global__ __launch_bounds__(256, 2) void sgemm_bf16_128x128x8(int m, int n, int k,
                                                                 const float *a,
                                                                 const float *b,
                                                                 float *c) {

  __shared__ __align__(16 * 1024) char smem[24 * 1024];
  float *ashare = reinterpret_cast<float *>(smem);
  float *bshare = reinterpret_cast<float *>(smem + 16 * 1024);
  
  float sum[8][8] = {0};
  float panelA[2][8] = {0}, panelB[2][8] = {0};

  // A: M x K, load 128 x 8 tile
  int from_a = (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8;
  
  // B: N x K, load 128 x 8 tile for B^T
  // Thread mapping: 128 x 2 (N x K_group)
  // n_idx: 0-127, k_group: 0-1 (each covers 4 K values)
  int n_idx = threadIdx.x / 2;         // 0-127
  int k_group = threadIdx.x % 2;       // 0 or 1
  int k_start = k_group * 4;           // 0 or 4
  int from_b = (blockIdx.x * 128 + n_idx) * k + k_start;

  float a_ldg_reg[4], b_ldg_reg[4];

  // Shared memory store addresses
  uint32_t a_sts_addr = smem_u32addr(ashare + (threadIdx.x % 8) * SMEM_LDA +
                                     (threadIdx.x / 8) * 4);
  // B: store at smem[k_start + i][n_idx]
  uint32_t b_sts_addr = smem_u32addr(bshare + k_start * SMEM_LDB + n_idx);

  // Shared memory load addresses for compute
  uint32_t aptr_base = smem_u32addr(ashare + (threadIdx.x / 16) * 4);
  uint32_t bptr_base = smem_u32addr(bshare + (threadIdx.x % 16) * 4);

  // Load first chunk
  {
    // Load A: 4 floats per thread (stride k in M dimension)
    ldg32_nc(a_ldg_reg[0], (const char *)(a + from_a));
    ldg32_nc(a_ldg_reg[1], (const char *)(a + from_a) + 1 * k * sizeof(float));
    ldg32_nc(a_ldg_reg[2], (const char *)(a + from_a) + 2 * k * sizeof(float));
    ldg32_nc(a_ldg_reg[3], (const char *)(a + from_a) + 3 * k * sizeof(float));
    sts128(a_ldg_reg[0], a_ldg_reg[1], a_ldg_reg[2], a_ldg_reg[3], a_sts_addr);

    // Load B: 4 consecutive elements in K dimension
    ldg32_nc(b_ldg_reg[0], (const char *)(b + from_b));
    ldg32_nc(b_ldg_reg[1], (const char *)(b + from_b) + 1 * sizeof(float));
    ldg32_nc(b_ldg_reg[2], (const char *)(b + from_b) + 2 * sizeof(float));
    ldg32_nc(b_ldg_reg[3], (const char *)(b + from_b) + 3 * sizeof(float));
    sts32(b_ldg_reg[0], b_sts_addr + 0 * SMEM_LDB * sizeof(float));
    sts32(b_ldg_reg[1], b_sts_addr + 1 * SMEM_LDB * sizeof(float));
    sts32(b_ldg_reg[2], b_sts_addr + 2 * SMEM_LDB * sizeof(float));
    sts32(b_ldg_reg[3], b_sts_addr + 3 * SMEM_LDB * sizeof(float));
    
    __syncthreads();
    
    from_a += 8;
    from_b += 8;
    a_sts_addr ^= 0x2000;
    b_sts_addr ^= 0x1000;
  }

  // Load first panel
  lds128(panelA[0][0], panelA[0][1], panelA[0][2], panelA[0][3], aptr_base);
  lds128(panelA[0][4], panelA[0][5], panelA[0][6], panelA[0][7],
         aptr_base + 64 * sizeof(float));
  lds128(panelB[0][0], panelB[0][1], panelB[0][2], panelB[0][3], bptr_base);
  lds128(panelB[0][4], panelB[0][5], panelB[0][6], panelB[0][7],
         bptr_base + 64 * sizeof(float));

  // Main loop
  for (int loop = 0; loop < k; loop += 8) {
#pragma unroll
    for (int subk = 0; subk < 8; ++subk) {

      if (subk == 7 && loop < k - 8) {
        sts128(a_ldg_reg[0], a_ldg_reg[1], a_ldg_reg[2], a_ldg_reg[3], a_sts_addr);
        sts32(b_ldg_reg[0], b_sts_addr + 0 * SMEM_LDB * sizeof(float));
        sts32(b_ldg_reg[1], b_sts_addr + 1 * SMEM_LDB * sizeof(float));
        sts32(b_ldg_reg[2], b_sts_addr + 2 * SMEM_LDB * sizeof(float));
        sts32(b_ldg_reg[3], b_sts_addr + 3 * SMEM_LDB * sizeof(float));
        __syncthreads();
        
        from_a += 8;
        from_b += 8;
        aptr_base ^= 0x2000;
        bptr_base ^= 0x1000;
        a_sts_addr ^= 0x2000;
        b_sts_addr ^= 0x1000;
      }

      const int pp = (subk + 1) % 2;
      lds128(panelA[pp][0], panelA[pp][1], panelA[pp][2], panelA[pp][3],
             aptr_base + ((subk + 1) % 8) * SMEM_LDA * sizeof(float));
      lds128(panelA[pp][4], panelA[pp][5], panelA[pp][6], panelA[pp][7],
             aptr_base + (((subk + 1) % 8) * SMEM_LDA + 64) * sizeof(float));
      lds128(panelB[pp][0], panelB[pp][1], panelB[pp][2], panelB[pp][3],
             bptr_base + ((subk + 1) % 8) * SMEM_LDB * sizeof(float));
      lds128(panelB[pp][4], panelB[pp][5], panelB[pp][6], panelB[pp][7],
             bptr_base + (((subk + 1) % 8) * SMEM_LDB + 64) * sizeof(float));

      if (subk == 0 && loop < k - 8) {
        ldg32_nc(a_ldg_reg[0], (const char *)(a + from_a));
        ldg32_nc(a_ldg_reg[1], (const char *)(a + from_a) + 1 * k * sizeof(float));
        ldg32_nc(a_ldg_reg[2], (const char *)(a + from_a) + 2 * k * sizeof(float));
        ldg32_nc(a_ldg_reg[3], (const char *)(a + from_a) + 3 * k * sizeof(float));
        ldg32_nc(b_ldg_reg[0], (const char *)(b + from_b));
        ldg32_nc(b_ldg_reg[1], (const char *)(b + from_b) + 1 * sizeof(float));
        ldg32_nc(b_ldg_reg[2], (const char *)(b + from_b) + 2 * sizeof(float));
        ldg32_nc(b_ldg_reg[3], (const char *)(b + from_b) + 3 * sizeof(float));
      }

      // Compute with BF16
#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          __nv_bfloat16 a_bf16 = float_to_bf16(panelA[subk % 2][i]);
          __nv_bfloat16 b_bf16 = float_to_bf16(panelB[subk % 2][j]);
          sum[i][j] += bf16_to_float(__hmul(a_bf16, b_bf16));
        }
      }
    }
  }

  // Store results
  int write_offset = (blockIdx.y * 128 + (threadIdx.x / 16) * 4) * n +
                     blockIdx.x * 128 + (threadIdx.x % 16) * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    stg128(sum[i][0], sum[i][1], sum[i][2], sum[i][3], c + write_offset + i * n);
    stg128(sum[i][4], sum[i][5], sum[i][6], sum[i][7], c + write_offset + i * n + 64);
    stg128(sum[i + 4][0], sum[i + 4][1], sum[i + 4][2], sum[i + 4][3],
           c + write_offset + (i + 64) * n);
    stg128(sum[i + 4][4], sum[i + 4][5], sum[i + 4][6], sum[i + 4][7],
           c + write_offset + (i + 64) * n + 64);
  }
}

#undef SMEM_LDA
#undef SMEM_LDB

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {
  constexpr int BLOCK = 128;
  dim3 grid((n + BLOCK - 1) / BLOCK, (m + BLOCK - 1) / BLOCK);
  sgemm_bf16_128x128x8<<<grid, 256>>>(m, n, k, d_A, d_B, d_C);
}

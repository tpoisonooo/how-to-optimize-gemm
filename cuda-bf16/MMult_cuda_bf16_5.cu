#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
#include <mma.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>

using namespace nvcuda;

/**
 * Optimized BF16 SGEMM kernel using Tensor Core
 * 
 * Improvements:
 * 1. Larger tile size: 128x128
 * 2. More warps: 8 warps (256 threads)
 * 3. Better warp layout: 4x2
 * 4. Each warp computes 32x64 using 2x8 m16n16 tiles
 * 
 * A (M x K), B (N x K), C = A * B^T
 * Inputs are bf16, output is f32
 */

__global__ void sgemm_bf16_tensorcore_optimized(int M, int N, int K,
                                                 const __nv_bfloat16 *a,
                                                 const __nv_bfloat16 *b,
                                                 float *c) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  
  // 4x2 warp layout
  const int warp_m = warp_id / 2;  // 0-3
  const int warp_n = warp_id % 2;  // 0-1
  
  const int block_m = blockIdx.y * 128;
  const int block_n = blockIdx.x * 128;
  
  // Shared memory for double buffering
  // A: 128 x 16 row-major
  // B: 16 x 128 col-major
  __shared__ __align__(16) __nv_bfloat16 smem_a[2][128 * 16];
  __shared__ __align__(16) __nv_bfloat16 smem_b[2][16 * 128];
  __shared__ __align__(16) float smem_c[128 * 128];
  
  // Each warp computes 32x64 output
  // Using 2x8 m16n16 tiles
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      wmma::fill_fragment(acc[i][j], 0.0f);
  
  // Load first tile
  int kk = 0;
  
  // Load A: 128 x 16
  // Each thread loads 8 elements
  for (int idx = threadIdx.x; idx < 128 * 16; idx += 256) {
    int row = idx / 16;
    int col = idx % 16;
    int g_row = block_m + row;
    int g_col = kk + col;
    
    if (g_row < M && g_col < K) {
      smem_a[0][row * 16 + col] = a[g_row * K + g_col];
    } else {
      smem_a[0][row * 16 + col] = __float2bfloat16(0.0f);
    }
  }
  
  // Load B: 16 x 128 (col-major)
  for (int idx = threadIdx.x; idx < 128 * 16; idx += 256) {
    int n = idx / 16;
    int k_idx = idx % 16;
    int g_n = block_n + n;
    int g_k = kk + k_idx;
    
    if (g_n < N && g_k < K) {
      smem_b[0][k_idx + n * 16] = b[g_n * K + g_k];
    } else {
      smem_b[0][k_idx + n * 16] = __float2bfloat16(0.0f);
    }
  }
  __syncthreads();
  
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
  
  // Main loop
  for (kk = 0; kk < K; kk += 16) {
    int buf = (kk / 16) % 2;
    int next_buf = 1 - buf;
    
    // Compute: each warp does 2x4 mma tiles for 32x64 output
    for (int i = 0; i < 2; ++i) {  // 2 tiles in M (32 total)
      for (int j = 0; j < 4; ++j) {  // 4 tiles in N (64 total)
        int a_row = warp_m * 32 + i * 16;
        int b_col = warp_n * 64 + j * 16;
        
        wmma::load_matrix_sync(a_frag, smem_a[buf] + a_row * 16, 16);
        wmma::load_matrix_sync(b_frag, smem_b[buf] + b_col * 16, 16);
        
        wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
      }
    }
    
    // Load next tile
    if (kk + 16 < K) {
      for (int idx = threadIdx.x; idx < 128 * 16; idx += 256) {
        int row = idx / 16;
        int col = idx % 16;
        int g_row = block_m + row;
        int g_col = kk + 16 + col;
        
        if (g_row < M && g_col < K) {
          smem_a[next_buf][row * 16 + col] = a[g_row * K + g_col];
        } else {
          smem_a[next_buf][row * 16 + col] = __float2bfloat16(0.0f);
        }
      }
      
      for (int idx = threadIdx.x; idx < 128 * 16; idx += 256) {
        int n = idx / 16;
        int k_idx = idx % 16;
        int g_n = block_n + n;
        int g_k = kk + 16 + k_idx;
        
        if (g_n < N && g_k < K) {
          smem_b[next_buf][k_idx + n * 16] = b[g_n * K + g_k];
        } else {
          smem_b[next_buf][k_idx + n * 16] = __float2bfloat16(0.0f);
        }
      }
      __syncthreads();
    }
  }
  
  // Store results
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      int c_row = warp_m * 32 + i * 16;
      int c_col = warp_n * 64 + j * 16;
      wmma::store_matrix_sync(smem_c + c_row * 128 + c_col, acc[i][j], 128, wmma::mem_row_major);
    }
  }
  __syncthreads();
  
  // Write to global memory
  for (int idx = threadIdx.x; idx < 128 * 128; idx += 256) {
    int r = idx / 128;
    int cc = idx % 128;
    int g_row = block_m + r;
    int g_col = block_n + cc;
    if (g_row < M && g_col < N) {
      c[g_row * N + g_col] = smem_c[r * 128 + cc];
    }
  }
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, __nv_bfloat16 *d_A, int lda,
              __nv_bfloat16 *d_B, int ldb, float *d_C, int ldc) {
  dim3 block(256);  // 8 warps
  dim3 grid((n + 127) / 128, (m + 127) / 128);
  sgemm_bf16_tensorcore_optimized<<<grid, block>>>(m, n, k, d_A, d_B, d_C);
}

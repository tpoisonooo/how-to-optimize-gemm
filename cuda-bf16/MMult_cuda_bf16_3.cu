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
 * BF16 SGEMM kernel using Tensor Core (wmma.m16n16k16)
 * 
 * A (M x K), B (N x K), C = A * B^T
 * 
 * wmma.m16n16k16:
 *   - A: 16 x 16 (row-major)
 *   - B: 16 x 16 (col-major)
 *   - C: 16 x 16 (row-major)
 * 
 * For C = A * B^T:
 *   - A is M x K row-major
 *   - B is N x K row-major, B^T is K x N
 *   - wmma B needs col-major, so we need K x N in col-major
 *   - K x N col-major = N x K row-major = B!
 *   - So B (N x K row-major) works directly for wmma B (col-major K x N)
 * 
 * Assume K and N are multiples of 16
 * 
 * Block: 64x64 tile with 128 threads (4 warps)
 * Each warp: 32x32 tile using 2x2 m16n16 tiles
 */

__global__ void sgemm_bf16_tensorcore(int M, int N, int K,
                                       const float *a,
                                       const float *b,
                                       float *c) {
  const int warp_id = threadIdx.x / 32;
  const int warp_m = warp_id / 2;
  const int warp_n = warp_id % 2;
  
  const int block_m = blockIdx.y * 64;
  const int block_n = blockIdx.x * 64;
  
  // Shared memory
  // A: 64 x 16 row-major (for wmma A row-major)
  // B: 16 x 64 col-major = 64 x 16 row-major (for wmma B col-major)
  __shared__ __align__(16) __nv_bfloat16 smem_a[2][64 * 16];
  __shared__ __align__(16) __nv_bfloat16 smem_b[2][16 * 64];  // K x N layout
  __shared__ __align__(16) float smem_c[64 * 64];
  
  // Accumulators
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      wmma::fill_fragment(acc[i][j], 0.0f);
  
  // Load first tile
  int kk = 0;
  
  // Load A to shared memory: 64 x 16 row-major
  for (int idx = threadIdx.x; idx < 64 * 16; idx += 128) {
    int row = idx / 16;
    int col = idx % 16;
    int g_row = block_m + row;
    int g_col = kk + col;
    
    if (g_row < M && g_col < K) {
      smem_a[0][row * 16 + col] = __float2bfloat16(a[g_row * K + g_col]);
    } else {
      smem_a[0][row * 16 + col] = __float2bfloat16(0.0f);
    }
  }
  
  // Load B to shared memory: 16 x 64 col-major
  // For col-major, element (k, n) is at k + n * 16
  // B[n][k] = b[n * K + k]
  // We want smem_b[k + n * 16] = B[n][k]
  for (int idx = threadIdx.x; idx < 64 * 16; idx += 128) {
    int n = idx / 16;  // 0-63
    int k_idx = idx % 16;  // 0-15
    int g_n = block_n + n;
    int g_k = kk + k_idx;
    
    if (g_n < N && g_k < K) {
      // smem_b is col-major K x N
      // smem_b[k + n * 16] = B[n][k]
      smem_b[0][k_idx + n * 16] = __float2bfloat16(b[g_n * K + g_k]);
    } else {
      smem_b[0][k_idx + n * 16] = __float2bfloat16(0.0f);
    }
  }
  __syncthreads();
  
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
  
  // Main loop over K
  for (kk = 0; kk < K; kk += 16) {
    int buf = (kk / 16) % 2;
    int next_buf = 1 - buf;
    
    // Each warp computes 2x2 mma tiles
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        // Load A from shared memory (row-major, leading dim 16)
        int a_row = warp_m * 32 + i * 16;
        wmma::load_matrix_sync(a_frag, smem_a[buf] + a_row * 16, 16);
        
        // Load B from shared memory (col-major, leading dim 16)
        // smem_b is K x N col-major
        // Element (k, n) is at k + n * 16
        // For tile starting at column b_col, offset is b_col * 16
        int b_col = warp_n * 32 + j * 16;
        wmma::load_matrix_sync(b_frag, smem_b[buf] + b_col * 16, 16);
        
        // MMA
        wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
      }
    }
    
    // Load next tile
    if (kk + 16 < K) {
      for (int idx = threadIdx.x; idx < 64 * 16; idx += 128) {
        int row = idx / 16;
        int col = idx % 16;
        int g_row = block_m + row;
        int g_col = kk + 16 + col;
        
        if (g_row < M && g_col < K) {
          smem_a[next_buf][row * 16 + col] = __float2bfloat16(a[g_row * K + g_col]);
        } else {
          smem_a[next_buf][row * 16 + col] = __float2bfloat16(0.0f);
        }
      }
      
      for (int idx = threadIdx.x; idx < 64 * 16; idx += 128) {
        int n = idx / 16;
        int k_idx = idx % 16;
        int g_n = block_n + n;
        int g_k = kk + 16 + k_idx;
        
        if (g_n < N && g_k < K) {
          smem_b[next_buf][k_idx + n * 16] = __float2bfloat16(b[g_n * K + g_k]);
        } else {
          smem_b[next_buf][k_idx + n * 16] = __float2bfloat16(0.0f);
        }
      }
      __syncthreads();
    }
  }
  
  // Store results to shared memory first
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      int c_row = warp_m * 32 + i * 16;
      int c_col = warp_n * 32 + j * 16;
      wmma::store_matrix_sync(smem_c + c_row * 64 + c_col, acc[i][j], 64, wmma::mem_row_major);
    }
  }
  __syncthreads();
  
  // Write to global memory
  for (int idx = threadIdx.x; idx < 64 * 64; idx += 128) {
    int r = idx / 64;
    int cc = idx % 64;
    int g_row = block_m + r;
    int g_col = block_n + cc;
    if (g_row < M && g_col < N) {
      c[g_row * N + g_col] = smem_c[r * 64 + cc];
    }
  }
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {
  dim3 block(128);
  dim3 grid((n + 63) / 64, (m + 63) / 64);
  sgemm_bf16_tensorcore<<<grid, block>>>(m, n, k, d_A, d_B, d_C);
}

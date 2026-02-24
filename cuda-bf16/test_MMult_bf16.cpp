#include <stdio.h>
#include "parameters.h"
#include <stdlib.h>
#include <string.h>
#include <cuda_bf16.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

void REF_MMult(int, int, int, float *, int, float *, int, float *, int);
void MY_MMult(cublasHandle_t, int, int, int, __nv_bfloat16 *, int, __nv_bfloat16 *, int,
              float *, int);
void copy_matrix(int, int, float *, int, float *, int);
void random_matrix(int, int, float *, int);
float compare_matrices(int, int, float *, int, float *, int);

double dclock();

// Helper to convert float array to bf16
void float_to_bf16_array(const float *src, __nv_bfloat16 *dst, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = __float2bfloat16(src[i]);
  }
}

int main() {
  // print gpu info
  cudaDeviceProp deviceProp;
  int devID = 0;
  checkCudaErrors(cudaSetDevice(devID));
  auto error = cudaGetDeviceProperties(&deviceProp, devID);
  if (error != cudaSuccess) {
    printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error,
           __LINE__);
    exit(EXIT_FAILURE);
  }
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
         deviceProp.name, deviceProp.major, deviceProp.minor);

  int p, m, n, k, rep;

  double dtime, dtime_best, gflops, diff;

  float *a, *b, *c, *cref, *cold;

  printf("MY_MMult = [\n");

  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));

  /* Time the "optimized" implementation */
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  for (p = PFIRST; p <= PLAST; p += PINC) {
    m = (M == -1 ? p : M);
    n = (N == -1 ? p : N);
    k = (K == -1 ? p : K);

    gflops = 2.0 * m * n * k * 1.0e-09;

    // A: M x K, B: N x K, C: M x N
    const int lda = k, ldb = k, ldc = n;

    /* Allocate space for the matrices */
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = n * k * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);
    const size_t mem_size_A_bf16 = m * k * sizeof(__nv_bfloat16);
    const size_t mem_size_B_bf16 = n * k * sizeof(__nv_bfloat16);
    
    a = (float *)malloc(mem_size_A);
    b = (float *)malloc(mem_size_B);
    c = (float *)malloc(mem_size_C);
    cold = (float *)malloc(mem_size_C);
    cref = (float *)malloc(mem_size_C);

    /* Generate random matrices A, B, Cold */
    random_matrix(m, k, a, lda);
    random_matrix(n, k, b, ldb);
    random_matrix(m, n, cold, ldc);
    memset(cold, 0, mem_size_C);
    memset(cref, 0, mem_size_C);

    /* Run the reference implementation so the answers can be compared */
    REF_MMult(m, n, k, a, lda, b, ldb, cref, ldc);

    // Convert A and B to bf16 on host
    __nv_bfloat16 *a_bf16 = (__nv_bfloat16 *)malloc(mem_size_A_bf16);
    __nv_bfloat16 *b_bf16 = (__nv_bfloat16 *)malloc(mem_size_B_bf16);
    float_to_bf16_array(a, a_bf16, m * k);
    float_to_bf16_array(b, b_bf16, n * k);

    /* Init device matrix with bf16 dtype */
    __nv_bfloat16 *d_A, *d_B;
    float *d_C;
    checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A_bf16));
    checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B_bf16));
    checkCudaErrors(cudaMemcpy(d_A, a_bf16, mem_size_A_bf16, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, b_bf16, mem_size_B_bf16, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    for (rep = 0; rep < NREPEATS; rep++) {
      /* Time your implementation */
      MY_MMult(handle, m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / NREPEATS;
    double flopsPerMatrixMul = 2.0 * m * k * n;
    double gflops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(cold, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    diff = compare_matrices(m, n, cold, ldc, cref, ldc);
    if (diff > 0.5f || diff < -0.5f) {
      printf("diff too big !\n");
      exit(-1);
    }
    printf("%d %.2f %le \n", p, gflops, diff);

    free(a);
    free(b);
    free(c);
    free(cold);
    free(cref);
    free(a_bf16);
    free(b_bf16);

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
  }

  checkCudaErrors(cublasDestroy(handle));

  printf("];\n");
  return 0;
}

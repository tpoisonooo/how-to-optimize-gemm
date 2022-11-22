#include <stdio.h>
// #include <malloc.h>
#include "parameters.h"
#include <stdlib.h>
#include <string.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

void REF_MMult(int, int, int, float *, int, float *, int, float *, int);
void MY_MMult(cublasHandle_t, int, int, int, float *, int, float *, int,
              float *, int);
void copy_matrix(int, int, float *, int, float *, int);
void random_matrix(int, int, float *, int);
float compare_matrices(int, int, float *, int, float *, int);

double dclock();

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
  // checkCudaErrors(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  /* Time the "optimized" implementation */
  cudaEvent_t start, stop;
  // Allocate CUDA events that we'll use for timing
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // printf( "create Handle\n");

  for (p = PFIRST; p <= PLAST; p += PINC) {
    m = (M == -1 ? p : M);
    n = (N == -1 ? p : N);
    k = (K == -1 ? p : K);

    gflops = 2.0 * m * n * k * 1.0e-09;

    const int lda = k, ldb = n, ldc = n;

    /* Allocate space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);
    a = (float *)malloc(mem_size_A);
    b = (float *)malloc(mem_size_B);
    c = (float *)malloc(mem_size_C);
    cold = (float *)malloc(mem_size_C);
    cref = (float *)malloc(mem_size_C);

    /* Generate random matrices A, B, Cold */
    random_matrix(m, k, a, m);
    random_matrix(k, n, b, k);
    random_matrix(m, n, cold, n);
    memset(cold, 0, mem_size_C);
    memset(cref, 0, mem_size_C);

    /* Init device matrix*/
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
    checkCudaErrors(cudaMemcpy(d_A, a, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, b, mem_size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

    /* Run the reference implementation so the answers can be compared */
    // printf( "init\n");

    REF_MMult(m, n, k, a, lda, b, ldb, cref, ldc);
    // printf( "benchmark\n");

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    for (rep = 0; rep < NREPEATS; rep++) {
      /* Time your implementation */
      MY_MMult(handle, m, n, k, d_A, k, d_B, n, d_C, n);
    }

    // printf( "mymmult\n");

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    // Wait for the stop event to complete
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

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
  }

  // Destroy the handle
  checkCudaErrors(cublasDestroy(handle));

  printf("];\n");
  return 0;
}

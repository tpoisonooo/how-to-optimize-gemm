#include <iostream>
// #include <malloc.h>
#include "parameters.h"
#include <cstdlib>
#include <cstring>
#define SPDLOG_ACTIVE_LEVEL 6

void REF_MMult(int, int, int, float *, float *, float *);
float MY_MMult(int, int, int, float *, float *, float *);
void copy_matrix(int, int, float *, float *);
void random_matrix(int, int, float *);
float compare_matrices(int, int, float *, float *);

double dclock();

int main() {
  int p, m, n, k;

  double diff;

  float *a, *b, *cref, *cold;

  printf("MY_MMult = [\n");

  for (p = PFIRST; p <= PLAST; p += PINC) {
    m = (M == -1 ? p : M);
    n = (N == -1 ? p : N);
    k = (K == -1 ? p : K);

    /* Allocate space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);
    constexpr size_t alignment = 64;
    a = (float*) std::aligned_alloc(alignment, mem_size_A * sizeof(float));
    b = (float*) std::aligned_alloc(alignment, mem_size_B * sizeof(float));
    cold = (float*) std::aligned_alloc(alignment, mem_size_C * sizeof(float));
    cref = (float*) std::aligned_alloc(alignment, mem_size_C * sizeof(float));

    /* Generate random matrices A, B, Cold */
    random_matrix(m, k, a);
    random_matrix(k, n, b);
    std::memset(cold, 0, mem_size_C);
    std::memset(cref, 0, mem_size_C);

    /* Run the reference implementation so the answers can be compared */
    REF_MMult(m, n, k, a, b, cref);

    float msecTotal = 0.0f;
    for (int rep = 0; rep < NREPEATS; rep++) {
      /* Time your implementation */
      msecTotal += MY_MMult(m, n, k, a, b, cold);
    }

    diff = compare_matrices(m, n, cold, cref);
    if (diff > 0.5f || diff < -0.5f) {
      fprintf(stdout, "diff too big !\n");
      exit(-1);
    }

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / NREPEATS;
    double flopsPerMatrixMul = 2.0 * m * k * n;
    double gflops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

    fprintf(stdout, "%d %.2f %le \n", p, gflops, diff);

    std::free(a);
    std::free(b);
    std::free(cold);
    std::free(cref);
  }

  fprintf(stdout, "];\n");
  return 0;
}

#include <stdio.h>
// #include <malloc.h>
#include "parameters.h"
#include <stdlib.h>
#include <string.h>

void REF_MMult(int, int, int, float *, int, float *, int, float *, int);
float MY_MMult(int, int, int, float *, float *, float *);
void copy_matrix(int, int, float *, int, float *, int);
void random_matrix(int, int, float *, int);
float compare_matrices(int, int, float *, int, float *, int);

double dclock();

int main() {
  int p, m, n, k, lda, ldb, ldc, rep;

  double dtime, dtime_best, gflops, diff;

  float *a, *b, *c, *cref, *cold;

  printf("MY_MMult = [\n");

  for (p = PFIRST; p <= PLAST; p += PINC) {
    m = (M == -1 ? p : M);
    n = (N == -1 ? p : N);
    k = (K == -1 ? p : K);

    gflops = 2.0 * m * n * k * 1.0e-09;

    lda = (LDA == -1 ? m : LDA);
    ldb = (LDB == -1 ? k : LDB);
    ldc = (LDC == -1 ? m : LDC);

    /* Allocate space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    const size_t mem_size_A = lda * (k + 1) * sizeof(float);
    const size_t mem_size_B = ldb * n * sizeof(float);
    const size_t mem_size_C = ldc * n * sizeof(float);
    a = (float *)malloc(mem_size_A);
    b = (float *)malloc(mem_size_B);
    c = (float *)malloc(mem_size_C);
    cold = (float *)malloc(mem_size_C);
    cref = (float *)malloc(mem_size_C);

    /* Generate random matrices A, B, Cold */
    random_matrix(m, k, a, lda);
    random_matrix(k, n, b, ldb);
    random_matrix(m, n, cold, ldc);
    memset(cold, 0, mem_size_C);
    memset(cref, 0, mem_size_C);

    /* Run the reference implementation so the answers can be compared */
    // printf( "init\n");

    REF_MMult(m, n, k, a, lda, b, ldb, cref, ldc);
    // printf( "benchmark\n");
    float avg_gflops = 0.0;
    for (rep = 0; rep < NREPEATS; rep++) {
      /* Time your implementation */
      avg_gflops = MY_MMult(m, n, k, a, b, c);
    }

    printf("%d %.2f %le \n", p, gflops, diff);

    free(a);
    free(b);
    free(c);
    free(cold);
    free(cref);
  }

  printf("];\n");
  return 0;
}

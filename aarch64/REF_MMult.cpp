/* Create macros so that the matrices are stored in row-major order */

#if 0
#include <cblas.h>
/* Routine for computing C = A * B + C */
void REF_MMult(int m, int n, int k, float *a, float *b, float *c) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k,
              b, n, 0.0f, c, n);
}

#else

#define A(i, j) a[(i) * k + (j)]
#define B(i, j) b[(i) * n + (j)]
#define C(i, j) c[(i) * n + (j)]
/* Routine for computing C = A * B + C */

void REF_MMult(int m, int n, int k, float *a, float *b, float *c) {
  int i, j, p;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      for (p = 0; p < k; p++) {
        C(i, j) += A(i, p) * B(p, j);
      }
    }
  }
}

#undef A
#undef B
#undef C
#endif

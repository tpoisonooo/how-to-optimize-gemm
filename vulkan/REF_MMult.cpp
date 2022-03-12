/* Create macros so that the matrices are stored in row-major order */
#define A(i, j) a[(i)*lda + (j)]
#define B(i, j) b[(i)*ldb + (j)]
#define C(i, j) c[(i)*ldc + (j)]

#include <cblas.h>
/* Routine for computing C = A * B + C */

void REF_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb,
               float *c, int ldc) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, lda,
              b, ldb, 0.0f, c, ldc);
}

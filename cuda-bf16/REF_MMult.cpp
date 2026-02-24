/* Create macros so that the matrices are stored in row-major order */
#define A(i, j) a[(i)*lda + (j)]
#define B(i, j) b[(i)*ldb + (j)]
#define C(i, j) c[(i)*ldc + (j)]

#include <cblas.h>
/* Routine for computing C = A * B^T + C */
/* A: M x K, B: N x K, C: M x N */

void REF_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb,
               float *c, int ldc) {
  // B is N x K, we need B^T which is K x N
  // cblas_sgemm: C = alpha * A * B + beta * C
  // CblasRowMajor: A is M x K, B^T is K x N, so we pass B as N x K with CblasTrans
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a, lda,
              b, ldb, 0.0f, c, ldc);
}

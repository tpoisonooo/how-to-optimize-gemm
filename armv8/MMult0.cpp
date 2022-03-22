/* Routine for computing C = A * B */

void MY_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb,
              float *c, int ldc) {
#define A(i, j) a[(i) * k + (j)]
#define B(i, j) b[(i) * n + (j)]
#define C(i, j) c[(i) * n + (j)]

  int i, j, p;

  for (i = 0; i < m; i++) {     /* Loop over the rows of C */
    for (j = 0; j < n; j++) {   /* Loop over the columns of C */
      for (p = 0; p < k; p++) { /* Update C( i,j ) with the inner
                                   product of the ith row of A and
                                   the jth column of B */
        C(i, j) = C(i, j) + A(i, p) * B(p, j);
      }
    }
  }
#undef A
#undef B
#undef C
}

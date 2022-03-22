/* Routine for computing C = A * B */

void AddDot(int, float *, float *, int, float *);

void MY_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb,
              float *c, int ldc) {
  int i, j;

#define A(i, j) a[(i) * k + (j)]
#define B(i, j) b[(i) * n + (j)]
#define C(i, j) c[(i) * n + (j)]

  for (j = 0; j < n; ++j) {   /* Loop over the columns of C */
    for (i = 0; i < m; ++i) { /* Loop over the rows of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
         and the jth column of B */

      AddDot(k, &A(i, 0), &B(0, j), ldb, &C(i, j));
    }
  }
#undef A
#undef B
#undef C
}

/* Create macro to let X( i ) equal the ith element of x */
void AddDot(int k, float *x, float *y, int ldb, float *gamma) {
  /* compute gamma := x' * y + gamma with vectors x and y of length n.

     Here x starts at location x with increment (stride) incx and y starts at
     location y and has (implicit) stride of 1.
  */
  for (int p = 0; p < k; p++) {
    *gamma += x[p] * y[p * ldb];
  }
}

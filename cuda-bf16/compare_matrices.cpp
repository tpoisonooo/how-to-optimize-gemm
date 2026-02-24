#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float compare_matrices(int m, int n, float *a, int lda, float *b, int ldb) {
  int i, j;
  float max_diff = 0.0, diff;

  for (j = 0; j < n; j++)
    for (i = 0; i < m; i++) {
      diff = fabs(a[j * lda + i] - b[j * lda + i]);
      max_diff = (diff > max_diff ? diff : max_diff);
    }

  return max_diff;
}

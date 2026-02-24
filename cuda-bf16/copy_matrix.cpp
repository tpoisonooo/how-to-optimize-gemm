#include <stdlib.h>

void copy_matrix(int m, int n, float *a, int lda, float *b, int ldb) {
  int i, j;

  for (j = 0; j < n; j++)
    for (i = 0; i < m; i++)
      b[j * ldb + i] = a[j * lda + i];
}

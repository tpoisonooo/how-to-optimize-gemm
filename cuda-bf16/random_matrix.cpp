#include <stdlib.h>
#include <time.h>

void random_matrix(int m, int n, float *a, int lda) {
  // srand(time(NULL));
  srand(2024);
  int i, j;

  for (j = 0; j < n; j++)
    for (i = 0; i < m; i++)
      a[j * lda + i] = 2.0 * (float)drand48() - 1.0;
}

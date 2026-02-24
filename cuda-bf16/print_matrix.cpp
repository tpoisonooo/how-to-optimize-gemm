#include <stdio.h>
#include <stdlib.h>

void print_matrix(int m, int n, float *a, int lda) {
  int i, j;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++)
      printf("%le ", a[j * lda + i]);
    printf("\n");
  }
  printf("\n");
}

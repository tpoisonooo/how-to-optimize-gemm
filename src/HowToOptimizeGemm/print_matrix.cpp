#include <stdio.h>

#define A(i, j) a[(i)*lda + (j)]

void print_matrix(int m, int n, float *a, int lda) {
  int i, j;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%.1f\t", A(i, j));
    }
    printf("\n");
  }
  printf("\n");
}

#define abs(x) ((x) < 0.0 ? -(x) : (x))

#include <stdio.h>

float compare_matrices(int m, int n, float *a, float *b) {
#define A(i, j) a[(i) * n + (j)]
#define B(i, j) b[(i) * n + (j)]
  //    printf("\n---result----\n");
  //    print_matrix(m, n, a, lda);
  //    printf("\n-------\n");
  //    print_matrix(m, n, b, ldb);
  //    printf("\n-------\n");
  int i, j;
  float max_diff = 0.0, diff;
  int printed = 0;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      diff = abs(A(i, j) - B(i, j));
      max_diff = (diff > max_diff ? diff : max_diff);
      if (0 == printed)
        if (max_diff > 0.5f || max_diff < -0.5f) {
          fprintf(stdout, "error: i %d  j %d diff %f  got %f  expect %f \n", i,
                  j, max_diff, A(i, j), B(i, j));
          printed = 1;
        }
    }
  }

  return max_diff;
#undef A
#undef B
}

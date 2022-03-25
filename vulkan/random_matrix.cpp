#include <stdlib.h>

void random_matrix(int m, int n, float *a) {
#define A(i, j) a[(i)*n + (j)]

  double drand48();
  int i, j;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
#if 1
      A(i, j) = 2.0 * (float)drand48() - 1.0;
#else
      A(i, j) = (j - i) % 3;
#endif
      // A(i, j) = 1;
    }
  }
#undef A
}

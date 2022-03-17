void copy_matrix(int m, int n, float *a, float *b) {
#define A(i, j) a[(i)*n + (j)]
#define B(i, j) b[(i)*n + (j)]

  int i, j;

  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      B(i, j) = A(i, j);
    }
  }

#undef A
#undef B
}

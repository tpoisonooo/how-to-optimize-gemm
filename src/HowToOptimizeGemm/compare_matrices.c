#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define abs( x ) ( (x) < 0.0 ? -(x) : (x) )

#include <stdio.h>

float compare_matrices( int m, int n, float *a, int lda, float *b, int ldb )
{
//    printf("\n---result----\n");
//    print_matrix(m, n, a, lda);
//    printf("\n-------\n");
//    print_matrix(m, n, b, ldb);
//    printf("\n-------\n");
  int i, j;
  float max_diff = 0.0, diff;
  int printed = 0;

  for ( i=0; i<m; i++ ){
    for ( j=0; j<n; j++ ){
      diff = abs( A( i,j ) - B( i,j ) );
      max_diff = ( diff > max_diff ? diff : max_diff );
      if(0 == printed)
      if(max_diff > 0.5f || max_diff < -0.5f) {
        printf("\n error: i %d  j %d diff %f", i, j, max_diff);
        printed = 1;
      }
    }
  }

  return max_diff;
}


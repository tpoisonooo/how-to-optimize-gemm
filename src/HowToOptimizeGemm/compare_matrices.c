#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]

#include <stdlib.h>
#include <stdio.h>

void print_int32_matrix( int m, int n, int32_t *a, int lda );
int32_t compare_matrices( int m, int n, int32_t *a, int lda, int32_t *b, int ldb )
{
//printf("\n---result----\n");
//print_int32_matrix(m, n, a, lda);
//printf("\n---baseline----\n");
//print_int32_matrix(m, n, b, ldb);
//printf("\n-------\n");
  int i, j;
  int max_diff = 0, diff;
  int printed = 0;

  for ( i=0; i<m; i++ ){
    for ( j=0; j<n; j++ ){
      diff = abs( A( i,j ) - B( i,j ) );
      max_diff = ( diff > max_diff ? diff : max_diff );
      if(0 == printed)
      if(max_diff != 0) {
        printf("\n error: i %d  j %d diff %d", i, j, max_diff);
        printed = 1;
      }
    }
  }

  return max_diff;
}


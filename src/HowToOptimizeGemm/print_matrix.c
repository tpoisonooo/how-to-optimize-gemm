#include <stdio.h>
#include <stdlib.h>

#define A( i, j ) a[ (i)*lda + (j) ]

void print_int8_matrix( int m, int n, int8_t *a, int lda )
{
  int i, j;

  for ( i=0; i<m; i++ ){
      for ( j=0; j<n; j++ ) {
        printf("%d\t", A( i,j ) );
      }
    printf("\n");
  }
  printf("\n");
}

void print_int32_matrix( int m, int n, int32_t *a, int lda )
{
  int i, j;

  for ( i=0; i<m; i++ ){
      for ( j=0; j<n; j++ ) {
        printf("%d\t", A( i,j ) );
      }
    printf("\n");
  }
  printf("\n");
}

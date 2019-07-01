#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void copy_int8_matrix( int m, int n, int8_t *a, int lda, int8_t *b, int ldb )
{
  int i, j;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ )
      B( i,j ) = A( i,j );
}

void copy_int32_matrix( int m, int n, int32_t*a, int lda, int32_t *b, int ldb )
{
  int i, j;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ )
      B( i,j ) = A( i,j );
}

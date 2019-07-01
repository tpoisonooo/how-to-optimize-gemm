/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */
#include <stdlib.h>

void REF_MMult( int m, int n, int k, int8_t *a, int lda, 
                                     int8_t *b, int ldb,
                                     int32_t *c, int ldc )
{
  int i, j, p;

  for ( i=0; i<m; i++ ){
    for ( j=0; j<n; j++ ){
      for ( p=0; p<k; p++ ){
        C( i,j ) = C( i,j ) +  A( i,p ) * B( p,j );
      }
    }
  }
}


  

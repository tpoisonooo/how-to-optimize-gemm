#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parameters.h"


#define A( i,j ) a[ (i)*lda + (j) ]

void random_int8_matrix(int m, int n, int8_t *a, int lda)
{
  double drand48();
  int i,j;
  int val = 0;
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
#if 0 
      a[i * lda + j] = (int8_t)val;
      val ++;      
#else
      a[i * lda + j] = (int8_t)(val % 3);
      val ++;
#endif
    }
  }
}

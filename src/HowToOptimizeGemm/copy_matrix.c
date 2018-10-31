#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]

void copy_matrix( int m, int n, float *a, int lda, float *b, int ldb )
{
  int i, j;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ )
      B( i,j ) = A( i,j );
}


#include <stdio.h>
// #include <malloc.h>
#include <stdlib.h>
#include <string.h>

#include "parameters.h"

void REF_MMult(int, int, int, int8_t*, int, int8_t *, int, int32_t *, int );
void MY_MMult(int, int, int, int8_t *, int, int8_t *, int, int32_t *, int );
void copy_int8_matrix(int, int, int8_t *, int, int8_t *, int );
void copy_int32_matrix(int, int, int32_t *, int, int32_t *, int );
void random_int8_matrix(int, int, int8_t *, int);
int32_t compare_matrices( int, int, int32_t *, int, int32_t *, int );

double dclock();

int main()
{
  int 
    p, 
    m, n, k,
    lda, ldb, ldc, 
    rep;

  double 
    dtime, dtime_best,        
    gflops;

  int32_t
    diff;

  int8_t 
    *a, *b;

  int32_t
    *c, *cref, *cold;    
  
  printf( "MY_MMult = [\n" );
    
  for ( p=PFIRST; p<=PLAST; p+=PINC ){
    m = ( M == -1 ? p : M );
    n = ( N == -1 ? p : N );
    k = ( K == -1 ? p : K );

    gflops = 2.0 * m * n * k * 1.0e-09;

    lda = ( LDA == -1 ? m : LDA );
    ldb = ( LDB == -1 ? k : LDB );
    ldc = ( LDC == -1 ? m : LDC );

    /* Allocate space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    a = ( int8_t * ) malloc( m * k * sizeof( int8_t ) );  
    b = ( int8_t * ) malloc( k * n * sizeof( int8_t ) );
    c = ( int32_t * ) malloc( m * n * sizeof( int32_t ) );
    cold = ( int32_t * ) malloc( m * n * sizeof( int32_t ) );
    cref = ( int32_t * ) malloc( m * n * sizeof( int32_t ) );

    /* Generate random matrices A, B, Cold */
    random_int8_matrix( m, k, a, lda );
    random_int8_matrix( k, n, b, ldb );
#if 1 
    memset(cold, 0, m * n * sizeof(int32_t));
#endif

    copy_int32_matrix( m, n, cold, ldc, cref, ldc );

    /* Run the reference implementation so the answers can be compared */

    REF_MMult( m, n, k, a, lda, b, ldb, cref, ldc );

    /* Time the "optimized" implementation */
    for ( rep=0; rep<NREPEATS; rep++ ){
      copy_int32_matrix( m, n, cold, ldc, c, ldc );

      /* Time your implementation */
      dtime = dclock();

      MY_MMult( m, n, k, a, lda, b, ldb, c, ldc );
      
      dtime = dclock() - dtime;

      if ( rep==0 )
        dtime_best = dtime;
      else
	    dtime_best = ( dtime < dtime_best ? dtime : dtime_best );
    }

    diff = compare_matrices( m, n, c, ldc, cref, ldc );
    if(diff != 0){
        exit(0);
    }

    printf( "%d %le %d \n", p, gflops / dtime_best, diff );
    fflush( stdout );

    free( a );
    free( b );
    free( c );
    free( cold );
    free( cref );
  }

  printf( "];\n" );

  exit( 0 );
}


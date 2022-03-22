#include "parameters.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

void REF_MMult(int, int, int, float *, float *, float *);
float MY_MMult(int, int, int, float *, int, float *, int, float *, int);
void copy_matrix(int, int, float *, float *);
void random_matrix(int, int, float *);
float compare_matrices(int, int, float *, float *);

double dclock();

int main()
{
  int 
    p, 
    m, n, k,
    rep;

  double 
    dtime, dtime_best,        
    gflops, 
    diff;

  float 
    *a, *b, *c, *cref, *cold;    
  
  printf( "MY_MMult = [\n" );
    
  for ( p=PFIRST; p<=PLAST; p+=PINC ){
    m = ( M == -1 ? p : M );
    n = ( N == -1 ? p : N );
    k = ( K == -1 ? p : K );

    gflops = 2.0 * m * n * k * 1.0e-09;

    /* Allocate space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);
    constexpr size_t alignment = 64;
    a = (float *)std::aligned_alloc(alignment, mem_size_A * sizeof(float));
    b = (float *)std::aligned_alloc(alignment, mem_size_B * sizeof(float));
    cold = (float *)std::aligned_alloc(alignment, mem_size_C * sizeof(float));
    cref = (float *)std::aligned_alloc(alignment, mem_size_C * sizeof(float));

    /* Generate random matrices A, B, Cold */
    random_matrix(m, k, a);
    random_matrix(k, n, b);
    std::memset(cold, 0, mem_size_C);
    std::memset(cref, 0, mem_size_C);

    /* Run the reference implementation so the answers can be compared */

    REF_MMult(m, n, k, a, b, cref);

    const int lda = k, ldb = n, ldc = n;
    /* Time the "optimized" implementation */
    for ( rep=0; rep<NREPEATS; rep++ ){
      copy_matrix( m, n, cold, c);

      /* Time your implementation */
      dtime = dclock();

      MY_MMult( m, n, k, a, lda, b, ldb, c, ldc );
      
      dtime = dclock() - dtime;

      if ( rep==0 )
        dtime_best = dtime;
      else
	    dtime_best = ( dtime < dtime_best ? dtime : dtime_best );
    }

    diff = compare_matrices(m, n, cold, cref);
    if (diff > 0.5f || diff < -0.5f) {
      fprintf(stdout, "%d diff too big: %le\n", p, diff);
      exit(-1);
    }

    printf( "%d %le %le \n", p, gflops / dtime_best, diff );
    fflush( stdout );

    std::free(a);
    std::free(b);
    std::free(cold);
    std::free(cref);
  }

  printf( "];\n" );

  exit( 0 );
}


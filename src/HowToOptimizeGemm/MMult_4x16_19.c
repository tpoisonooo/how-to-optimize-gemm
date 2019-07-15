#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error("only support arm neon")
#endif

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Block sizes */
#define DEBUG_PACK_SHAPE
#undef DEBUG_PACK_SHAPE
#define DEBUG_PRINT_A
#define DEBUG_PRINT_B
#define DEBUG_PRINT_C
#undef DEBUG_PRINT_B
#undef DEBUG_PRINT_A
#undef DEBUG_PRINT_C
#undef DEBUG_PRINT_DATA

/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define min(i, j) ((i) < (j) ? (i): (j))

/**
Target: 24gflops on RK3399
*/
//#define GEMM_N (384)  // GEMM_R
//#define GEMM_M (4096)  // GEMM_P
//#define GEMM_K (256)  // GEMM_Q
#define GEMM_N (384)  // GEMM_R
#define GEMM_M (4096)  // GEMM_P
#define GEMM_K (256)  // GEMM_Q
#define GEMM_UNROLL_K (16)
#define GEMM_UNROLL_M (4)
#define GEMM_UNROLL_N (4)
#define KERNEL_4x16  kernel_4x16

double dclock();
void print_int8_matrix( int m, int n, int8_t *a, int lda);
void print_int32_matrix( int m, int n, int32_t *a, int lda);
/* Routine for computing C = A * B + C */
void packN_16(int k, int n, int8_t* from, int ldb, int8_t* to);
void packZ_16(int m, int k, int8_t* from, int lda, int8_t* to);
void kernel_4x16(int m, int n, int k, 
        int8_t* sa, int8_t* sb, int32_t* sc, int ldc);


int8_t* fastMalloc(int size){
    void* ptr = 0;
    int iRet = posix_memalign(&ptr, 16, size * sizeof(int8_t));
    assert(0 == iRet);
    return ptr;
}

/* Suppose that m%4==0 and n%4==0 and k%4==0, avoiding process boundary !! */
void MY_MMult(int m, int n, int k, int8_t * restrict a, int lda,
                                   int8_t * restrict b, int ldb,
                                   int32_t * restrict c, int ldc,
                                   double *packZ_cost,
                                   double *packN_cost,
                                   double *kernel_cost)
{
#if (defined DEBUG_PRINT_A) || (defined DEBUG_PRINT_B || defined DEBUG_PRINT_C)
    printf("\n--- a ----\n");
    print_int8_matrix(m, k, a, lda);
    printf("\n--- b ----\n");
    print_int8_matrix(k, n, b, ldb);
    printf("\n-------\n");
#endif
    int8_t* restrict sa = fastMalloc(m * k);
    int8_t* restrict sb = fastMalloc(k * n);
    double temp;

#if 1 
    int ms, mms, ns, ks;
    int min_m, min_mm, min_n, min_k;
    int l1stride = 1;
    for (ms = 0; ms < m; ms += GEMM_M) {
        min_m = m - ms;
        if (min_m > GEMM_M) {
            min_m = GEMM_M;
        }

        for (ks = 0; ks < k; ks += min_k){
            min_k = k - ks;
            if (min_k >= (GEMM_K << 1)) {
                min_k = GEMM_K;
            } else if (min_k > GEMM_K) {
                min_k = (min_k / 2 + GEMM_UNROLL_K - 1) & ~(GEMM_UNROLL_K - 1);
            }

            // first packB
            min_n = n;
            if (n >= GEMM_N * 2) {
                min_n = GEMM_N;
            } else if(n > GEMM_N) {
                min_n = (min_n / 2 + GEMM_UNROLL_N - 1) & ~(GEMM_UNROLL_N - 1);
            } else {
                l1stride = 1;
            }

            temp = dclock();
            packN_16(min_k, min_n, b + ks * ldb, ldb, sb);
            (*packN_cost) += (dclock() - temp);

#ifdef DEBUG_PRINT_B
            printf("\n ----- sb -- k n offset -- %d %d %d \n", min_k, min_n, ks * ldb);
            print_int8_matrix(m, k, sb, ldb);
#endif

            // micro kernel, split A Block to smaller Panel
            for (mms = ms; mms < ms + min_m; mms += min_mm) {
                min_mm = (ms + min_m) - mms;
                if (min_mm >= 3 * GEMM_UNROLL_M) {
                    min_mm = 3 * GEMM_UNROLL_M;
                } else if(min_mm >= 2 * GEMM_UNROLL_M) {
                    min_mm = 2 * GEMM_UNROLL_M;
                } else if(min_mm > GEMM_UNROLL_M) {
                    min_mm = GEMM_UNROLL_M;
                }

                // coninueous packA
                temp = dclock();
                packZ_16(min_mm, min_k, a + mms * lda + ks, lda, sa + min_k * (mms - ms) * l1stride);
                (*packZ_cost) += (dclock() - temp);

#ifdef DEBUG_PRINT_A
                printf("\n ----- sa --m k  %d %d-- \n", mms - ms, min_k);
                print_int8_matrix(m, k, sa, lda);
#endif

                temp = dclock();
                KERNEL_4x16(min_mm, min_n, min_k, sa + l1stride * min_k * (mms - ms), sb, c + mms * ldc, ldc);
                (*kernel_cost) += (dclock() - temp);

#ifdef DEBUG_PRINT_C
                printf("\n---first kernel----\n");
                print_int32_matrix(m, n, c, ldc);
#endif
            }

            // the first B Block has been packed, proc the others 
            for (ns = min_n; ns < n; ns += min_n) {
                min_n = n - ns;
                if (min_n >= GEMM_N * 2) {
                    min_n = GEMM_N; 
                } else if(min_n > GEMM_N) {
                    min_n = (min_n / 2 + GEMM_UNROLL_N - 1) & ~(GEMM_UNROLL_N - 1);
                }

                temp = dclock();
                packN_16(min_k, min_n, b + ns + ldb * ks, ldb, sb);
                (*packN_cost) += (dclock() - temp);

#ifdef DEBUG_PRINT_B
                printf("\n ----- sb -- k n offset -- %d %d %d\n", min_k, min_n, ks * ldb + ns);
                print_int8_matrix(m, k, sb, ldb);
#endif
                KERNEL_4x16(min_m, min_n, min_k, sa, sb, c + ms * ldc + ns, ldc);

#ifdef DEBUG_PRRINT_C
                printf("\n----second kernel---\n");
                print_int32_matrix(m, n, c, ldc);
#endif
            }
        }
    }
#endif

    free(sa);
    free(sb);
}

/**

int8_t* a: A
int8_t* b: (B)T
int8_t* c: C

C = A * (B)T

A1 A2 A3    B1 B4 B7
A4 A5 A6  x B2 B5 B8 => C1 C4 C7 C2 C5 C8 C3 C6 C9 (packed)
A7 A8 A9    B3 B4 B9

Calculation sequence:
1st. calculate C1
2st. calculate C4
3st. calculate C7
...
9st. calculate C9

A1-A9/B1-B9 is packed block, not single number.
C1-C9 is 4x4 block, not single number.

Output
C1 C2 C3
C4 C5 C6
C7 C8 C9

 */
// step 1. 8x8 calculation passed
// step 2. mxn passed
// step 3. assembly

void kernel_sub_v1(int m, int n, int k, int8_t *sa, int8_t *sb, int32_t *sc, int ldc) {
    int8_t *a = sa;
    int32_t *c = sc;
    for (int i = 0; i < m; ++i) {
        int8_t *b = sb; 

        for (int j = 0; j < n; ++j) {
            
            for (int x = 0; x < k; ++x) {
                c[j] += (int32_t)(a[x]) * b[x];
            }
            b += k;            
        }
        a += k;
        c += ldc;
    } 
}

extern void kernel_sub_m4n4k16(int8_t *sa, int8_t *sb, int32_t *sc, int ldc);

static void kernel_sub_v2(int m, int n, int k, int8_t *sa, int8_t *sb, int32_t *sc, int ldc) {
    if (4 == m && 4 == n && k==16) {
        kernel_sub_m4n4k16(sa, sb, sc, ldc * sizeof(ldc));
        return;
    }

    int8_t *a = sa;
    int32_t *c = sc;
    for (int i = 0; i < m; ++i) {
        int8_t *b = sb; 

        for (int j = 0; j < n; ++j) {
            
            for (int x = 0; x < k; ++x) {
                c[j] += (int32_t)(a[x]) * b[x];
            }
            b += k;            
        }
        a += k;
        c += ldc;
    } 
}

// get c[m, n] output
static void kernel_mn(int m, int n, int k, int8_t *sa, int8_t *sb, int32_t *sc, int ldc) {
    //sum_all( A4xsubk * Bsubkx4 )
    int8_t *a = sa, *b = sb;
    int shift = 4;
    while (k > 0) {
        int repeat = k >> shift;
        int step = 1 << shift;
        for (int i = 0; i < repeat; ++i) {
            kernel_sub_v2(m, n, step, a, b, sc, ldc);
            a += m * step;
            b += n * step; 
        }
        k -= (repeat << shift);
        shift--;
    }
}

// proc m lines
void kernel_m(int m, int n, int k, int8_t *sa, int8_t *sb, int32_t *sc, int ldc) {
    // m == 4
    int nn = n;
    int8_t *b = sb;
    int32_t *c = sc;

    while(nn >= 4) {
        kernel_mn(m, 4, k, sa, b, c, ldc);
        b += 4 * k;
        c += 4;        
        nn -= 4;
    };

    while(nn >= 2) {
        kernel_mn(m, 2, k, sa, b, c, ldc);
        b += 2 * k;
        c += 2;        
        nn -= 2;
    };

    while(nn >= 1) {
        kernel_mn(m, 1, k, sa, b, c, ldc);
        b += 1 * k;
        c += 1;
        nn -= 1;
    }
}

void kernel_4x16(int m, int n, int k,
        int8_t* sa, int8_t* sb, int32_t* sc, int ldc){
    int mm = m;
    int8_t *a = sa;
    int32_t *c = sc;
    while(mm >= 4){
        kernel_m(4, n, k, a, sb, c, ldc);
        a += 4 * k;
        c += 4 * ldc;
        mm -= 4;
    };

    while(mm >= 2){
        kernel_m(2, n, k, a, sb, c, ldc);
        a += 2 * k;
        c += 2 * ldc;
        mm -= 2;
    };

    while(mm >= 1){
        kernel_m(1, n, k, a, sb, c, ldc);
        a += k;
        c += ldc;
        mm--;
    };
}

static void packZ_sub(int8_t *from, int lda, int8_t * restrict to, int m, int n, int repeat) {
    int8_t *ptr;
    for (int r = 0; r < repeat; ++r) {
        ptr = from;
        for (int i = 0; i < m; ++i) {
            memcpy(to, ptr, n * sizeof(int8_t));
            to += n;
            ptr += lda;
        }
        from += n;
    }
}

// pack4x16
void packZ_16(int m, int k, int8_t* from, int lda, int8_t* to) {
#ifdef DEBUG_PACK_SHAPE
    printf("\n packZ_16, m=%d, k=%d", m, k);
#endif
    // TODO to be optimize
    int i, repeat;
    int col, proc_col;
    int8_t *a_offset = from;
    int8_t *col_offset = a_offset;
    int8_t *row_offset = col_offset;
    int8_t *c = to;

    int shift = 2;
    while (m > 0) {
        assert(shift >= 0);
        i = m >> shift;
        const int proc_row = 1 << shift;
        while (i > 0) {
            col = k;
            col_offset = row_offset;

            // proc 16x col
            repeat = col >> 4;
            proc_col = repeat << 4;

            packZ_sub(col_offset, lda, c, proc_row, 16, repeat);
            col_offset += proc_col;
            c += proc_row * proc_col;
            col -= proc_col;

            // proc 8x col
            repeat = col >> 3;
            proc_col = repeat << 3;

            packZ_sub(col_offset, lda, c, proc_row, 8, repeat);
            col_offset += proc_col;
            c += proc_row * proc_col;
            col -= proc_col;

            // proc 4x col
            repeat = col >> 2;
            proc_col = repeat << 2;

            packZ_sub(col_offset, lda, c, proc_row, 4, repeat); 
            col_offset += proc_col;
            c += proc_row * proc_col;
            col -= proc_col;

            // proc 2x col
            repeat = col >> 1;
            proc_col = repeat << 1;

            packZ_sub(col_offset, lda, c, proc_row, 2, repeat);
            col_offset += proc_col;
            c += proc_row * proc_col;
            col -= proc_col;

            // prco 1x col
            packZ_sub(col_offset, lda, c, proc_row, 1, col);
            col_offset += col;
            c += proc_row * col;

            row_offset += proc_row * lda;
            --i;
        };
        a_offset += ((m >> shift) << shift) * lda;
        row_offset = a_offset;
        m -= ((m >> shift) << shift);
        --shift;
    }
}

void packN_sub(int8_t * restrict from, int ldb, int8_t * restrict to, int m, int n, int repeat) {
    int8_t *ctemp[16] = {0};
    
    for (int r = 0; r < repeat; ++r) {

        if (m == 1) {
            ctemp[0] = from;
        } else if (m == 2) {
            ctemp[0] = from;
            ctemp[1] = from + ldb;
        } else if (m == 4) {
            ctemp[0] = from;
            ctemp[1] = from + ldb;
            ctemp[2] = from + 2 * ldb;
            ctemp[3] = from + 3 * ldb;
        } else if (m == 8) {
            ctemp[0] = from;
            ctemp[1] = from + ldb;
            ctemp[2] = from + 2 * ldb;
            ctemp[3] = from + 3 * ldb;
            ctemp[4] = from + 4 * ldb;
            ctemp[5] = from + 5 * ldb;
            ctemp[6] = from + 6 * ldb;
            ctemp[7] = from + 7 * ldb;
        } else if (m == 16) {
            ctemp[0] = from;
            ctemp[1] = from + ldb;
            ctemp[2] = from + 2 * ldb;
            ctemp[3] = from + 3 * ldb;
            ctemp[4] = from + 4 * ldb;
            ctemp[5] = from + 5 * ldb;
            ctemp[6] = from + 6 * ldb;
            ctemp[7] = from + 7 * ldb;
            ctemp[8] = from + 8 * ldb;
            ctemp[9] = from + 9 * ldb;
            ctemp[10] = from + 10 * ldb;
            ctemp[11] = from + 11 * ldb;
            ctemp[12] = from + 12 * ldb;
            ctemp[13] = from + 13 * ldb;
            ctemp[14] = from + 14 * ldb;
            ctemp[15] = from + 15 * ldb;
        } else {
            assert(0);
        }

        for (int i = 0; i < n; ++i) {
            if (m == 1) {
                to[0] = ctemp[0][i];
            } else if (m == 2) {
                to[0] = ctemp[0][i];
                to[1] = ctemp[1][i];
            } else if (m == 4) {
                to[0] = ctemp[0][i];
                to[1] = ctemp[1][i];
                to[2] = ctemp[2][i];
                to[3] = ctemp[3][i];
            } else if (m == 8) {
                to[0] = ctemp[0][i];
                to[1] = ctemp[1][i];
                to[2] = ctemp[2][i];
                to[3] = ctemp[3][i];
                to[4] = ctemp[4][i];
                to[5] = ctemp[5][i];
                to[6] = ctemp[6][i];
                to[7] = ctemp[7][i];
            } else if (m == 16) {
                to[0] = ctemp[0][i];
                to[1] = ctemp[1][i];
                to[2] = ctemp[2][i];
                to[3] = ctemp[3][i];
                to[4] = ctemp[4][i];
                to[5] = ctemp[5][i];
                to[6] = ctemp[6][i];
                to[7] = ctemp[7][i];
                to[8] = ctemp[8][i];
                to[9] = ctemp[9][i];
                to[10] = ctemp[10][i];
                to[11] = ctemp[11][i];
                to[12] = ctemp[12][i];
                to[13] = ctemp[13][i];
                to[14] = ctemp[14][i];
                to[15] = ctemp[15][i];
            } else {
                assert(0);
            }
            to += m;
        }
        from += ldb * m;
    }
}

// pack16x4
void packN_16(int k, int n, int8_t* from, int ldb, int8_t* to) {
#ifdef DEBUG_PACK_SHAPE
    printf("\n packN_16, k=%d, n=%d", k, n);
#endif

    int i, repeat;
    int row;
    int proc_row;

    int8_t *a_offset = from;
    int8_t *a_offset1 = a_offset;
    int8_t *c = to;

    int shift = 2;
    while (n > 0) {
        assert(shift >= 0);
        i = n >> shift;
        const int proc_col = 1 << shift;
        while (i > 0) {
            row = k;

            // proc 16x row
            repeat = row >> 4;
            proc_row = repeat << 4;
            packN_sub(a_offset1, ldb, c, 16, proc_col, repeat);
            a_offset1 += proc_row * ldb;
            c += proc_row * proc_col;
            row -= proc_row;

            // proc 8x row
            repeat = row >> 3;
            proc_row = repeat << 3;
            packN_sub(a_offset1, ldb, c, 8, proc_col, repeat);
            a_offset1 += proc_row * ldb;
            c += proc_row * proc_col;
            row -= proc_row;

            // proc 4x row
            repeat = row >> 2;
            proc_row = repeat << 2;
            packN_sub(a_offset1, ldb, c, 4, proc_col, repeat); 
            a_offset1 += proc_row * ldb;
            c += proc_row * proc_col;
            row -= proc_row;

            // proc 2x row
            repeat = row >> 1;
            proc_row = repeat << 1;
            packN_sub(a_offset1, ldb, c, 2, proc_col, repeat);
            a_offset1 += proc_row * ldb;
            c += proc_row * proc_col;
            row -= proc_row;

            // proc 1x row
            repeat = row;
            proc_row = repeat;
            packN_sub(a_offset1, ldb, c, 1, proc_col, repeat);
            a_offset1 += proc_row * ldb;
            c += proc_col * row;
            row -= proc_row;

            --i;
            a_offset += proc_col;
            a_offset1 = a_offset;
        };
        n -= ((n >> shift) << shift);
        --shift;
    }
}

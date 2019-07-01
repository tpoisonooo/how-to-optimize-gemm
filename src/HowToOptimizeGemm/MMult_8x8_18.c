#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error("only support arm neon")
#endif

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* Block sizes */
#define DEBUG_PACK_SHAPE
#undef DEBUG_PACK_SHAPE
#define DEBUG_PRINT_A
#undef DEBUG_PRINT_B
#undef DEBUG_PRINT_A
// #undef DEBUG_PRINT_DATA

/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define min(i, j) ((i) < (j) ? (i): (j))

/**
Target: 24gflops on RK3399
*/
#define GEMM_N (12)  // GEMM_R
#define GEMM_M (12)  // GEMM_P
#define GEMM_K (12)  // GEMM_Q
#define GEMM_UNROLL (8)
#define KERNEL_8x8  kernel_8x8

/* Routine for computing C = A * B + C */
void packN_8(int k, int n, int8_t* from, int ldb, int8_t* to);
void packZ_8(int m, int k, int8_t* from, int lda, int8_t* to);
void kernel_8x8(int m, int n, int k, 
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
                                   int8_t * restrict c, int ldc )
{
#if (defined DEBUG_PRINT_A) || (defined DEBUG_PRINT_B)
    printf("\n--- a ----\n");
    print_int8_matrix(m, k, a, lda);
    printf("\n--- b ----\n");
    print_int8_matrix(k, n, b, ldb);
    printf("\n-------\n");
#endif
    int8_t* restrict sa = fastMalloc(m * k);
    int8_t* restrict sb = fastMalloc(k * n);

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
                min_k = (min_k / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
            }

            // first packB
            min_n = n;
            if (n >= GEMM_N * 2) {
                min_n = GEMM_N;
            } else if(n > GEMM_N) {
                min_n = (min_n / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
            } else {
                l1stride = 1;
            }
            packN_8(min_k, min_n, b + ks * ldb, ldb, sb);
#ifdef DEBUG_PRINT_B
            printf("\n ----- sb -- k n offset -- %d %d %d \n", min_k, min_n, ks * ldb);
            print_int8_matrix(m, k, sb, ldb);
#endif

            // micro kernel, split A Block to smaller Panel
            for (mms = ms; mms < ms + min_m; mms += min_mm) {
                min_mm = (ms + min_m) - mms;
                if (min_mm >= 3 * GEMM_UNROLL) {
                    min_mm = 3 * GEMM_UNROLL;
                } else if(min_mm >= 2 * GEMM_UNROLL) {
                    min_mm = 2 * GEMM_UNROLL;
                } else if(min_mm > GEMM_UNROLL) {
                    min_mm = GEMM_UNROLL;
                }

                // coninueous packA
                packZ_8(min_mm, min_k, a + mms * lda + ks, lda, sa + min_k * (mms - ms) * l1stride);
#ifdef DEBUG_PRINT_A
                printf("\n ----- sa --m k  %d %d-- \n", mms - ms, min_k);
                print_int8_matrix(m, k, sa, lda);
#endif

                // KERNEL_8x8(min_mm, min_n, min_k, sa + l1stride * min_k * (mms - ms), sb, c + mms * ldc, ldc);
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
                    min_n = (min_n / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
                }

                packN_8(min_k, min_n, b + ns + ldb * ks, ldb, sb);
#ifdef DEBUG_PRINT_B
                printf("\n ----- sb -- k n offset -- %d %d %d\n", min_k, min_n, ks * ldb + ns);
                print_int8_matrix(m, k, sb, ldb);
#endif
                // KERNEL_8x8(min_m, min_n, min_k, sa, sb, c + ms * ldc + ns, ldc);

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
void kernel_8x8(int m, int n, int k, 
        int8_t* sa, int8_t* sb, int32_t* sc, int ldc){
}

void packZ_sub(int8_t *from, int lda, int8_t *to, int m, int n, int repeat) {
    for (int r = 0; r < repeat; ++r) {
        for (int i = 0; i < m; ++i) {
            memcpy(to, from, n * sizeof(int8_t));
            to += n;
            from += lda;
        }    
    }
}

void packZ_8(int m, int k, int8_t* from, int lda, int8_t* to) {
#ifdef DEBUG_PACK_SHAPE
    printf("\n packZ_8, m=%d, k=%d", m, k);
#endif
    // TODO to be optimize
    int i, repeat;
    int col, proc_col;
    int8_t * a_offset = from;
    int8_t * a_offset1 = a_offset;

    int shift = 3;
    while (m > 0) {
        assert(shift >= 0);
        i = m >> shift;
        const int proc_row = i << shift;
        while (i > 0) {
            col = k;

            // proc 8x col
            repeat = col >> 3;
            proc_col = repeat << 3;

            packZ_sub(a_offset1, lda, to, proc_row, 8, repeat);
            a_offset1 += proc_col;
            to += proc_row * proc_col;
            col -= proc_col;

            // proc 4x col
            repeat = col >> 2;
            proc_col = repeat << 2;

            packZ_sub(a_offset1, lda, to, proc_row, 4, repeat); 
            a_offset1 += proc_col;
            to += proc_row * proc_col;
            col -= proc_col;

            // proc 2x col
            repeat = col >> 1;
            proc_col = repeat << 1;

            packZ_sub(a_offset1, lda, to, proc_row, 2, repeat);
            a_offset1 += proc_col;
            to += proc_row * proc_col;
            col -= proc_col;


            // prco 1x col
            packZ_sub(a_offset1, lda, to, proc_row, 1, col);
            a_offset1 += col;
            to += proc_row * col;

            --i;
        };
        a_offset += ((m >> shift) << shift) * lda;
        a_offset1 = a_offset;
        m -= ((m >> shift) << shift);
        --shift;
    }

}

void packN_sub(int8_t * restrict from, int ldb, int8_t * restrict to, int m, int n, int repeat) {
    int8_t *ctemp[8] = {0};
    
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
            } else {
                assert(0);
            }
            to += m;
        }
        from += ldb * m;
    }
}

void packN_8(int k, int n, int8_t* from, int ldb, int8_t* to) {
#ifdef DEBUG_PACK_SHAPE
    printf("\n packN_8, k=%d, n=%d", k, n);
#endif

    int i, repeat;
    int row;
    int proc_row;

    int8_t *a_offset = from;
    int8_t *a_offset1 = a_offset;

    int shift = 3;
    while (n > 0) {
        assert(shift >= 0);
        i = n >> shift;
        const int proc_col = i << shift;
        while (i > 0) {
            row = k;

            // proc 8x row
            repeat = row >> 3;
            proc_row = repeat << 3;
            packN_sub(a_offset1, ldb, to, 8, proc_col, repeat);
            a_offset1 += proc_row * ldb;
            to += proc_row * proc_col;
            row -= proc_row;

            // proc 4x row
            repeat = row >> 2;
            proc_row = repeat << 2;
            packN_sub(a_offset1, ldb, to, 4, proc_col, repeat); 
            a_offset1 += proc_row * ldb;
            to += proc_row * proc_col;
            row -= proc_row;

            // proc 2x row
            repeat = row >> 1;
            proc_row = repeat << 1;
            packN_sub(a_offset1, ldb, to, 2, proc_col, repeat);
            a_offset1 += proc_row * ldb;
            to += proc_row * proc_col;
            row -= proc_row;

            // proc 1x row
            repeat = row;
            proc_row = repeat;
            packN_sub(a_offset1, ldb, to, 1, proc_col, repeat);
            a_offset1 += proc_row * ldb;
            to += proc_col * row;
            row -= proc_row;

            --i;
        };
        a_offset += ((n >> shift) << shift);
        a_offset1 = a_offset;
        n -= ((n >> shift) << shift);
        --shift;
    }
}

#if 0
    i = m >> 3;
    while(i > 0){
    // proc 8x row
        --i;
        kk = k;

        // proc 8x col
        j = kk >> 3;
        packZ_sub(from + a_offset1, lda, to, 8, 8);
        a_offset1 += 8;
        to += 64;
        kk -= (8*j);

        // proc 4x col
        j = kk >> 2;
        packZ_sub(from + a_offset1, lda, to, 8, 4); 
        a_offset1 += 4;
        to += 32;
        kk -= (4*j);

        // proc 2x col
        j = kk >> 1;
        packZ_sub(from + a_offset1, lda, to, 8, 2);
        a_offset1 += 2;
        to += 16;
        kk -= (2*j);

        j = kk;
        packZ_sub(from + a_offset1, lda, to, 8, 1);
        a_offset1 += 1;
        to += 8;
        kk -= (j);
    }
    m -= (8*i);
    a_offset = from + 8 * i * lda;
    a_offset1 = a_offset;

    i = m >> 2;
    while(i > 0){
    // proc 4x row
        --i;
        kk = k;

        // proc 8x col
        j = kk >> 3;
        packZ_sub(from + a_offset1, lda, to, 4, 8);
        a_offset1 += 8;
        to += 32;
        kk -= (8*j);

        // proc 4x col
        j = kk >> 2;
        packZ_sub(from + a_offset1, lda, to, 4, 4); 
        a_offset1 += 4;
        to += 16;
        kk -= (4*j);

        // proc 2x col
        j = kk >> 1;
        packZ_sub(from + a_offset1, lda, to, 4, 2);
        a_offset1 += 2;
        to += 8;
        kk -= (2*j);

        j = kk;
        packZ_sub(from + a_offset1, lda, to, 4, 1);
        a_offset1 += 1;
        to += 4;
        kk -= j;
    }
    m -= (4*i);
    a_offset = from + 4 * i * lda;
    a_offset1 = a_offset;

    i = m >> 1;
    while(i > 0){
    // proc 2x row
        --i;
        kk = k;

        // proc 8x col
        j = kk >> 3;
        packZ_sub(from + a_offset1, lda, to, 2, 8);
        a_offset1 += 8;
        to += 16;
        kk -= (8*j);

        // proc 4x col
        j = kk >> 2;
        packZ_sub(from + a_offset1, lda, to, 2, 4); 
        a_offset1 += 4;
        to += 8;
        kk -= (4*j);

        // proc 2x col
        j = kk >> 1;
        packZ_sub(from + a_offset1, lda, to, 2, 2);
        a_offset1 += 2;
        to += 4;
        kk -= (2*j);

        j = kk;
        packZ_sub(from + a_offset1, lda, to, 2, 1);
        a_offset1 += 1;
        to += 2;
        kk -= j;
    }
    m -= (2*i);
    a_offset = from + 2 * i * lda;
    a_offset1 = a_offset;

    i = m;
    while(i > 0){
    // proi 1x line
        --i;
        kk = k;

        // proc 8x col
        j = kk >> 3;
        packZ_sub(from + a_offset1, lda, to, 1, 8);
        a_offset1 += 8;
        to += 8;
        kk -= (8*j);

        // proc 4x col
        j = kk >> 2;
        packZ_sub(from + a_offset1, lda, to, 1, 4); 
        a_offset1 += 4;
        to += 4;
        kk -= (4*j);

        // proc 2x col
        j = kk >> 1;
        packZ_sub(from + a_offset1, lda, to, 1, 2);
        a_offset1 += 2;
        to += 2;
        kk -= (2*j);

        j = kk;
        packZ_sub(from + a_offset1, lda, to, 1, 1);
        a_offset1 += 1;
        to += 8;
        kk -= j;
    }
    m -= (i);
    a_offset = from + i * lda;
    a_offset1 = a_offset;
#endif

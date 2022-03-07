#include <arm_neon.h>

#include <assert.h>
#include <stdlib.h>

/* Block sizes */
#define DEBUG_PACK_SHAPE
#undef DEBUG_PACK_SHAPE
#define DEBUG_PRINT_DATA
#undef DEBUG_PRINT_DATA

/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define min(i, j) ((i) < (j) ? (i): (j))

/**
About GEMM_K or kc:
1. mc = kc, since we have to maxmize (2 * mc * kc/(2 * mc + kc))
2. The equation exists provided kc << n.
3. mc * kc <= K

About GEMM_M or mc:
1. The larger mc * nc, the better calculation efficiency
2. We prepare to load A into L2 cache. Avoiding TLB miss (which would
stall CPU), subset of A should remains so until no longer needed.

About KENEL_4x4, mr=4 and nr=4
In order to move data efficiently to the registers.
Here we use C_block = A_panel x Transpose(B_panel)

In accordance to page.14 "6. MOE DETAILS YET",




L1d cahce = 32K, and L2 cache = 2MB. `getconf -a | grep PAGESIZE` = 4096.
Thus L1d is not the Cannikin, it is constraint to page size.

min_nn * kc <= PAGESIZE/2,  4 <= min_nn <= 12, so that 170 <= kc <= 512, we use 256.
After reading 6.4, rk3399 L2 cache is large, mc = 1MB / 256 = 4096 


*/
#define GEMM_N (384)  // GEMM_R
#define GEMM_M (4096)  // GEMM_P
#define GEMM_K (256)  // GEMM_Q
#define GEMM_UNROLL (4)
#define KERNEL_4x4  kernel_4x4_v3

float* fastMalloc(int size){
    void* ptr = 0;
    int iRet = posix_memalign(&ptr, 32, size * sizeof(float));
    assert(0 == iRet);
    return (float*)ptr;
}

/**

float* a: A
float* b: (B)T
float* c: C

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
void kernel_4x4_v3(int m, int n, int k,
    float* sa, float * sb, float* sc, int ldc) {
    assert(m > 0 && n > 0 && k > 0);
    assert(m % 4 == 0 && n % 4 == 0 && k % 4 == 0);

    float *a = sa, *b = sb, *c = sc;

    int i, j;
    for(i = 0; i < m; i += 4) {
        for(j = 0; j < n; j += 4) {
            
            float *destptr0 = c;
            float *destptr1 = c + ldc;
            float *destptr2 = c + 2 * ldc;
            float *destptr3 = c + 3 * ldc;
            const float *r0 = b;
            const float *r1 = a;

            int tmpk = k;

            if(tmpk > 0){
                asm volatile(
                    // float *destptr0 = c;
                    "vld1.f32   {d16-d17}, [%1]       \n"

                    // float *destptr1 = c + ldc;
                    "vld1.f32   {d18-d19}, [%2]       \n"

                    // float *destptr2 = c + 2 * ldc;
                    "vld1.f32   {d20-d21}, [%3]       \n"

                    // float *destptr3 = c + 3 * ldc;
                    "vld1.f32   {d22-d23}, [%4]       \n"

                    "0:                               \n"

                    "pld        [%5, #0x180]            \n"
                    "pld        [%6, #0x180]            \n"
		    "vldr	d8, [%6]		\n"
                    "vldr   	d0, [%5]        	\n"
		    "vldr   	d1, [%5, #8]		\n"
		    "vldr	d9, [%6, #8]		\n"
                    "vmla.f32   q8, q0, d8[0]         \n"
		    "vldr       d2, [%5, #16]		\n"
                    "vmla.f32   q9, q0, d8[1]         \n"
		    "vldr       d3, [%5, #24]		\n"
                    "vmla.f32   q10, q0, d9[0]        \n"
		    // L
		    "vldr	d10, [%6, #16]		\n"
                    "vmla.f32   q11, q0, d9[1]        \n"
		    "vldr	d11, [%6, #24]		\n"
                    "vmla.f32   q8, q1, d10[0]        \n"
		    "vldr       d4, [%5, #32]		\n"
                    "vmla.f32   q9, q1, d10[1]        \n"
		    // L
		    "vldr       d5, [%5, #40]		\n"
                    "vmla.f32   q10, q1, d11[0]       \n"
		    "vldr	d12, [%6, #32]		\n"
                    "vmla.f32   q11, q1, d11[1]       \n"
		    "vldr	d13, [%6, #40]		\n"
                    "vmla.f32   q8, q2, d12[0]        \n"
		    // L
		    "vldr       d6, [%5, #48]		\n"
                    "vmla.f32   q9, q2, d12[1]        \n"
		    "vldr       d7, [%5, #56]		\n"
                    "vmla.f32   q10, q2, d13[0]       \n"
		    "add        %5, %5, #64		\n"
		    "vldr	d14, [%6, #48]		\n"
                    "vmla.f32   q11, q2, d13[1]       \n"
		    // L

                    "vmla.f32   q8, q3, d14[0]        \n"
		    "vldr	d15, [%6, #56]		\n"
                    "vmla.f32   q9, q3, d14[1]        \n"
		    "add	%6, %6, #64		\n"
                    "vmla.f32   q10, q3, d15[0]       \n"
                    "vmla.f32   q11, q3, d15[1]       \n"

                    // k-=4
                    "subs       %0, #4                \n"
                    "bne        0b                    \n"

                    // *destptr0 += sum0;
                    "vst1.f32   {d16-d17}, [%1]      \n"
                    // *destptr1 += sum1;
                    "vst1.f32   {d18-d19}, [%2]      \n"
                    // *destptr2 += sum2;
                    "vst1.f32   {d20-d21}, [%3]      \n"
                    // *destptr3 += sum3;
                    "vst1.f32   {d22-d23}, [%4]      \n"

                    : "=r"(tmpk),      // %0
                    "=r"(destptr0), // %1
                    "=r"(destptr1), // %2
                    "=r"(destptr2), // %3
                    "=r"(destptr3), // %4
                    "=r"(r0),      // %5
                    "=r"(r1) //%6
                    : "0"(tmpk),
                    "1"(destptr0),
                    "2"(destptr1),
                    "3"(destptr2),
                    "4"(destptr3),
                    "5"(r0),
                    "6"(r1)

                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
                );
            }
            
            b += k * 4;

            // __builtin_prefetch(b, 0, 3);
            // __builtin_prefetch(a, 0, 3);

            // float32x4_t v24 = vld1q_f32(c);
            // float32x4_t v25 = vld1q_f32(c + ldc);
            // float32x4_t v26 = vld1q_f32(c + 2*ldc);
            // float32x4_t v27 = vld1q_f32(c + 3*ldc);
           
            // for(int l = 0; l < k; l += 4) {
            //     float32x4_t v0 = vld1q_f32(b);
            //     float32x4_t v16 = vld1q_f32(a);

            //     v24 = vmlaq_lane_f32(v24, v0, vget_low_f32(v16), 0);
            //     v25 = vmlaq_lane_f32(v25, v0, vget_low_f32(v16), 1);
            //     v26 = vmlaq_lane_f32(v26, v0, vget_high_f32(v16), 0);
            //     v27 = vmlaq_lane_f32(v27, v0, vget_high_f32(v16), 1);

            //     float32x4_t v1 = vld1q_f32(b + 4);
            //     float32x4_t v17 = vld1q_f32(a + 4);

            //     v24 = vmlaq_lane_f32(v24, v1, vget_low_f32(v17), 0);
            //     v25 = vmlaq_lane_f32(v25, v1, vget_low_f32(v17), 1);
            //     v26 = vmlaq_lane_f32(v26, v1, vget_high_f32(v17), 0);
            //     v27 = vmlaq_lane_f32(v27, v1, vget_high_f32(v17), 1);

            //     float32x4_t v2 = vld1q_f32(b + 8);
            //     float32x4_t v18 = vld1q_f32(a + 8);

            //     v24 = vmlaq_lane_f32(v24, v2, vget_low_f32(v18), 0);
            //     v25 = vmlaq_lane_f32(v25, v2, vget_low_f32(v18), 1);
            //     v26 = vmlaq_lane_f32(v26, v2, vget_high_f32(v18), 0);
            //     v27 = vmlaq_lane_f32(v27, v2, vget_high_f32(v18), 1);

            //     float32x4_t v3 = vld1q_f32(b + 12);
            //     float32x4_t v19 = vld1q_f32(a + 12);

            //     v24 = vmlaq_lane_f32(v24, v3, vget_low_f32(v19), 0);
            //     v25 = vmlaq_lane_f32(v25, v3, vget_low_f32(v19), 1);
            //     v26 = vmlaq_lane_f32(v26, v3, vget_high_f32(v19), 0);
            //     v27 = vmlaq_lane_f32(v27, v3, vget_high_f32(v19), 1);

            //     // __builtin_prefetch(b+16, 0, 3);
            //     // __builtin_prefetch(a+16, 0, 3);

            //     b += 16;
            //     a += 16;
            // } // endl
            
            // v24 = vaddq_f32(vld1q_f32(c), v24);
            // v25 = vaddq_f32(vld1q_f32(c + ldc), v25);
            // v26 = vaddq_f32(vld1q_f32(c + 2*ldc), v26);
            // v27 = vaddq_f32(vld1q_f32(c + 3*ldc), v27);

            // vst1q_f32(c, v24);
            // vst1q_f32(c + ldc, v25);
            // vst1q_f32(c + 2 * ldc, v26);
            // vst1q_f32(c + 3 * ldc, v27);
            c += 4;
            // a -= 4 * k;
        } // endj
        sc += ldc*4;
        c = sc;;
        a += 4*k;
        b = sb;
    }// endi
}


/**
pack A means

Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7

8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f

Pack it zigzag

Output:
0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7
8 8 8 8 9 9 9 9 a a a a b b b b 
c c c c d d d d e e e e f f f f

Draw it with a line
*/
void packA_4(int m, int k, float* from, int lda, float* to) {
#ifdef DEBUG_PACK_SHAPE
    printf("\n packA_4, m=%d, k=%d", m, k);
#endif
    assert( k != 0 && m != 0 && k % 4 == 0 && m % 4 == 0);
    int i, j;

    float *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    float *b_offset;
    float  ctemp1,  ctemp2,  ctemp3,  ctemp4;
    float  ctemp5,  ctemp6,  ctemp7,  ctemp8;
    float  ctemp9, ctemp10, ctemp11, ctemp12;
    float ctemp13, ctemp14, ctemp15, ctemp16;

    a_offset = from;
    b_offset = to;

    j = (m >> 2);
    do{
        a_offset1  = a_offset;
        a_offset2  = a_offset1 + lda;
        a_offset3  = a_offset2 + lda;
        a_offset4  = a_offset3 + lda;
        a_offset += 4 * lda;

        i = (k >> 2);
        do{
            ctemp1  = *(a_offset1 + 0);
            ctemp2  = *(a_offset1 + 1);
            ctemp3  = *(a_offset1 + 2);
            ctemp4  = *(a_offset1 + 3);

            ctemp5  = *(a_offset2 + 0);
            ctemp6  = *(a_offset2 + 1);
            ctemp7  = *(a_offset2 + 2);
            ctemp8  = *(a_offset2 + 3);

            ctemp9  = *(a_offset3 + 0);
            ctemp10 = *(a_offset3 + 1);
            ctemp11 = *(a_offset3 + 2);
            ctemp12 = *(a_offset3 + 3);

            ctemp13 = *(a_offset4 + 0);
            ctemp14 = *(a_offset4 + 1);
            ctemp15 = *(a_offset4 + 2);
            ctemp16 = *(a_offset4 + 3);

            *(b_offset +  0) = ctemp1;
            *(b_offset +  1) = ctemp5;
            *(b_offset +  2) = ctemp9;
            *(b_offset +  3) = ctemp13;

            *(b_offset +  4) = ctemp2;
            *(b_offset +  5) = ctemp6;
            *(b_offset +  6) = ctemp10;
            *(b_offset +  7) = ctemp14;

            *(b_offset +  8) = ctemp3;
            *(b_offset +  9) = ctemp7;
            *(b_offset + 10) = ctemp11;
            *(b_offset + 11) = ctemp15;

            *(b_offset + 12) = ctemp4;
            *(b_offset + 13) = ctemp8;
            *(b_offset + 14) = ctemp12;
            *(b_offset + 15) = ctemp16;

            a_offset1 += 4;
            a_offset2 += 4;
            a_offset3 += 4;
            a_offset4 += 4;

            b_offset += 16;
            i --;
        }while(i > 0);
        j --;
    }while(j > 0);
}

/*
suppose that k and n is mutiple of 4
pack B means

Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7

8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f

Pack it zigzag, not like pack A

Output:
0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
8 9 a b 8 9 a b 8 9 a b 8 9 a b
4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7
c d e f c d e f c d e f c d e f
*/
void packB_4(int k, int n, float* from, int ldb, float* to) {
    assert( k != 0 && n != 0 && k % 4 == 0 && n % 4 == 0);
    int i, j;

    float *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    float *b_offset, *b_offset1;
    float  ctemp1,  ctemp2,  ctemp3,  ctemp4;
    float  ctemp5,  ctemp6,  ctemp7,  ctemp8;
    float  ctemp9, ctemp10, ctemp11, ctemp12;
    float ctemp13, ctemp14, ctemp15, ctemp16;
    a_offset   = from;
    b_offset   = to;

    j = (k >> 2);
    do{
        a_offset1  = a_offset;
        a_offset2  = a_offset1 + ldb;
        a_offset3  = a_offset2 + ldb;
        a_offset4  = a_offset3 + ldb;
        a_offset  += 4 * ldb;

        b_offset1  = b_offset;
        b_offset  += 16;

        i = (n >> 2);
        do{
            ctemp1  = *(a_offset1 + 0);
            ctemp2  = *(a_offset1 + 1);
            ctemp3  = *(a_offset1 + 2);
            ctemp4  = *(a_offset1 + 3);

            ctemp5  = *(a_offset2 + 0);
            ctemp6  = *(a_offset2 + 1);
            ctemp7  = *(a_offset2 + 2);
            ctemp8  = *(a_offset2 + 3);

            ctemp9  = *(a_offset3 + 0);
            ctemp10 = *(a_offset3 + 1);
            ctemp11 = *(a_offset3 + 2);
            ctemp12 = *(a_offset3 + 3);

            ctemp13 = *(a_offset4 + 0);
            ctemp14 = *(a_offset4 + 1);
            ctemp15 = *(a_offset4 + 2);
            ctemp16 = *(a_offset4 + 3);

            a_offset1 += 4;
            a_offset2 += 4;
            a_offset3 += 4;
            a_offset4 += 4;

            *(b_offset1 +  0) = ctemp1;
            *(b_offset1 +  1) = ctemp2;
            *(b_offset1 +  2) = ctemp3;
            *(b_offset1 +  3) = ctemp4;

            *(b_offset1 +  4) = ctemp5;
            *(b_offset1 +  5) = ctemp6;
            *(b_offset1 +  6) = ctemp7;
            *(b_offset1 +  7) = ctemp8;

            *(b_offset1 +  8) = ctemp9;
            *(b_offset1 +  9) = ctemp10;
            *(b_offset1 + 10) = ctemp11;
            *(b_offset1 + 11) = ctemp12;

            *(b_offset1 + 12) = ctemp13;
            *(b_offset1 + 13) = ctemp14;
            *(b_offset1 + 14) = ctemp15;
            *(b_offset1 + 15) = ctemp16;

            b_offset1 += k * 4;
            i --;
        }while(i > 0);
        j --;
    }while(j > 0);
}

/* Suppose that m%4==0 and n%4==0 and k%4==0, avoiding process boundary !! */
void MY_MMult(int m, int n, int k, float *  a, int lda,
                                   float *  b, int ldb,
                                   float *  c, int ldc )
{
#ifdef DEBUG_PRINT_DATA
    printf("\n-------\n");
    print_matrix(m, k, a, lda);
    printf("\n-------\n");
    print_matrix(k, n, b, ldb);
    printf("\n-------\n");
#endif
    float*  sa = fastMalloc(m * k);
    float*  sb = fastMalloc(k * n);

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
                l1stride = 0;
            }
            packB_4(min_k, min_n, b + ks * ldb, ldb, sb);

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
                packA_4(min_mm, min_k, a + mms * lda + ks, lda, sa + min_k * (mms - ms) * l1stride);

                KERNEL_4x4(min_mm, min_n, min_k, sa + l1stride * min_k * (mms - ms), sb, c + mms * ldc, ldc);
#ifdef DEBUG_PRINT_DATA
                printf("\n---first kernel----\n");
                print_matrix(m, n, c, ldc);
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

                packB_4(min_k, min_n, b + ns + ldb * ks, ldb, sb);
                KERNEL_4x4(min_m, min_n, min_k, sa, sb, c + ms * ldc + ns, ldc);
#ifdef DEBUG_PRINT_DATA
                printf("\n----second kernel---\n");
                print_matrix(m, n, c, ldc);
#endif
            }
        }
    }

    free(sa);
    free(sb);
}

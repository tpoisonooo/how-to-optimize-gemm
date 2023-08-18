#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error("arm neon not supported")
#endif

#include <assert.h>
#include <stdlib.h>

/* Block sizes */
#define DEBUG_PACK_SHAPE
#undef DEBUG_PACK_SHAPE
#define DEBUG_PRINT_DATA
#undef DEBUG_PRINT_DATA

/* Create macros so that the matrices are stored in row-major order */

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define min(i, j) ((i) < (j) ? (i) : (j))

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

min_nn * kc <= PAGESIZE/2,  4 <= min_nn <= 12, so that 170 <= kc <= 512, we use
256. After reading 6.4, rk3399 L2 cache is large, mc = 1MB / 256 = 4096


*/
#define GEMM_N (384)  // GEMM_R
#define GEMM_M (4096) // GEMM_P
#define GEMM_K (256)  // GEMM_Q
#define GEMM_UNROLL_4   (4)
#define GEMM_UNROLL_8   (8)
#define GEMM_UNROLL_12  (12)

#define KERNEL_8x12 kernel_8x12

/* Routine for computing C = A * B + C */
void packB_12(int k, int n, float *from, int ldb, float *to);
void packA_8(int m, int k, float *from, int lda, float *to);
void kernel_8x12(int m, int n, int k, float *sa, float *sb, float *sc, int ldc);

float *fastMalloc(int size) {
  void *ptr = 0;
  int iRet = posix_memalign(&ptr, 64, size * sizeof(float));
  assert(0 == iRet);
  return (float *)ptr;
}

/* Suppose that m%8==0 and n%12==0 and k%4==0, avoiding process boundary !! */
void MY_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb,
              float *c, int ldc) {
#ifdef DEBUG_PRINT_DATA
  printf("\n-------\n");
  print_matrix(m, k, a, lda);
  printf("\n-------\n");
  print_matrix(k, n, b, ldb);
  printf("\n-------\n");
#endif
  float *sa = fastMalloc(m * k);
  float *sb = fastMalloc(k * n);

  int ms, mms, ns, ks;
  int min_m, min_mm, min_n, min_k;
  int l1stride = 1;
  for (ms = 0; ms < m; ms += GEMM_M) {
    min_m = m - ms;
    if (min_m > GEMM_M) {
      min_m = GEMM_M;
    }

    for (ks = 0; ks < k; ks += min_k) {
      min_k = k - ks;
      if (min_k >= (GEMM_K << 1)) {
        min_k = GEMM_K;
      } else if (min_k > GEMM_K) {
        min_k = (min_k / 2 + GEMM_UNROLL_4 - 1) & ~(GEMM_UNROLL_4 - 1);
      }

      // first packB
      min_n = n;
      if (n >= GEMM_N * 2) {
        min_n = GEMM_N;
      } else if (n > GEMM_N) {
        min_n = ((min_n / 2 + GEMM_UNROLL_12 - 1) / GEMM_UNROLL_12) * GEMM_UNROLL_12;
      } else {
        l1stride = 0;
      }
      packB_12(min_k, min_n, b + ks * ldb, ldb, sb);

      // micro kernel, split A Block to smaller Panel
      for (mms = ms; mms < ms + min_m; mms += min_mm) {
        min_mm = (ms + min_m) - mms;
        if (min_mm >= 3 * GEMM_UNROLL_8) {
          min_mm = 3 * GEMM_UNROLL_8;
        } else if (min_mm >= 2 * GEMM_UNROLL_8) {
          min_mm = 2 * GEMM_UNROLL_8;
        } else if (min_mm > GEMM_UNROLL_8) {
          min_mm = GEMM_UNROLL_8;
        }

        // coninueous packA
        packA_8(min_mm, min_k, a + mms * lda + ks, lda,
                sa + min_k * (mms - ms) * l1stride);

        KERNEL_8x12(min_mm, min_n, min_k, sa + l1stride * min_k * (mms - ms), sb,
                   c + mms * ldc, ldc);
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
        } else if (min_n > GEMM_N) {
          min_n = (min_n / 2 + GEMM_UNROLL_8 - 1) & ~(GEMM_UNROLL_8 - 1);
        }

        packB_12(min_k, min_n, b + ns + ldb * ks, ldb, sb);
        KERNEL_8x12(min_m, min_n, min_k, sa, sb, c + ms * ldc + ns, ldc);
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
#ifdef __aarch64__
void kernel_8x12(int m, int n, int k, float *sa, float *sb, float *sc,
                   int ldc) {
  assert(m > 0 && n > 0 && k > 0);
  assert(m % 8 == 0 && n % 12 == 0 && k % 4 == 0);

  float *a = sa, *b = sb, *c = sc;
  int i, j;
#if __aarch64__
  int ldc_offset = ldc * sizeof(float);
#endif
  for (i = 0; i < m; i += 8) {
    for (j = 0; j < n; j += 12) {
#ifdef __aarch64__
      asm volatile(
        ".macro MatMul_8x12                           \n"
        "ld1    {v27.4s, v28.4s},          [%0], #32  \n"
        "ld1    {v24.4s, v25.4s, v26.4s},  [%1], #48  \n"

        "fmla   v0.4s,    v24.4s,   v27.s[0]          \n"
        "fmla   v3.4s,    v24.4s,   v27.s[1]          \n"
        "fmla   v6.4s,    v24.4s,   v27.s[2]          \n"
        "fmla   v9.4s,    v24.4s,   v27.s[3]          \n"
        "fmla   v12.4s,   v24.4s,   v28.s[0]          \n"
        "fmla   v15.4s,   v24.4s,   v28.s[1]          \n"
        "fmla   v18.4s,   v24.4s,   v28.s[2]          \n"
        "fmla   v21.4s,   v24.4s,   v28.s[3]          \n"

        "fmla   v1.4s,    v25.4s,   v27.s[0]          \n"
        "fmla   v4.4s,    v25.4s,   v27.s[1]          \n"
        "fmla   v7.4s,    v25.4s,   v27.s[2]          \n"
        "fmla   v10.4s,   v25.4s,   v27.s[3]          \n"
        "fmla   v13.4s,   v25.4s,   v28.s[0]          \n"
        "fmla   v16.4s,   v25.4s,   v28.s[1]          \n"
        "fmla   v19.4s,   v25.4s,   v28.s[2]          \n"
        "fmla   v22.4s,   v25.4s,   v28.s[3]          \n"

        "fmla   v2.4s,    v26.4s,   v27.s[0]          \n"
        "fmla   v5.4s,    v26.4s,   v27.s[1]          \n"
        "fmla   v8.4s,    v26.4s,   v27.s[2]          \n"
        "fmla   v11.4s,   v26.4s,   v27.s[3]          \n"
        "fmla   v14.4s,   v26.4s,   v28.s[0]          \n"
        "fmla   v17.4s,   v26.4s,   v28.s[1]          \n"
        "fmla   v20.4s,   v26.4s,   v28.s[2]          \n"
        "fmla   v23.4s,   v26.4s,   v28.s[3]          \n"
        ".endm                                        \n"

        "asr x8,      %4,           2         \n"
        "ldr  q0,     [%2]                    \n"
        "add  x11,    %2,           %3        \n"
        "ldr  q1,     [%2,  #16]              \n"
        "ldr  q2,     [%2,  #32]              \n"

        "ldr  q3,     [x11]                   \n"
        "add  x12,    x11,          %3        \n"
        "ldr  q4,     [x11, #16]              \n"
        "ldr  q5,     [x11, #32]              \n"

        "ldr  q6,     [x12]                   \n"
        "add  x13,    x12,          %3        \n"
        "ldr  q7,     [x12, #16]              \n"
        "ldr  q8,     [x12, #32]              \n"

        "ldr  q9,     [x13]                   \n"
        "add  x14,    x13,          %3        \n"
        "ldr  q10,    [x13, #16]              \n"
        "ldr  q11,    [x13, #32]              \n"

        "ldr  q12,    [x14]                   \n"
        "add  x15,    x14,          %3        \n"
        "ldr  q13,    [x14, #16]              \n"
        "ldr  q14,    [x14, #32]              \n"

        "ldr  q15,    [x15]                   \n"
        "add  x16,    x15,          %3        \n"
        "ldr  q16,    [x15, #16]              \n"
        "ldr  q17,    [x15, #32]              \n"

        "ldr  q18,    [x16]                   \n"
        "add  x17,    x16,          %3        \n"
        "ldr  q19,    [x16, #16]              \n"
        "ldr  q20,    [x16, #32]              \n"

        "ldr  q21,    [x17]                   \n"
        "ldr  q22,    [x17, #16]              \n"
        "ldr  q23,    [x17, #32]              \n"

        "run:                                 \n"
        "MatMul_8x12                          \n"
        "MatMul_8x12                          \n"
        "MatMul_8x12                          \n"
        "MatMul_8x12                          \n"
        "subs   x8,   x8,   #1                \n"
        "bne    run                           \n"
        "st1    {v0.4s, v1.4s, v2.4s},  [%2]  \n"
        "st1    {v3.4s, v4.4s, v5.4s},  [x11] \n"
        "st1    {v6.4s, v7.4s, v8.4s},  [x12] \n"
        "st1    {v9.4s, v10.4s,v11.4s}, [x13] \n"
        "st1    {v12.4s,v13.4s,v14.4s}, [x14] \n"
        "st1    {v15.4s,v16.4s,v17.4s}, [x15] \n"
        "st1    {v18.4s,v19.4s,v20.4s}, [x16] \n"
        "st1    {v21.4s,v22.4s,v23.4s}, [x17] \n"
        "                                     \n"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(ldc_offset), "=r"(k)
        : "0"(a), "1"(b), "2"(c), "3"(ldc_offset), "4"(k)
        : "memory", "cc", "x8", "x11", "x12","x13", "x14","x15","x16","x17",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
          "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
          "v21", "v22", "v23", "v24", "v25", "v26", "v27","v28");
#endif
      c += 12;
      a -= 8 * k;
    } // endj
    sc += ldc * 8;
    c = sc;
    ;
    a += 8 * k;
    b = sb;
  } // endi
}
#endif

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
0 0 0 0 8 8 8 8 1 1 1 1 9 9 9 9
2 2 2 2 a a a a 3 3 3 3 b b b b
4 4 4 4 c c c c 5 5 5 5 d d d d 
6 6 6 6 e e e e 7 7 7 7 f f f f

Draw it with a line
*/
#ifdef __aarch64__
void packA_8(int m, int k, float *from, int lda, float *to) {

  assert(k != 0 && m != 0 && k % 4 == 0 && m % 8 == 0);

  const float *a_offset = from;
  float *b_offset = to;

  for(int i = 0; i < m; i += 8)
  {
    const float * a_offset1 = a_offset;
    const float * a_offset2 = a_offset1 + lda;
    const float * a_offset3 = a_offset2 + lda;
    const float * a_offset4 = a_offset3 + lda;
    const float * a_offset5 = a_offset4 + lda;
    const float * a_offset6 = a_offset5 + lda;
    const float * a_offset7 = a_offset6 + lda;
    const float * a_offset8 = a_offset7 + lda;
    a_offset += 8 * lda;

    for(int j = 0; j < k; j += 4)
    {
      float32x4_t _v0 = vld1q_f32(a_offset1);
      float32x4_t _v1 = vld1q_f32(a_offset2);
      float32x4_t _v2 = vld1q_f32(a_offset3);
      float32x4_t _v3 = vld1q_f32(a_offset4);
      float32x4_t _v4 = vld1q_f32(a_offset5);
      float32x4_t _v5 = vld1q_f32(a_offset6);
      float32x4_t _v6 = vld1q_f32(a_offset7);
      float32x4_t _v7 = vld1q_f32(a_offset8);

      a_offset1 += 4;
      a_offset2 += 4;
      a_offset3 += 4;
      a_offset4 += 4;
      a_offset5 += 4;
      a_offset6 += 4;
      a_offset7 += 4;
      a_offset8 += 4;

      float32x4x2_t _vv0 = vtrnq_f32(_v0, _v1);
      float32x4x2_t _vv1 = vtrnq_f32(_v2, _v3);
      float32x4x2_t _vv2 = vtrnq_f32(_v4, _v5);
      float32x4x2_t _vv3 = vtrnq_f32(_v6, _v7);

      float32x4_t _v8 = vcombine_f32(vget_low_f32(_vv0.val[0]), vget_low_f32(_vv1.val[0]));
      float32x4_t _v9 = vcombine_f32(vget_low_f32(_vv0.val[1]), vget_low_f32(_vv1.val[1]));
      float32x4_t _v10 = vcombine_f32(vget_high_f32(_vv0.val[0]), vget_high_f32(_vv1.val[0]));
      float32x4_t _v11 = vcombine_f32(vget_high_f32(_vv0.val[1]), vget_high_f32(_vv1.val[1]));

      float32x4_t _v12 = vcombine_f32(vget_low_f32(_vv2.val[0]), vget_low_f32(_vv3.val[0]));
      float32x4_t _v13 = vcombine_f32(vget_low_f32(_vv2.val[1]), vget_low_f32(_vv3.val[1]));
      float32x4_t _v14 = vcombine_f32(vget_high_f32(_vv2.val[0]), vget_high_f32(_vv3.val[0]));
      float32x4_t _v15 = vcombine_f32(vget_high_f32(_vv2.val[1]), vget_high_f32(_vv3.val[1]));

      vst1q_f32(b_offset + 0,   _v8);
      vst1q_f32(b_offset + 4,   _v12);
      vst1q_f32(b_offset + 8,   _v9);
      vst1q_f32(b_offset + 12,  _v13);
      vst1q_f32(b_offset + 16,  _v10);
      vst1q_f32(b_offset + 20,  _v14);
      vst1q_f32(b_offset + 24,  _v11);
      vst1q_f32(b_offset + 28,  _v15);
      b_offset += 32;
    }
  }
}
#endif // __aarch64__

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
#ifdef __aarch64__
void packB_12(int k, int n, float *from, int ldb, float *to) {
  // printf("packB_12, k=%d, n=%d\n", k, n);
  assert(k != 0 && n != 0 && n % 12 == 0);

  for(int i = 0; i < k; i++)
  {
    const float *a_offset1 = from + i * ldb;
    float * b_offset = to + i * 12;
    for(int j = 0; j < n; j += 12)
    {
      float32x4_t _v0   = vld1q_f32(a_offset1);
      float32x4_t _v1   = vld1q_f32(a_offset1+4);
      float32x4_t _v2   = vld1q_f32(a_offset1+8);
      a_offset1 += 12;
      
      vst1q_f32(b_offset,     _v0);
      vst1q_f32(b_offset+4,   _v1);
      vst1q_f32(b_offset+8,   _v2);
      b_offset += 12*k;
    }
  }
}
#endif // __aarch64__
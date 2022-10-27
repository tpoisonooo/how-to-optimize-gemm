#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error("arm neon not supported")
#endif

/* Block sizes */
#define mc 256
#define kc 128

/* Create macros so that the matrices are stored in row-major order */

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define min(i, j) ((i) < (j) ? (i) : (j))

/* Routine for computing C = A * B + C */

void AddDot4x4(int, float *, int, float *, int, float *, int);

void InnerKernel(int, int, int, float *, int, float *, int, float *, int);

void MY_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb,
              float *c, int ldc) {
  int i, p, pb, ib;
  for (p = 0; p < k; p += kc) {
    pb = min(k - p, kc);
    for (i = 0; i < m; i += mc) {
      ib = min(m - i, mc);
      InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
    }
  }
}

void InnerKernel(int m, int n, int k, float *a, int lda, float *b, int ldb,
                 float *c, int ldc) {
  int i, j;

  for (j = 0; j < n; j += 4) {   /* Loop over the columns of C, unrolled by 4 */
    for (i = 0; i < m; i += 4) { /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
         one routine (four inner products) */

      AddDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

void AddDot4x4(int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 )
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 )
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 )
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 )

     in the original matrix C

     In this version, we use registers for elements in the current row
     of B as well */

  float
      /* Point to the current elements in the four rows of A */
      *a_0p_pntr,
      *a_1p_pntr, *a_2p_pntr, *a_3p_pntr;

  a_0p_pntr = &A(0, 0);
  a_1p_pntr = &A(1, 0);
  a_2p_pntr = &A(2, 0);
  a_3p_pntr = &A(3, 0);

  float32x4_t c_0p_sum = { 0 };
  float32x4_t c_1p_sum = { 0 };
  float32x4_t c_2p_sum = { 0 };
  float32x4_t c_3p_sum = { 0 };

  register float a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;

  for (int p = 0; p < k; ++p) {
    float32x4_t b_reg = vld1q_f32(&B(p, 0));

    a_0p_reg = *a_0p_pntr++;
    a_1p_reg = *a_1p_pntr++;
    a_2p_reg = *a_2p_pntr++;
    a_3p_reg = *a_3p_pntr++;

    c_0p_sum = vmlaq_n_f32(c_0p_sum, b_reg, a_0p_reg);
    c_1p_sum = vmlaq_n_f32(c_1p_sum, b_reg, a_1p_reg);
    c_2p_sum = vmlaq_n_f32(c_2p_sum, b_reg, a_2p_reg);
    c_3p_sum = vmlaq_n_f32(c_3p_sum, b_reg, a_3p_reg);
  }

  float *c_pntr = 0;
  c_pntr = &C(0, 0);
  float32x4_t c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_0p_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &C(1, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_1p_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &C(2, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_2p_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &C(3, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_3p_sum);
  vst1q_f32(c_pntr, c_reg);
}

// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/* Test and benchmark for GEMM: C = alpha * A * B + beta * C.
 *
 * A is M x K with row stride RSA and column stride CSA.
 * B is K x N with row stride RSB and column stride CSB.
 * C is M x N with row stride RSC and column stride CSC.
 *
 * Users must provide the values of the following as compiler flags using -D:
 *  - All dimensions M, N, and K;
 *  - Both RSA and CSA; or neither (implying RSA = K, CSA = 1)
 *  - Both RSB and CSB; or neither (implying RSB = N, CSB = 1)
 *  - Both RSC and CSC; or neither (implying RSC = N, CSC = 1)
 *  - Both ALPHA and BETA, as floating point literals.
 */

#if !(defined(ENABLE_TEST) || defined(ENABLE_BENCHMARK))
#error Must define at least one of ENABLE_TEST and ENABLE_BENCHMARK
#endif

#ifndef M
#error Must define M
#endif

#ifndef N
#error Must define N
#endif

#ifndef K
#error Must define K
#endif

#ifndef ALPHA
#error Must define ALPHA
#endif

#ifndef BETA
#error Must define BETA
#endif

#if defined(ENABLE_TEST)
#include <math.h>
#endif

#if defined(SKL_TEST_RAND_MIN_I8) || defined(SKL_TEST_RAND_MAX_I8)
#error "SKL_TEST_RAND_{MIN,MAX}_I8 already defined and must not be"
#endif

// The min value for E4M3 is 2^-6, represented by 0.0001.000, which is 8 in
// int8_t
#define SKL_TEST_RAND_MIN_I8 8
// The max value for E4M3 is 1.75 * 2^8, represented by 0.1111.110, which is 126
// in int8_t
#define SKL_TEST_RAND_MAX_I8 126

#ifndef SKL_TEST_PERF_REPORT
#define SKL_TEST_PERF_REPORT report_perf_mpc
#endif

// NOLINTNEXTLINE(misc-include-cleaner)
#include "skl-ref.h"
#include "skl-test.h"
#include "skl.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__riscv_xsfmm32a8f)
void skl_gemm_a1b01_f8e4m3c_f8e4m3_f32_xsfmm32a8f_wrapper(
    size_t m, size_t n, size_t k, float alpha, const uint8_t *a, size_t rsa,
    size_t csa, const uint8_t *b, size_t rsb, size_t csb, float beta, float *c,
    size_t rsc, size_t csc) {
  int status = 0;
  SKL_TEST_REQUIRE(status, rsa == 1);
  SKL_TEST_REQUIRE(status, csb == 1);
  SKL_TEST_REQUIRE(status, csc == 1);
  SKL_TEST_REQUIRE(status, alpha == 1.f);
  SKL_TEST_REQUIRE(status, beta == 0.f || beta == 1.f);
  if (status) {
    exit(status);
  }
  skl_gemm_a1b01_f8e4m3c_f8e4m3_f32_xsfmm32a8f(m, n, k, a, csa, b, rsb, c, rsc,
                                               beta != 0.f);
}
#endif

/* The macros below set the matrix strides. */
#if !defined(RSA) && !defined(CSA)
#define RSA K // Default to row major
#define CSA 1
#elif !defined(RSA) || !defined(CSA)
#error Must define both RSA and CSA, or neither
#endif
#if !defined(RSB) && !defined(CSB)
#define RSB N // Default to row major
#define CSB 1
#elif !defined(RSB) || !defined(CSB)
#error Must define both RSB and CSB, or neither
#endif
#if !defined(RSC) && !defined(CSC)
#define RSC N // Default to row major
#define CSC 1
#elif !defined(RSC) || !defined(CSC)
#error Must define both RSC and CSC, or neither
#endif

/* Input, output, reference, and error bound arrays */
enum {
  ALIGN = 4096,
  CLEN = ((M - 1) * RSC + (N - 1) * CSC + 1),
  ALEN = ((M - 1) * RSA + (K - 1) * CSA + 1),
  BLEN = ((K - 1) * RSB + (N - 1) * CSB + 1),
};
_Alignas(ALIGN) uint8_t a[ALEN];
_Alignas(ALIGN) uint8_t b[BLEN];
_Alignas(ALIGN) float c[CLEN];
#if defined(ENABLE_TEST)
double a_wide[ALEN];
double b_wide[BLEN];
float ref_c[CLEN], test_c[CLEN];
double bound[CLEN];
#endif // ENABLE_TEST

#if defined(ENABLE_TEST)
uint8_t get_random_f8e4m3(void) {
  /* For e4m3, 0.0001.000 is the min positive normal number
   * and 0.1111.110 is the max positive normal number
   */
  const uint8_t min = 8;
  const uint8_t max = 126;
  size_t range = max - min + 1;
  return min + (rand() % range);
}
#else
/* To save simulation time, just use an increasing int value when tests are
 * disabled.
 */
uint8_t get_random_f8e4m3(void) {
  /* For e4m3, 0.0001.000 is the min positive normal number
   * and 0.1111.110 is the max positive normal number
   */
  static uint8_t next = 0;
  const uint8_t min = 8;
  const uint8_t max = 126;
  size_t range = max - min + 1;
  return min + (next++ % range);
}
#endif // ENABLE_TEST

void init_random_f8e4m3(uint8_t *arr, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    arr[i] = get_random_f8e4m3();
  }
}

#if defined(ENABLE_TEST)
/**
 * @brief Scalar OFP8 E4M3 matrix-matrix multiplication with float32
 * accumulator.
 *
 * @param m - Number of rows in matrices A and C.
 * @param n - Number of columns in matrices B and C.
 * @param k - Number of columns in A and rows in B (inner dimension).
 * @param alpha - Scalar multiplier for A * B product.
 * @param a - Pointer to matrix A.
 * @param rsa - Row stride of matrix A in elements.
 * @param csa - Column stride of matrix A in elements.
 * @param b - Pointer to matrix B.
 * @param rsb - Row stride of matrix B in elements.
 * @param csb - Column stride of matrix B in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to matrix C.
 * @param rsc - Row stride of matrix C in elements.
 * @param csc - Column stride of matrix C in elements.
 *
 * Computes `C = alpha * A * B + beta * C` for matrices A, B, and C.
 * This generic GEMM function defines the semantics of all optimized OFP8 E4M3
 * GEMM kernels with FP32 accumulators in SKL.
 *
 * The entries of A and B are 8-bit floating point numbers in E4M3 format,
 * type-punned as 8-bit unsigned integers.
 *
 * Matrices may be in row-major or column-major order, depending on the strides.
 * For row-major matrices, the column stride is 1, and the row stride is the
 * leading dimension. For column-major matrices, the reverse is true.
 *
 * @note
 * This function is for API documentation purposes only, and should not be used
 * for performance applications.
 */
static inline void skl_gemm_f8e4m3rc_f8e4m3rc_f32rc_scalar(
    size_t m, size_t n, size_t k, float alpha, const uint8_t *a, size_t rsa,
    size_t csa, const uint8_t *b, size_t rsb, size_t csb, float beta, float *c,
    size_t rsc, size_t csc) {
  for (size_t ii = 0; ii < m; ii++) {
    for (size_t jj = 0; jj < n; jj++) {
      float acc = 0;
      for (size_t kk = 0; kk < k; kk++) {
        acc += skl_cvt_f8e4m3_f32(a[ii * rsa + kk * csa]) *
               skl_cvt_f8e4m3_f32(b[kk * rsb + jj * csb]);
      }
      c[ii * rsc + jj * csc] = beta * c[ii * rsc + jj * csc] + alpha * acc;
    }
  }
}

/* Check result after executing test and reference functions.
 *
 * Compares the test_c and ref_c matrices based on a bound derived from
 * problem parameters. Returns nonzero in case of an error, and prints
 * first incorrect output matrix element. */
int check_error(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_gemm_f8e4m3rc_f8e4m3rc_f32rc_scalar(M, N, K, ALPHA, a, RSA, CSA, b, RSB,
                                          CSB, BETA, ref_c, RSC, CSC);

  //
  // Compute the error bound array for comparing test vs reference results.
  //
  // For any matrix M, define |M| as the matrix of absolute values where:
  //     |M|(i,j) = |M(i,j)|
  //
  // Let u = 2^-P be the maximum relative roundoff error for a floating-point
  // type with P-1 mantissa bits.
  //
  // For GEMM operations, the error between computed and exact results is
  // bounded by:
  //     ((1 + u)^(K + 2) - 1) * (|alpha| * |A| * |B| + |beta| * |C|)
  //
  // Since both test and reference results have roundoff errors, we double this
  // bound using the triangle inequality to get the final comparison threshold.
  //
  for (size_t i = 0; i < ALEN; ++i) {
    a_wide[i] = fabs((double)skl_cvt_f8e4m3_f32(a[i]));
  }
  for (size_t i = 0; i < BLEN; ++i) {
    b_wide[i] = fabs((double)skl_cvt_f8e4m3_f32(b[i]));
  }
  for (size_t i = 0; i < CLEN; ++i) {
    bound[i] = fabs((double)c[i]);
  }
  const int P = 24; // 23 bits of mantissa for float32 accumulator
  const double u = ldexp(1.0, -P); // Maximum relative roundoff error
  // Compute 2 * ((1 + u)^(K + 2) - 1) by change of base formula:
  const double roundoff_scaling = 2 * expm1((K + 2) * log1p(u));
  skl_gemm_f64rc_f64rc_f64rc_scalar(
      M, N, K, roundoff_scaling * fabs((double)ALPHA), a_wide, RSA, CSA, b_wide,
      RSB, CSB, roundoff_scaling * fabs((double)BETA), bound, RSC, CSC);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t idx = i * RSC + j * CSC;
      if (fabs((double)test_c[idx] - (double)ref_c[idx]) > bound[idx]) {
        printf("result [%zu, %zu] (%f) != reference (%f)\n", i, j, test_c[idx],
               ref_c[idx]);
        return 1;
      }
    }
  }

  return 0;
}
#endif // ENABLE_TEST

int main(void) {
  int res = EXIT_SUCCESS;

  printf("%s:\n", skl_test_name);
  printf("M = %u, N = %u, K = %u\n", M, N, K);
  printf("ALPHA = %f, BETA = %f\n", ALPHA, BETA);
  printf("RSA = %u, CSA = %u\n", RSA, CSA);
  printf("RSB = %u, CSB = %u\n", RSB, CSB);
  printf("RSC = %u, CSC = %u\n", RSC, CSC);

  /* Populate the matrices. */
  skl_test_init_i8((int8_t *)a, ALEN, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  skl_test_init_i8((int8_t *)b, BLEN, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  skl_test_init_f32(c, CLEN, SKL_TEST_MIN_F32, SKL_TEST_MAX_F32);

#if defined(ENABLE_TEST)
  /* Make copies of C to write the reference and test outputs to. */
  memcpy(ref_c, c, CLEN * sizeof(float));
  memcpy(test_c, c, CLEN * sizeof(float));
  SKL_TEST_NAME(M, N, K, ALPHA, a, RSA, CSA, b, RSB, CSB, BETA, test_c, RSC,
                CSC);
  res += check_error();
#endif // ENABLE_TEST

  SKL_BENCHMARK_RUN(skl_test_name, M * N * K, SKL_TEST_WARMUP, SKL_TEST_NAME, M,
                    N, K, ALPHA, a, RSA, CSA, b, RSB, CSB, BETA, c, RSC, CSC);

  return res;
}

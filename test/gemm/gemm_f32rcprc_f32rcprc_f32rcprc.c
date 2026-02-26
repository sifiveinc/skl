// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/* Test and benchmark for packed GEMM: C = alpha * A * B + beta * C.
 *
 * A is packed M1 x K1 with row stride RSA1 and column stride CSA1.
 * Blocks of A are M0 x K0 with row stride RSA0 and column stride CSA0.
 * B is packed K1 x N1 with row stride RSB1 and column stride CSB1.
 * Blocks of B are K0 x N0 with row stride RSB0 and column stride CSB0.
 * C is packed M1 x N1 with row stride RSC1 and column stride CSC1.
 * Blocks of C are M0 x N0 with row stride RSC0 and column stride CSC0.
 *
 * Users must provide the values of the following as compiler flags using -D:
 *  - All dimensions M0, N0, K0, M1, N1, and K1;
 *  - Both RSA0 and CSA0; or neither (implying RSA0 = K0, CSA0 = 1)
 *  - Both RSA1 and CSA1; or neither (implying RSA1 = M0 * K0 * K1,
 *    CSA1 = M0 * K0)
 *  - Both RSB0 and CSB0; or neither (implying RSB0 = N0, CSB0 = 1)
 *  - Both RSB1 and CSB1; or neither (implying RSB1 = K0 * N0 * N1,
 *    CSB1 = K0 * N0)
 *  - Both RSC0 and CSC0; or neither (implying RSC0 = N0, CSC0 = 1)
 *  - Both RSC1 and CSC1; or neither (implying RSC1 = M0 * N0 * N1,
 *    CSC1 = M0 * N0)
 *  - Both ALPHA and BETA, as floating point literals.
 */

#if !(defined(ENABLE_TEST) || defined(ENABLE_BENCHMARK))
#error Must define at least one of ENABLE_TEST and ENABLE_BENCHMARK
#endif

#ifndef M0
#error Must define M0
#endif

#ifndef N0
#error Must define N0
#endif

#ifndef K0
#error Must define K0
#endif

#ifndef M1
#error Must define M1
#endif

#ifndef N1
#error Must define N1
#endif

#ifndef K1
#error Must define K1
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

#ifndef SKL_TEST_PERF_REPORT
#define SKL_TEST_PERF_REPORT report_perf_mpc
#endif

#include "skl-test.h"
#include "skl.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Include various FP32 GEMM kernels depending on ISA compatibility. */
#if defined(__riscv_xsfmm32a32f)
void skl_gemm_a1b01_f32pc_f32cp_f32rcp_xsfmm32a32f_wrapper(
    size_t m0, size_t n0, size_t k0, size_t m1, size_t n1, size_t k1,
    float alpha, const float *a_pack, size_t rsa0,
    __attribute__((unused)) size_t csa0, size_t rsa1, size_t csa1,
    const float *b_pack, __attribute__((unused)) size_t rsb0, size_t csb0,
    size_t rsb1, size_t csb1, float beta, float *c_pack, size_t rsc0,
    size_t csc0, size_t rsc1, size_t csc1) {
  int status = 0;
  size_t te = skl_get_te_xsfmmbase();
  SKL_TEST_REQUIRE(status, m0 == te);
  SKL_TEST_REQUIRE(status, n0 == te);
  SKL_TEST_REQUIRE(status, k0 == 1);
  SKL_TEST_REQUIRE(status, rsa0 == 1);
  SKL_TEST_REQUIRE(status, csa1 == m0 * k0);
  SKL_TEST_REQUIRE(status, csb0 == 1);
  SKL_TEST_REQUIRE(status, rsb1 == k0 * n0);
  SKL_TEST_REQUIRE(status, rsc0 == n0);
  SKL_TEST_REQUIRE(status, csc0 == 1);
  SKL_TEST_REQUIRE(status, alpha == 1.f);
  SKL_TEST_REQUIRE(status, beta == 0.f || beta == 1.f);
  if (status) {
    exit(status);
  }
  skl_gemm_a1b01_f32pc_f32cp_f32rcp_xsfmm32a32f(m1, n1, k0 * k1, a_pack, rsa1,
                                                b_pack, csb1, c_pack, rsc1,
                                                csc1, beta != 0.f);
}
#endif

/* The macros below set the matrix strides. */
#if !defined(RSA0) && !defined(CSA0)
#define RSA0 K0 // Default to row major
#define CSA0 1
#elif !defined(RSA0) || !defined(CSA0)
#error Must define both RSA0 and CSA0, or neither
#endif
#if !defined(RSA1) && !defined(CSA1)
#define RSA1 (M0 * K0 * K1) // Default to row major
#define CSA1 (M0 * K0)
#elif !defined(RSA1) || !defined(CSA1)
#error Must define both RSA1 and CSA1, or neither
#endif
#if !defined(RSB0) && !defined(CSB0)
#define RSB0 N0 // Default to row major
#define CSB0 1
#elif !defined(RSB0) || !defined(CSB0)
#error Must define both RSB0 and CSB0, or neither
#endif
#if !defined(RSB1) && !defined(CSB1)
#define RSB1 (K0 * N0 * N1) // Default to row major
#define CSB1 (K0 * N0)
#elif !defined(RSB1) || !defined(CSB1)
#error Must define both RSB1 and CSB1, or neither
#endif
#if !defined(RSC0) && !defined(CSC0)
#define RSC0 N0 // Default to row major
#define CSC0 1
#elif !defined(RSC0) || !defined(CSC0)
#error Must define both RSC0 and CSC0, or neither
#endif
#if !defined(RSC1) && !defined(CSC1)
#define RSC1 (M0 * N0 * N1) // Default to row major
#define CSC1 (M0 * N0)
#elif !defined(RSC1) || !defined(CSC1)
#error Must define both RSC1 and CSC1, or neither
#endif

/* Input, output, reference, and error bound arrays */
enum {
  ALIGN = 4096,
  // NOLINTBEGIN(misc-redundant-expression)
  CLEN = ((M1 - 1) * RSC1 + (N1 - 1) * CSC1 + (M0 - 1) * RSC0 +
          (N0 - 1) * CSC0 + 1),
  ALEN = ((M1 - 1) * RSA1 + (K1 - 1) * CSA1 + (M0 - 1) * RSA0 +
          (K0 - 1) * CSA0 + 1),
  BLEN = ((K1 - 1) * RSB1 + (N1 - 1) * CSB1 + (K0 - 1) * RSB0 +
          (N0 - 1) * CSB0 + 1),
  // NOLINTEND(misc-redundant-expression)
};
_Alignas(ALIGN) float a[ALEN];
_Alignas(ALIGN) float b[BLEN];
_Alignas(ALIGN) float c[CLEN];
#if defined(ENABLE_TEST)
double a_wide[ALEN];
double b_wide[BLEN];
float ref_c[CLEN], test_c[CLEN];
double bound[CLEN];
#endif // ENABLE_TEST

#if defined(ENABLE_TEST)
/* Check result after executing test and reference functions.
 *
 * Compares the test_c and ref_c matrices based on a bound derived from
 * problem parameters. Returns nonzero in case of an error, and prints
 * first incorrect output matrix element. */
int check_error(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_gemm_f32rcprc_f32rcprc_f32rcprc_scalar(
      M0, N0, K0, M1, N1, K1, ALPHA, a, RSA0, CSA0, (size_t)RSA1, (size_t)CSA1,
      b, RSB0, CSB0, (size_t)RSB1, (size_t)CSB1, BETA, ref_c, RSC0, CSC0,
      (size_t)RSC1, (size_t)CSC1);

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
    a_wide[i] = fabsf(a[i]);
  }
  for (size_t i = 0; i < BLEN; ++i) {
    b_wide[i] = fabsf(b[i]);
  }
  for (size_t i = 0; i < CLEN; ++i) {
    bound[i] = fabsf(c[i]);
  }
  const int P = 24; // 23 bits of mantissa for float32 accumulator
  const double u = ldexp(1.0, -P); // Maximum relative roundoff error
  // Compute 2 * ((1 + u)^(K + 2) - 1) by change of base formula:
  const double roundoff_scaling = 2 * expm1((K0 * K1 + 2) * log1p(u));
  skl_gemm_f64rcprc_f64rcprc_f64rcprc_scalar(
      M0, N0, K0, M1, N1, K1, roundoff_scaling * fabs((double)ALPHA), a_wide,
      RSA0, CSA0, (size_t)RSA1, (size_t)CSA1, b_wide, RSB0, CSB0, (size_t)RSB1,
      (size_t)CSB1, roundoff_scaling * fabs((double)BETA), bound, RSC0, CSC0,
      (size_t)RSC1, (size_t)CSC1);

  /* Compare the reference and test outputs. */
  for (size_t i1 = 0; i1 < M1; ++i1) {
    for (size_t j1 = 0; j1 < N1; ++j1) {
      for (size_t i0 = 0; i0 < M0; ++i0) {
        for (size_t j0 = 0; j0 < N0; ++j0) {
          size_t idx =
              i1 * (size_t)RSC1 + j1 * (size_t)CSC1 + i0 * RSC0 + j0 * CSC0;
          if (fabs((double)test_c[idx] - (double)ref_c[idx]) > bound[idx]) {
            printf("result [%zu, %zu, %zu, %zu] (%f) != reference (%f)\n", i0,
                   j0, i1, j1, test_c[idx], ref_c[idx]);
            return 1;
          }
        }
      }
    }
  }

  return 0;
}
#endif // ENABLE_TEST

int gemm_f32rcprc_f32rcprc_f32rcprc_main(void) {
  int res = EXIT_SUCCESS;

  printf("%s:\n", skl_test_name);
  printf("M0 = %u, N0 = %u, K0 = %u\n", M0, N0, K0);
  printf("M1 = %u, N1 = %u, K1 = %u\n", M1, N1, K1);
  printf("ALPHA = %f, BETA = %f\n", ALPHA, BETA);
  printf("RSA0 = %u, CSA0 = %u, RSA1 = %u, CSA1 = %u\n", RSA0, CSA0, RSA1,
         CSA1);
  printf("RSB0 = %u, CSB0 = %u, RSB1 = %u, CSB1 = %u\n", RSB0, CSB0, RSB1,
         CSB1);
  printf("RSC0 = %u, CSC0 = %u, RSC1 = %u, CSC1 = %u\n", RSC0, CSC0, RSC1,
         CSC1);

  /* Populate the matrices. */
  skl_test_init_f32(a, ALEN, SKL_TEST_MIN_F32, SKL_TEST_MAX_F32);
  skl_test_init_f32(b, BLEN, SKL_TEST_MIN_F32, SKL_TEST_MAX_F32);
  skl_test_init_f32(c, CLEN, SKL_TEST_MIN_F32, SKL_TEST_MAX_F32);

#if defined(ENABLE_TEST)
  /* Make copies of C to write the reference and test outputs to. */
  memcpy(ref_c, c, CLEN * sizeof(float));
  memcpy(test_c, c, CLEN * sizeof(float));
  SKL_TEST_NAME(M0, N0, K0, M1, N1, K1, ALPHA, a, RSA0, CSA0, (size_t)RSA1,
                (size_t)CSA1, b, RSB0, CSB0, (size_t)RSB1, (size_t)CSB1, BETA,
                test_c, RSC0, CSC0, (size_t)RSC1, (size_t)CSC1);
  res += check_error();
#endif // ENABLE_TEST

  SKL_BENCHMARK_RUN(skl_test_name, M0 * N0 * K0 * M1 * N1 * K1, SKL_TEST_WARMUP,
                    SKL_TEST_NAME, M0, N0, K0, M1, N1, K1, ALPHA, a, RSA0, CSA0,
                    (size_t)RSA1, (size_t)CSA1, b, RSB0, CSB0, (size_t)RSB1,
                    (size_t)CSB1, BETA, c, RSC0, CSC0, (size_t)RSC1,
                    (size_t)CSC1);

  return res;
}

#if defined(SKL_TEST_MAIN)
int main(void) { return gemm_f32rcprc_f32rcprc_f32rcprc_main(); }
#endif

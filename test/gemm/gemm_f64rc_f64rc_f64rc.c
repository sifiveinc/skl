// Copyright 2025 SiFive, Inc.
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

#if defined(__riscv_zve64d)
void skl_gemm_f64_f64_f64_zve64d_x390_wrapper(
    size_t m, size_t n, size_t k, double alpha, const double *a, size_t rsa,
    size_t csa, const double *b, size_t rsb, size_t csb, double beta, double *c,
    size_t rsc, size_t csc) {
  int status = 0;
  SKL_TEST_REQUIRE(status, csa == 1);
  SKL_TEST_REQUIRE(status, csb == 1);
  SKL_TEST_REQUIRE(status, csc == 1);
  if (status) {
    exit(status);
  }
  skl_gemm_f64_f64_f64_zve64d_x390(m, n, k, alpha, a, rsa, b, rsb, beta, c,
                                   rsc);
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
_Alignas(ALIGN) double a[ALEN];
_Alignas(ALIGN) double b[BLEN];
_Alignas(ALIGN) double c[CLEN];
#if defined(ENABLE_TEST)
long double a_wide[ALEN];
long double b_wide[BLEN];
double ref_c[CLEN], test_c[CLEN];
long double bound[CLEN];
#endif // ENABLE_TEST

#if defined(ENABLE_TEST)
/* Check result after executing test and reference functions.
 *
 * Compares the test_c and ref_c matrices based on a bound derived from
 * problem parameters. Returns nonzero in case of an error, and prints
 * first incorrect output matrix element. */
int check_error(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_gemm_f64rc_f64rc_f64rc_scalar(M, N, K, ALPHA, a, RSA, CSA, b, RSB, CSB,
                                    BETA, ref_c, RSC, CSC);

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
    a_wide[i] = fabsl(a[i]);
  }
  for (size_t i = 0; i < BLEN; ++i) {
    b_wide[i] = fabsl(b[i]);
  }
  for (size_t i = 0; i < CLEN; ++i) {
    bound[i] = fabsl(c[i]);
  }
  const int P = 53; // 52 bits of mantissa for float64 accumulator
  const long double u = ldexpl(1.0, -P); // Maximum relative roundoff error
  // Compute 2 * ((1 + u)^(K + 2) - 1) by change of base formula:
  const long double roundoff_scaling = 2 * expm1l((K + 2) * log1pl(u));
  skl_gemm_f128rc_f128rc_f128rc_scalar(
      M, N, K, roundoff_scaling * fabsl(ALPHA), a_wide, RSA, CSA, b_wide, RSB,
      CSB, roundoff_scaling * fabsl(BETA), bound, RSC, CSC);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t idx = i * RSC + j * CSC;
      if (fabsl((long double)test_c[idx] - (long double)ref_c[idx]) >
          bound[idx]) {
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
  SKL_TEST_INIT_F64(a, ALEN);
  SKL_TEST_INIT_F64(b, BLEN);
  SKL_TEST_INIT_F64(c, CLEN);

#if defined(ENABLE_TEST)
  /* Make copies of C to write the reference and test outputs to. */
  memcpy(ref_c, c, CLEN * sizeof(double));
  memcpy(test_c, c, CLEN * sizeof(double));
  SKL_TEST_NAME(M, N, K, ALPHA, a, RSA, CSA, b, RSB, CSB, BETA, test_c, RSC,
                CSC);
  res += check_error();
#endif // ENABLE_TEST

  SKL_BENCHMARK_RUN(skl_test_name, M * N * K, SKL_TEST_WARMUP, SKL_TEST_NAME, M,
                    N, K, ALPHA, a, RSA, CSA, b, RSB, CSB, BETA, c, RSC, CSC);

  return res;
}

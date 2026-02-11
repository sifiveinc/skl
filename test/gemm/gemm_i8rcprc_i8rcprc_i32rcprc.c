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
 *  - Both ALPHA and BETA, as integer literals.
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

#include "skl-test.h"
#include "skl.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Include various int8 GEMM kernels depending on ISA compatibility. */
#if defined(__riscv_xsfmm32a8i)
void skl_gemm_a1b01_i8pc_i8cp_i32rcp_xsfmm32a8i_wrapper(
    size_t m0, size_t n0, size_t k0, size_t m1, size_t n1, size_t k1,
    int32_t alpha, const int8_t *a_pack, size_t rsa0,
    __attribute__((unused)) size_t csa0, size_t rsa1, size_t csa1,
    const int8_t *b_pack, __attribute__((unused)) size_t rsb0, size_t csb0,
    size_t rsb1, size_t csb1, int32_t beta, int32_t *c_pack, size_t rsc0,
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
  SKL_TEST_REQUIRE(status, alpha == 1);
  SKL_TEST_REQUIRE(status, beta == 0 || beta == 1);
  if (status) {
    exit(status);
  }
  skl_gemm_a1b01_i8pc_i8cp_i32rcp_xsfmm32a8i(m1, n1, k0 * k1, a_pack, rsa1,
                                             b_pack, csb1, c_pack, rsc1, csc1,
                                             beta != 0);
}
#endif

/* Include various int8 GEMM kernels depending on ISA compatibility. */
#if defined(__riscv_xsfvqdotq)
void skl_gemm_a1b01_i8_i8pc_i32_xsfvqdotq_wrapper(
    size_t m0, size_t n0, size_t k0, size_t m1, size_t n1, size_t k1,
    int32_t alpha, const int8_t *a_pack, __attribute__((unused)) size_t rsa0,
    size_t csa0, size_t rsa1, size_t csa1, const int8_t *b_pack, size_t rsb0,
    __attribute__((unused)) size_t csb0, size_t rsb1, size_t csb1, int32_t beta,
    int32_t *c_pack, __attribute__((unused)) size_t rsc0,
    __attribute__((unused)) size_t csc0, size_t rsc1, size_t csc1) {
  int status = 0;
  SKL_TEST_REQUIRE(status, m0 == 1);
  SKL_TEST_REQUIRE(status, n0 == 1);
  SKL_TEST_REQUIRE(status, k0 == 4);
  SKL_TEST_REQUIRE(status, csa0 == 1);
  SKL_TEST_REQUIRE(status, rsa1 >= k1 * csa1);
  SKL_TEST_REQUIRE(status, csa1 == m0 * k0);
  SKL_TEST_REQUIRE(status, rsb0 == 1);
  SKL_TEST_REQUIRE(status, rsb1 >= n1 * csb1);
  SKL_TEST_REQUIRE(status, csb1 == k0 * n0);
  SKL_TEST_REQUIRE(status, rsc1 >= n1 * csc1);
  SKL_TEST_REQUIRE(status, csc1 == m0 * n0);
  SKL_TEST_REQUIRE(status, alpha == 1);
  SKL_TEST_REQUIRE(status, beta == 0 || beta == 1);
  if (status) {
    exit(status);
  }
  skl_gemm_a1b01_i8_i8pc_i32_xsfvqdotq(
      m0 * m1, n0 * n1, k0 * k1, a_pack /* == a */, rsa1 /* == rsa */, b_pack,
      rsb1, c_pack /* == c */, rsc1 /* == rsc */, beta != 0);
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
_Alignas(ALIGN) int8_t a[ALEN];
_Alignas(ALIGN) int8_t b[BLEN];
_Alignas(ALIGN) int32_t c[CLEN];
#if defined(ENABLE_TEST)
int32_t ref_c[CLEN], test_c[CLEN];
#endif // ENABLE_TEST

#if defined(ENABLE_TEST)
/* Check result after executing test and reference functions.
 *
 * Compares the test_c and ref_c matrices based on a bound derived from
 * problem parameters. Returns nonzero in case of an error, and prints
 * first incorrect output matrix element. */
int check_error(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_gemm_i8rcprc_i8rcprc_i32rcprc_scalar(
      M0, N0, K0, M1, N1, K1, ALPHA, a, (size_t)RSA0, (size_t)CSA0,
      (size_t)RSA1, (size_t)CSA1, b, (size_t)RSB0, (size_t)CSB0, (size_t)RSB1,
      (size_t)CSB1, BETA, ref_c, (size_t)RSC0, (size_t)CSC0, (size_t)RSC1,
      (size_t)CSC1);

  /* Compare the reference and test outputs. */
  for (size_t i1 = 0; i1 < M1; ++i1) {
    for (size_t j1 = 0; j1 < N1; ++j1) {
      for (size_t i0 = 0; i0 < M0; ++i0) {
        for (size_t j0 = 0; j0 < N0; ++j0) {
          size_t idx = i1 * (size_t)RSC1 + j1 * (size_t)CSC1 +
                       i0 * (size_t)RSC0 + j0 * (size_t)CSC0;
          if (test_c[idx] != ref_c[idx]) {
            printf("result [%zu, %zu, %zu, %zu] (%d) != reference (%d)\n", i0,
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

#define STR(S) #S
#define TEST_LABEL(S) STR(S)
#define PRINT_TEST_NAME(S) printf(TEST_LABEL(S) ":\n");

int main(void) {
  int res = EXIT_SUCCESS;

  PRINT_TEST_NAME(SKL_TEST_NAME);
  printf("M0 = %u, N0 = %u, K0 = %u\n", M0, N0, K0);
  printf("M1 = %u, N1 = %u, K1 = %u\n", M1, N1, K1);
  printf("ALPHA = %d, BETA = %d\n", ALPHA, BETA);
  printf("RSA0 = %u, CSA0 = %u, RSA1 = %u, CSA1 = %u\n", RSA0, CSA0, RSA1,
         CSA1);
  printf("RSB0 = %u, CSB0 = %u, RSB1 = %u, CSB1 = %u\n", RSB0, CSB0, RSB1,
         CSB1);
  printf("RSC0 = %u, CSC0 = %u, RSC1 = %u, CSC1 = %u\n", RSC0, CSC0, RSC1,
         CSC1);

  /* Populate the matrices. */
  skl_test_init_i8(a, ALEN, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  skl_test_init_i8(b, BLEN, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  skl_test_init_i32(c, CLEN, SKL_TEST_MIN_I32, SKL_TEST_MAX_I32);

#if defined(ENABLE_TEST)
  /* Make copies of C to write the reference and test outputs to. */
  memcpy(ref_c, c, CLEN * sizeof(int32_t));
  memcpy(test_c, c, CLEN * sizeof(int32_t));
  SKL_TEST_NAME(M0, N0, K0, M1, N1, K1, ALPHA, a, (size_t)RSA0, (size_t)CSA0,
                (size_t)RSA1, (size_t)CSA1, b, (size_t)RSB0, (size_t)CSB0,
                (size_t)RSB1, (size_t)CSB1, BETA, test_c, (size_t)RSC0,
                (size_t)CSC0, (size_t)RSC1, (size_t)CSC1);
  res += check_error();
#endif // ENABLE_TEST

  SKL_BENCHMARK_RUN(TEST_LABEL(SKL_TEST_NAME), M0 * N0 * K0 * M1 * N1 * K1,
                    SKL_TEST_WARMUP, SKL_TEST_NAME, M0, N0, K0, M1, N1, K1,
                    ALPHA, a, (size_t)RSA0, (size_t)CSA0, (size_t)RSA1,
                    (size_t)CSA1, b, (size_t)RSB0, (size_t)CSB0, (size_t)RSB1,
                    (size_t)CSB1, BETA, c, (size_t)RSC0, (size_t)CSC0,
                    (size_t)RSC1, (size_t)CSC1);

  return res;
}

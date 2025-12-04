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
 *  - Both ALPHA and BETA, as integer literals.
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

#include "skl-test.h"
#include "skl.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__riscv_zve32x)
void skl_gemm_i8_i8_i32_zve32x_x390_wrapper(
    size_t m, size_t n, size_t k, int32_t alpha, const int8_t *a, size_t rsa,
    size_t csa, const int8_t *b, size_t rsb, size_t csb, int32_t beta,
    int32_t *c, size_t rsc, size_t csc) {
  int status = 0;
  SKL_TEST_REQUIRE(status, csa == 1);
  SKL_TEST_REQUIRE(status, csb == 1);
  SKL_TEST_REQUIRE(status, csc == 1);
  if (status) {
    exit(status);
  }
  skl_gemm_i8_i8_i32_zve32x_x390(m, n, k, alpha, a, rsa, b, rsb, beta, c, rsc);
}
#endif

#if defined(__riscv_xsfmm32a8i)
void skl_gemm_a1b01_i8c_i8_i32_xsfmm32a8i_wrapper(
    size_t m, size_t n, size_t k, int32_t alpha, const int8_t *a, size_t rsa,
    size_t csa, const int8_t *b, size_t rsb, size_t csb, int32_t beta,
    int32_t *c, size_t rsc, size_t csc) {
  int status = 0;
  SKL_TEST_REQUIRE(status, rsa == 1);
  SKL_TEST_REQUIRE(status, csb == 1);
  SKL_TEST_REQUIRE(status, csc == 1);
  SKL_TEST_REQUIRE(status, alpha == 1);
  SKL_TEST_REQUIRE(status, beta == 0 || beta == 1);
  if (status) {
    exit(status);
  }
  skl_gemm_a1b01_i8c_i8_i32_xsfmm32a8i(m, n, k, a, csa, b, rsb, c, rsc,
                                       beta != 0);
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
_Alignas(ALIGN) int8_t a[ALEN];
_Alignas(ALIGN) int8_t b[BLEN];
_Alignas(ALIGN) int32_t c[CLEN];
#if defined(ENABLE_TEST)
int32_t ref_c[CLEN], test_c[CLEN];
#endif // ENABLE_TEST

#if defined(ENABLE_TEST)
/* Check result after executing test and reference functions.
 *
 * Compares the test_c and ref_c matrices. Returns nonzero in case of an error,
 * and prints first incorrect output matrix element. */
int check_error(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_gemm_i8rc_i8rc_i32rc_scalar(M, N, K, ALPHA, a, RSA, CSA, b, RSB, CSB,
                                  BETA, ref_c, RSC, CSC);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t idx = i * RSC + j * CSC;
      if (test_c[idx] != ref_c[idx]) {
        printf("result [%zu, %zu] (%d) != reference (%d)\n", i, j, test_c[idx],
               ref_c[idx]);
        return 1;
      }
    }
  }

  return 0;
}
#endif // ENABLE_TEST

#define TEST_LABEL(S) #S ":\n"
#define PRINT_TEST_NAME(S) printf(TEST_LABEL(S));

int main(void) {
  int res = EXIT_SUCCESS;

  PRINT_TEST_NAME(SKL_TEST_NAME);
  printf("M = %u, N = %u, K = %u\n", M, N, K);
  printf("ALPHA = %d, BETA = %d\n", ALPHA, BETA);
  printf("RSA = %u, CSA = %u\n", RSA, CSA);
  printf("RSB = %u, CSB = %u\n", RSB, CSB);
  printf("RSC = %u, CSC = %u\n", RSC, CSC);

  /* Populate the matrices. */
  skl_test_init_i8(a, ALEN, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  skl_test_init_i8(b, BLEN, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  skl_test_init_i32(c, CLEN, SKL_TEST_MIN_I32, SKL_TEST_MAX_I32);

#if defined(ENABLE_TEST)
  /* Make copies of C to write the reference and test outputs to. */
  memcpy(ref_c, c, CLEN * sizeof(int32_t));
  memcpy(test_c, c, CLEN * sizeof(int32_t));
  SKL_TEST_NAME(M, N, K, ALPHA, a, RSA, CSA, b, RSB, CSB, BETA, test_c, RSC,
                CSC);
  res += check_error();
#endif // ENABLE_TEST

#if defined(ENABLE_BENCHMARK)
  /* Warmup run */
  SKL_TEST_NAME(M, N, K, ALPHA, a, RSA, CSA, b, RSB, CSB, BETA, c, RSC, CSC);

  /* Benchmark matrix matmul. */
  riscv_fence();
  uint64_t c0 = riscv_read_mcycle();

  SKL_TEST_NAME(M, N, K, ALPHA, a, RSA, CSA, b, RSB, CSB, BETA, c, RSC, CSC);

  riscv_fence();
  uint64_t c1 = riscv_read_mcycle();
  uint64_t cycles = c1 - c0;

  printf("Cycle count: %" PRIu64 "\n", cycles);
  printf("MACCs / cycle = ");
  print_float((float)(M * N * K) / (float)cycles);
  printf("\n");
#endif // ENABLE_BENCHMARK

  return res;
}

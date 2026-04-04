// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

#if !defined(__riscv_xsfvqdotq)
#error This source file requires compiler support for the Xsfvqdotq extension.
#endif

/* Test and benchmark for Xsfvqdotq GEMM: C = alpha * A * B + beta * C.
 *
 * A is M x K row-major with row stride RSA.
 * B is K x N row-major with row stride RSB.
 * C is M x N row-major with row stride RSC.
 *
 * B is first packed into B_pack before applying the Xsfvqdotq GEMM kernel.
 *
 * Users must provide the values of the following as compiler flags using -D:
 *  - All dimensions M, N, and K
 *  - ALPHA, as an integer literal.
 *  - BETA, as an integer literal.
 *
 * Users may optionally provide the values of the following:
 *  - RSA, which must be >= K (the default is RSA = K)
 *  - RSB, which must be >= N (the default is RSB = N)
 *  - RSC, which must be >= N (the default is RSC = N)
 *  - RSB1, the row stride of B_pack, which must be >= 4 * N (the default is
 *    RSB1 = 4 * N).
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

#if defined(ENABLE_TEST)
#include <string.h> // For memcpy
#endif

/* The macros below set the matrix strides. */
#if !defined(RSA)
#define RSA K
#endif
#if !defined(RSB)
#define RSB N
#endif
#if !defined(RSC)
#define RSC N
#endif

#if !defined(RSB1)
#define RSB1 (K0 * N)
#endif

enum {
  ALIGN = 4096, // Align all matrices to 4096 bytes
  CSA = 1,
  CSB = 1,
  CSC = 1,
  K0 = 4, // Xsfvqdotq K0 constraint
  K1 = ((K + (K0 - 1)) / K0),
  ALEN = M * RSA,
  BLEN = K * RSB,
  CLEN = M * RSC,
  BLEN_PACKED = K1 * RSB1,
};

_Alignas(ALIGN) int8_t a[ALEN];
_Alignas(ALIGN) int8_t b[BLEN];
_Alignas(ALIGN) int8_t b_pack[BLEN_PACKED];
_Alignas(ALIGN) int32_t c[CLEN];
#if defined(ENABLE_TEST)
_Alignas(ALIGN) int32_t ref_c[CLEN];
_Alignas(ALIGN) int32_t test_c[CLEN];
#endif // ENABLE_TEST

#if defined(ENABLE_TEST)
/* Check result after executing test and reference functions.
 *
 * Compares the test_c and ref_c matrices. Returns nonzero in case of an error,
 * and prints first incorrect output matrix element. */
int check_error(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_gemm_i8rc_i8rc_i32rc_ref(M, N, K, ALPHA, a, (size_t)RSA, (size_t)CSA,
                                  b, (size_t)RSB, (size_t)CSB, BETA, ref_c,
                                  (size_t)RSC, (size_t)CSC);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < RSC; ++j) {
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
  int status = 0;
  SKL_TEST_REQUIRE(status, RSA >= K);
  SKL_TEST_REQUIRE(status, RSB >= N);
  SKL_TEST_REQUIRE(status, RSC >= N);
  SKL_TEST_REQUIRE(status, RSB1 >= K0 * N);
  if (status) {
    exit(status);
  }

  int res = EXIT_SUCCESS;

  PRINT_TEST_NAME(SKL_TEST_NAME);
  printf("M = %u, N = %u, K = %u\n", M, N, K);
  printf("ALPHA = %d, BETA = %d\n", ALPHA, BETA);
  printf("RSA = %u, RSB = %u, RSC = %u\n", RSA, RSB, RSC);
  printf("RSB1 = %u\n", RSB1);

  /* Populate the matrices. */
  skl_test_init_i8(a, ALEN, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  skl_test_init_i8(b, BLEN, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  skl_test_init_i8(b_pack, BLEN_PACKED, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  skl_test_init_i32(c, CLEN, SKL_TEST_MIN_I32, SKL_TEST_MAX_I32);

  skl_pack_b_i8_xsfvqdotq(K, N, b, (size_t)RSB, b_pack, (size_t)RSB1);

#if defined(ENABLE_TEST)
  /* Make copies of C to write the reference and test outputs to. */
  memcpy(ref_c, c, CLEN * sizeof(int32_t));
  memcpy(test_c, c, CLEN * sizeof(int32_t));
  skl_gemm_i8_i8pc_i32_xsfvqdotq(M, N, K, ALPHA, a, (size_t)RSA, b_pack,
                                 (size_t)RSB1, BETA, test_c, (size_t)RSC);
  res += check_error();
#endif // ENABLE_TEST

  SKL_BENCHMARK_RUN("skl_gemm_i8_i8pc_i32_xsfvqdotq", M * N * K,
                    SKL_TEST_WARMUP, skl_gemm_i8_i8pc_i32_xsfvqdotq, M, N, K,
                    ALPHA, a, (size_t)RSA, b_pack, (size_t)RSB1, BETA, c,
                    (size_t)RSC);

  return res;
}

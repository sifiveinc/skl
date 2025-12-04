// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

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

#ifndef BETA
#error Must define BETA
#elif !(BETA == 0 || BETA == 1)
#error BETA must be 0 or 1
#endif

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

#if !defined(__riscv_xsfvqdotq)
#error This source file requires compiler support for the Xsfvqdotq extension.
#endif

#include "skl-test.h"
#include "skl.h"
#include <inttypes.h>
#include <riscv_vector.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(ENABLE_TEST)
#include <string.h> // For memcpy
#endif

enum {
  ALIGN = 4096, // Align all matrices to 4096 bytes
  ALEN = M * K,
  BLEN = K * N,
  CLEN = M * N,
  RSA = K,
  CSA = 1,
  RSB = N,
  CSB = 1,
  RSC = N,
  CSC = 1,
  ALPHA = 1,
  ACCUM = (BETA == 1), // NOLINT(misc-redundant-expression)
  KPACK = ((K + 3) / 4) * 4,
  BLEN_PACKED = KPACK * N,
};
_Alignas(ALIGN) int8_t a[ALEN];
_Alignas(ALIGN) int8_t b[BLEN];
_Alignas(ALIGN) int8_t b_pack[BLEN_PACKED];
_Alignas(ALIGN) int32_t c[CLEN];
#if defined(ENABLE_TEST)
_Alignas(ALIGN) int32_t c_ref[CLEN];
_Alignas(ALIGN) int32_t c_test[CLEN];
#endif // ENABLE_TEST

#if defined(ENABLE_TEST)
/* Check result after executing test and reference functions.
 *
 * Compares the test_c and ref_c matrices. Returns nonzero in case of an error,
 * and prints first incorrect output matrix element. */
int check_error(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_gemm_i8rc_i8rc_i32rc_scalar(M, N, K, ALPHA, a, RSA, CSA, b, RSB, CSB,
                                  BETA, c_ref, RSC, CSC);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < RSC; ++j) {
      size_t idx = i * RSC + j * CSC;
      if (c_test[idx] != c_ref[idx]) {
        printf("result [%zu, %zu] (%d) != reference (%d)\n", i, j, c_test[idx],
               c_ref[idx]);
        return 1;
      }
    }
  }

  return 0;
}
#endif // ENABLE_TEST

int main(void) {
  int res = EXIT_SUCCESS;
  skl_test_init_i8(a, ALEN, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  skl_test_init_i8(b, BLEN, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  skl_test_init_i32(c, CLEN, SKL_TEST_MIN_I32, SKL_TEST_MAX_I32);
  printf("M = %u, N = %u, K = %u\n", M, N, K);
  printf("ALPHA = %d, BETA = %d\n", ALPHA, BETA);

#if defined(ENABLE_TEST)
  memcpy(c_ref, c, CLEN * sizeof(int32_t));
  memcpy(c_test, c, CLEN * sizeof(int32_t));

  skl_pack_b_i8_xsfvqdotq(K, N, b, RSB, b_pack);
  skl_gemm_a1b01_i8_i8p_i32_xsfvqdotq(M, N, K, a, RSA, b_pack, RSB, c_test, RSC,
                                      ACCUM);

  res += check_error();
#endif // ENABLE_TEST

#if defined(ENABLE_BENCHMARK)
  // k dimension of packed buffer should be multiple of 4.
  skl_pack_b_i8_xsfvqdotq(K, N, b, RSB, b_pack);

  // warmup.
  skl_gemm_a1b01_i8_i8p_i32_xsfvqdotq(M, N, K, a, RSA, b_pack, RSB, c, RSC,
                                      ACCUM);

  // benchmark.
  uint64_t c0 = riscv_read_mcycle();
  skl_gemm_a1b01_i8_i8p_i32_xsfvqdotq(M, N, K, a, RSA, b_pack, RSB, c, RSC,
                                      ACCUM);
  uint64_t c1 = riscv_read_mcycle();
  uint64_t cycles = c1 - c0;
  printf("Cycle count: %" PRIu64 "\n", cycles);
  printf("MACCs / cycle: ");
  print_float((float)(M * N * K) / (float)cycles);
  printf("\n");
#endif // ENABLE_BENCHMARK

  return res;
}

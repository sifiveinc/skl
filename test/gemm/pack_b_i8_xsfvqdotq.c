// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

/* Test and benchmark for skl_pack_b_i8_xsfvqdotq.
 *
 * B is a K x N row-major matrix with row stride RSB.
 * B_pack is a packed K1 x N1 row-major matrix with row stride RSB1 and K0 x N0
 * column-major tiles, where K0 = 4, N0 = 1, K1 = ceil(K / K0), and N1 = N.
 *
 * Users must provide the values of the following as compiler flags using -D:
 *  - Dimensions K and N
 *
 * Users may optionally provide the values of the following:
 *  - RSB, which must be >= N (the default is RSB = N)
 *  - RSB1, which must be >= 4 * N (the default is RSB1 = 4 * N).
 */

#if !(defined(ENABLE_TEST) || defined(ENABLE_BENCHMARK))
#error Must define at least one of ENABLE_TEST and ENABLE_BENCHMARK
#endif

#ifndef K
#error Must define K
#endif

#ifndef N
#error Must define N
#endif

#ifndef SKL_TEST_PERF_REPORT
#define SKL_TEST_PERF_REPORT report_perf_mpc
#endif

#include "skl-test.h"
#include "skl.h" // NOLINT(misc-include-cleaner)
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(ENABLE_TEST)
#include <string.h> // For memcpy
#endif

/* The macros below set the matrix strides. */
#if !defined(RSB)
#define RSB N
#endif

#if !defined(RSB1)
#define RSB1 (K0 * N)
#endif

enum {
  ALIGN = 4096,
  CSB = 1,
  K0 = 4, // xsfvqdotq K0 constraint
  N0 = 1, // xsfvqdotq N0 constraint
  K1 = (K + (K0 - 1)) / K0,
  RSB0 = 1,
  CSB0 = 0,
  CSB1 = K0 * N0,
  BLEN = K * RSB,
  BLEN_PACKED = K1 * RSB1,
};

_Alignas(ALIGN) int8_t b[BLEN];
_Alignas(ALIGN) int8_t b_pack[BLEN_PACKED];
#if defined(ENABLE_TEST)
_Alignas(ALIGN) int8_t ref_b_pack[BLEN_PACKED];
_Alignas(ALIGN) int8_t test_b_pack[BLEN_PACKED];
#endif // ENABLE_TEST

#if defined(ENABLE_TEST)
void skl_pack_i8_scalar(size_t m, size_t n, const int8_t *c, size_t rsc,
                        size_t csc, size_t m0, size_t n0, int8_t *c_pack,
                        size_t rsc0, size_t csc0, size_t rsc1, size_t csc1) {
  size_t m1 = (m + m0 - 1) / m0; // ceil(m / m0)
  size_t n1 = (n + n0 - 1) / n0; // ceil(n / n0)
  for (size_t ii1 = 0; ii1 < m1; ++ii1) {
    for (size_t jj1 = 0; jj1 < n1; ++jj1) {
      const int8_t *c_block = c + ii1 * m0 * rsc + jj1 * n0 * csc;
      int8_t *c_pack_block = c_pack + ii1 * rsc1 + jj1 * csc1;
      for (size_t ii0 = 0; ii0 < m0; ++ii0) {
        for (size_t jj0 = 0; jj0 < n0; ++jj0) {
          if (ii1 * m0 + ii0 < m && jj1 * n0 + jj0 < n) {
            c_pack_block[ii0 * rsc0 + jj0 * csc0] =
                c_block[ii0 * rsc + jj0 * csc];
          } else {
            // Pad with zeros
            c_pack_block[ii0 * rsc0 + jj0 * csc0] = 0;
          }
        }
      }
    }
  }
}

int check_error(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_pack_i8_scalar(K, N, b, RSB, CSB, K0, N0, ref_b_pack, RSB0, CSB0,
                     (size_t)RSB1, CSB1);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < BLEN_PACKED; ++i) {
    if (test_b_pack[i] != ref_b_pack[i]) {
      printf("result [%zu] (%d) != reference (%d)\n", i, test_b_pack[i],
             ref_b_pack[i]);
      return 1;
    }
  }

  return 0;
}
#endif

#define TEST_LABEL(S) #S ":\n"
#define PRINT_TEST_NAME(S) printf(TEST_LABEL(S));

int main(void) {
  int status = 0;
  SKL_TEST_REQUIRE(status, RSB >= N);
  SKL_TEST_REQUIRE(status, RSB1 >= 4 * N);
  if (status) {
    exit(status);
  }

  int res = EXIT_SUCCESS;

  PRINT_TEST_NAME(SKL_TEST_NAME);
  printf("K = %u, N = %u\n", K, N);
  printf("RSB = %u, RSB1 = %u\n", RSB, RSB1);

  /* Populate the matrices. */
  SKL_TEST_INIT_I8(b, BLEN);
  SKL_TEST_INIT_I8(b_pack, BLEN_PACKED);

#if defined(ENABLE_TEST)
  memcpy(ref_b_pack, b_pack, BLEN_PACKED * sizeof(int8_t));
  memcpy(test_b_pack, b_pack, BLEN_PACKED * sizeof(int8_t));

  skl_pack_b_i8_xsfvqdotq(K, N, b, RSB, test_b_pack, (size_t)RSB1);
  res += check_error();
#endif // ENABLE_TEST

  SKL_BENCHMARK_RUN(skl_test_name, K * N, SKL_TEST_WARMUP, SKL_TEST_NAME, K, N,
                    b, RSB, b_pack, (size_t)RSB1);

  return res;
}

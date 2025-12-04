// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !(defined(ENABLE_TEST) || defined(ENABLE_BENCHMARK))
#error Must define at least one of ENABLE_TEST and ENABLE_BENCHMARK
#endif

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
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
#include <string.h>

enum {
  ALIGN = 4096, /* Align all matrices to 4096 bytes */
  DIFF_TOLERANCE = 0,
};

void init_zero_i8(int8_t *arr, size_t len) {
  memset(arr, 0, len * sizeof(int8_t));
}

/**
 * General Pack implementation for i8 data type.
 */
void skl_pack_i8(size_t m, size_t n, const int8_t *c, size_t ldc,
                 int8_t *c_packed, size_t MT, size_t NT, size_t rsc_packed,
                 size_t csc_packed, size_t rsc_t, size_t csc_t) {
  size_t m_packed = (m + MT - 1) / MT; // ceil(m / MT)
  size_t n_packed = (n + NT - 1) / NT; // ceil(n / NT)
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      c_packed[(i / MT) * rsc_packed + (j / NT) * csc_packed +
               (i % MT) * rsc_t + (j % NT) * csc_t] = c[i * ldc + j];
    }
    // pad right edge
    for (size_t j = n; j < n_packed * NT; ++j) {
      c_packed[(i / MT) * rsc_packed + (j / NT) * csc_packed +
               (i % MT) * rsc_t + (j % NT) * csc_t] = 0;
    }
  }

  // pad bottom edge
  for (size_t i = m; i < m_packed * MT; ++i) {
    for (size_t j = 0; j < n_packed * NT; ++j) {
      c_packed[(i / MT) * rsc_packed + (j / NT) * csc_packed +
               (i % MT) * rsc_t + (j % NT) * csc_t] = 0;
    }
  }
}

bool repack_verify(const int8_t *goldens, const int8_t *results, size_t k,
                   size_t n, uint64_t diff_tolerance) {
  bool pass = true;
  // flat to 1-d array for verification.
  for (size_t i = 0; i < k * n; i++) {
    int8_t golden = goldens[i];
    int8_t result = results[i];
    uint64_t diff = result > golden ? (uint64_t)(result) - (uint64_t)(golden)
                                    : (uint64_t)(golden) - (uint64_t)(result);
    if (diff > diff_tolerance) {
      pass = false;
      printf("results[%zu]: %d != goldens[%zu]: %d (diff %lu > tol %lu)\n", i,
             result, i, golden, diff, diff_tolerance);
    }
  }
  return pass;
}

int main(void) {
  int res = EXIT_SUCCESS;
  const size_t k0 = 4;
  const size_t K = 5;
  const size_t N = 8;
  const size_t KK = (K + k0 - 1) / k0 * k0; // round up to multiple of k0.
  printf("K: %zu, N: %zu, KK: %zu\n", K, N, KK);

  int8_t *b = (int8_t *)aligned_alloc(ALIGN, K * N * sizeof(int8_t));
  int8_t *b_pack = (int8_t *)aligned_alloc(ALIGN, KK * N * sizeof(int8_t));
#if defined(ENABLE_TEST)
  int8_t *b_pack_ref = (int8_t *)aligned_alloc(ALIGN, KK * N * sizeof(int8_t));

  skl_test_init_i8(b, K * N, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  init_zero_i8(b_pack_ref, KK * N);
  init_zero_i8(b_pack, KK * N);

  skl_pack_i8(K, N, b, N, b_pack_ref, k0, 1, 4 * N, 4, 1, 4);
  skl_pack_b_i8_xsfvqdotq(K, N, b, N, b_pack);
  if (!repack_verify(b_pack_ref, b_pack, KK, N, DIFF_TOLERANCE)) {
    res = EXIT_FAILURE;
    printf("b_pack verification failed\n");
  }

  free(b_pack_ref);
#endif // ENABLE_TEST

#if defined(ENABLE_BENCHMARK)
  skl_test_init_i8(b, K * N, SKL_TEST_MIN_I8, SKL_TEST_MAX_I8);
  init_zero_i8(b_pack, KK * N);

  // warmup.
  skl_pack_b_i8_xsfvqdotq(K, N, b, N, b_pack);

  // benchmark.
  uint64_t c0 = riscv_read_mcycle();
  skl_pack_b_i8_xsfvqdotq(K, N, b, N, b_pack);
  uint64_t c1 = riscv_read_mcycle();
  uint64_t cycles = c1 - c0;
  printf("Cycle count: %" PRIu64 "\n", cycles);
  printf("Output matrix size: %zu x %zu\n", KK, N);
  printf("Throughput (elements / cycle): ");
  print_float((float)(KK * N) / (float)cycles);
  printf("\n");

#endif // ENABLE_BENCHMARK

  free(b);
  free(b_pack);
  return res;
}

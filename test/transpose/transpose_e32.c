// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !(defined(ENABLE_TEST) || defined(ENABLE_BENCHMARK))
#error Must define at least one of ENABLE_TEST and ENABLE_BENCHMARK
#endif

#include "skl-test.h"

#include "skl-ref.h"
// NOLINTNEXTLINE(misc-include-cleaner)
#include "skl.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(ENABLE_TEST)
#include <string.h> // For memcpy
#endif

enum {
  ALIGN = 4096,
  M = 128,
  N = 64,
  RSA = N,
  RSAT = M,
  ALEN = M * RSA,
  ATLEN = N * RSAT
};

_Alignas(ALIGN) float a[ALEN];
_Alignas(ALIGN) float at[ATLEN];
#if defined(ENABLE_TEST)
float ref_at[ATLEN], test_at[ATLEN];
#endif // ENABLE_TEST

#if defined(ENABLE_TEST)
int check_error(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_transpose_e32_scalar(M, N, (uint32_t *)a, RSA, (uint32_t *)ref_at, RSAT);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < RSAT; ++j) {
      if (test_at[i * RSAT + j] != ref_at[i * RSAT + j]) {
        printf("result [%zu, %zu] (%f) != reference (%f)\n", i, j,
               test_at[i * RSAT + j], ref_at[i * RSAT + j]);
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

  /* Populate the matrices. */
  skl_test_init_f32(a, ALEN, SKL_TEST_MIN_F32, SKL_TEST_MAX_F32);
  skl_test_init_f32(at, ATLEN, SKL_TEST_MIN_F32, SKL_TEST_MAX_F32);

#if defined(ENABLE_TEST)
  /* Make copies of A^T to write the reference and test outputs to. */
  memcpy(ref_at, at, ATLEN * sizeof(float));
  memcpy(test_at, at, ATLEN * sizeof(float));
  SKL_TEST_NAME(M, N, (uint32_t *)a, RSA, (uint32_t *)test_at, RSAT);
  res += check_error();
#endif // ENABLE_TEST

  SKL_BENCHMARK_RUN(skl_test_name, M * N, SKL_TEST_WARMUP, SKL_TEST_NAME, M, N,
                    (uint32_t *)a, RSA, (uint32_t *)at, RSAT);

  return res;
}

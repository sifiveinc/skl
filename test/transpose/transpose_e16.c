// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#if !(defined(ENABLE_TEST) || defined(ENABLE_BENCHMARK))
#error Must define at least one of ENABLE_TEST and ENABLE_BENCHMARK
#endif

#include "skl-test.h"

#include "skl-ref.h"
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

_Alignas(ALIGN) _Float16 a[ALEN];
_Alignas(ALIGN) _Float16 at[ATLEN];
#if defined(ENABLE_TEST)
_Float16 ref_at[ATLEN], test_at[ATLEN];
#endif // ENABLE_TEST

#if defined(ENABLE_TEST)
int check_error(void) {
  /* Compute the reference matrix output. */
  skl_transpose_e16_ref(M, N, (uint16_t *)a, RSA, (uint16_t *)ref_at, RSAT);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < RSAT; ++j) {
      if (test_at[i * RSAT + j] != ref_at[i * RSAT + j]) {
        printf("result [%zu, %zu] (%f) != reference (%f)\n", i, j,
               (float)test_at[i * RSAT + j], (float)ref_at[i * RSAT + j]);
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
  skl_test_init_f16(a, ALEN, SKL_TEST_MIN_F16, SKL_TEST_MAX_F16);
  skl_test_init_f16(at, ATLEN, SKL_TEST_MIN_F16, SKL_TEST_MAX_F16);

#if defined(ENABLE_TEST)
  /* Make copies of A^T to write the reference and test outputs to. */
  memcpy(ref_at, at, ATLEN * sizeof(_Float16));
  memcpy(test_at, at, ATLEN * sizeof(_Float16));
  SKL_TEST_NAME(M, N, (uint16_t *)a, RSA, (uint16_t *)test_at, RSAT);
  res += check_error();
#endif // ENABLE_TEST

  SKL_BENCHMARK_RUN(skl_test_name, M * N, SKL_TEST_WARMUP, SKL_TEST_NAME, M, N,
                    (uint16_t *)a, RSA, (uint16_t *)at, RSAT);

  return res;
}

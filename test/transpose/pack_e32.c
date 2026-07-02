// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !(defined(ENABLE_TEST) || defined(ENABLE_BENCHMARK))
#error Must define at least one of ENABLE_TEST and ENABLE_BENCHMARK
#endif

#ifndef M0
#error Must define M0
#endif

#ifndef N0
#error Must define N0
#endif

#ifndef M
#error Must define M
#endif

#ifndef N
#error Must define N
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

#if defined(ENABLE_TEST)
#include <string.h> // For memcpy
#endif

#if defined(__riscv_xsfmmbase)
void skl_pack_tex1_e32_e32pc_xsfmmbase_wrapper(
    size_t m, size_t n, const uint32_t *a, size_t rsa, size_t m0, size_t n0,
    uint32_t *a_pack, size_t rsa0, __attribute((unused)) size_t csa0,
    size_t rsa1, __attribute((unused)) size_t csa1, uint32_t padding_value) {
  int status = 0;
  SKL_TEST_REQUIRE(status, m0 == skl_get_te_xsfmmbase());
  SKL_TEST_REQUIRE(status, n0 == 1);
  SKL_TEST_REQUIRE(status, rsa0 == 1);
  if (status) {
    exit(status);
  }
  skl_pack_tex1_e32_e32pc_xsfmmbase(m, n, a, rsa, a_pack, rsa1, padding_value);
}
#endif

/* The macros below set the matrix strides. */
#if !defined(RSA)
#define RSA N // Default to row major
#endif
#if !defined(RSA0) && !defined(CSA0)
#define RSA0 N0 // Default to row major
#define CSA0 1
#elif !defined(RSA0) || !defined(CSA0)
#error Must define both RSA0 and CSA0, or neither
#endif
#if !defined(RSA1) && !defined(CSA1)
#define RSA1 (N1 * CSA1) // Default to row major
#define CSA1 (TILELEN)
#elif !defined(RSA1) || !defined(CSA1)
#error Must define both RSA1 and CSA1, or neither
#endif
#if !defined(PADDING_VALUE)
#define PADDING_VALUE 0
#endif

enum {
  ALIGN = 4096,
  M1 = (M + M0 - 1) / M0,
  N1 = (N + N0 - 1) / N0,
  TILELEN = RSA0 >= CSA0 ? M0 * RSA0 : N0 * CSA0,
  ALEN = M * RSA,
  APACKLEN = RSA1 >= CSA1 ? M1 * RSA1 : N1 * CSA1
};

_Alignas(ALIGN) float a[ALEN];
_Alignas(ALIGN) float a_pack[APACKLEN];
#if defined(ENABLE_TEST)
float ref_a_pack[APACKLEN], test_a_pack[APACKLEN];
#endif // ENABLE_TEST

static void skl_pack_e32_scalar(size_t m, size_t n, const uint32_t *a,
                                size_t rsa, size_t m0, size_t n0,
                                uint32_t *a_pack, size_t rsa0, size_t csa0,
                                size_t rsa1, size_t csa1,
                                uint32_t padding_value) {
  const size_t m1 = (m + m0 - 1) / m0;
  const size_t n1 = (n + n0 - 1) / n0;
  for (size_t i1 = 0; i1 < m1; ++i1)
    for (size_t j1 = 0; j1 < n1; ++j1)
      for (size_t i0 = 0; i0 < m0; ++i0)
        for (size_t j0 = 0; j0 < n0; ++j0)
          if (i1 * m0 + i0 < m && j1 * n0 + j0 < n)
            a_pack[i1 * rsa1 + j1 * csa1 + i0 * rsa0 + j0 * csa0] =
                a[(i1 * m0 + i0) * rsa + j1 * n0 + j0];
          else
            a_pack[i1 * rsa1 + j1 * csa1 + i0 * rsa0 + j0 * csa0] =
                padding_value;
}

#if defined(ENABLE_TEST)
int check_error(void) {
  /* Compute the reference (scalar) matrix output. */
  skl_pack_e32_scalar(M, N, (uint32_t *)a, (size_t)RSA, (size_t)M0, (size_t)N0,
                      (uint32_t *)ref_a_pack, (size_t)RSA0, (size_t)CSA0,
                      (size_t)RSA1, (size_t)CSA1, PADDING_VALUE);

  /* Compare the reference and test outputs. */
  for (size_t i1 = 0; i1 < M1; ++i1) {
    for (size_t j1 = 0; j1 < N1; ++j1) {
      for (size_t i0 = 0; i0 < M0; ++i0) {
        for (size_t j0 = 0; j0 < N0; ++j0) {
          size_t idx = i1 * (size_t)RSA1 + j1 * (size_t)CSA1 +
                       i0 * (size_t)RSA0 + j0 * (size_t)CSA0;
          if (test_a_pack[idx] != ref_a_pack[idx]) {
            printf("result [%zu, %zu, %zu, %zu] (%f) != reference (%f)\n", i1,
                   j1, i0, j0, test_a_pack[idx], ref_a_pack[idx]);
            return 1;
          }
        }
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
  for (size_t i = 0; i < ALEN; ++i)
    a[i] = (float)i;
  skl_test_init_f32(a_pack, APACKLEN, -1, -1);

#if defined(ENABLE_TEST)
  /* Make copies of A_pack to write the reference and test outputs to. */
  memcpy(ref_a_pack, a_pack, APACKLEN * sizeof(float));
  memcpy(test_a_pack, a_pack, APACKLEN * sizeof(float));
  SKL_TEST_NAME(M, N, (uint32_t *)a, (size_t)RSA, M0, N0,
                (uint32_t *)test_a_pack, (size_t)RSA0, (size_t)CSA0,
                (size_t)RSA1, (size_t)CSA1, PADDING_VALUE);
  res += check_error();
#endif // ENABLE_TEST

  SKL_BENCHMARK_RUN(skl_test_name, M1 * N1 * M0 * N0, SKL_TEST_WARMUP,
                    SKL_TEST_NAME, M, N, (uint32_t *)a, (size_t)RSA, M0, N0,
                    (uint32_t *)test_a_pack, (size_t)RSA0, (size_t)CSA0,
                    (size_t)RSA1, (size_t)CSA1, PADDING_VALUE);

  return res;
}

// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "skl-test-driver.h"
#include "skl.h"
#include "transpose/transpose_e8.h"

#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv_zve32x)
#error "This file requires the Zve32x extension."
#endif

/**
 * @brief Test cases for transpose with Zve32x extension.
 *
 * This test uses the transpose_e8 harness.
 */

#define TEST                                                                   \
  .steps = {                                                                   \
      .init = transpose_e8_init,                                               \
      .warmup = NULL,                                                          \
      .execute = execute,                                                      \
      .verify = transpose_e8_verify,                                           \
      .report = NULL,                                                          \
      .cleanup = transpose_e8_cleanup,                                         \
  }

#define BENCH                                                                  \
  .steps = {                                                                   \
      .init = transpose_e8_init,                                               \
      .warmup = execute,                                                       \
      .execute = execute,                                                      \
      .verify = NULL,                                                          \
      .report = transpose_e8_report,                                           \
      .cleanup = transpose_e8_cleanup,                                         \
  }

static void execute(skl_test_t *t);

// clang-format off
transpose_e8_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
  // Benchmark tests
  {BENCH, .m = 64, .n = 128},
#endif

#ifdef SKL_ENABLE_TESTS
  // Verification tests
  /* Edge cases */
  {TEST, .m =   0, .n =   0},
  {TEST, .m =   0, .n =   1},
  {TEST, .m =   1, .n =   0},
  {TEST, .m =   1, .n =   1},

  /* e8m1 boundary*/
  {TEST, .m =  31, .n =  32},
  {TEST, .m =  32, .n =  31},
  {TEST, .m =  32, .n =  32},
  {TEST, .m =  32, .n =  33},
  {TEST, .m =  33, .n =  32},

  /* Wide/tall matrices */
  {TEST, .m =   1, .n =  32},
  {TEST, .m =   2, .n =  32},
  {TEST, .m =  32, .n =   1},
  {TEST, .m =  32, .n =   2},

  /* Nontrivial leading dimensions */
  {TEST, .m =  32, .n =  32, .rsa =  64, .rsat =  64},
  {TEST, .m =  64, .n =  32, .rsa =  64, .rsat =  64},
  {TEST, .m =  32, .n =  64, .rsa =  64, .rsat =  64},
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_transpose_e8_zve32x",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(transpose_e8_t),
                                 .tests = tests};

static void execute(skl_test_t *t) {
  const transpose_e8_t *h = (transpose_e8_t *)t->harness;

  skl_transpose_e8_zve32x(h->m, h->n, h->a.data, h->rsa, h->at.data, h->rsat);
}

int main(void) {
  // Set default row strides to row length
  for (size_t i = 0; i < suite.num_tests; i++) {
    tests[i].rsa = tests[i].rsa ? tests[i].rsa : tests[i].n;
    tests[i].rsat = tests[i].rsat ? tests[i].rsat : tests[i].m;
  }

  return skl_test_driver_run_suite(&suite);
}

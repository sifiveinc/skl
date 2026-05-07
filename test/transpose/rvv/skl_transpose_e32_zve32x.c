// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-test-driver.h"
#include "skl.h"
#include "transpose/transpose_e32.h"

#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv_zve32x)
#error "This file requires the Zve32x extension."
#endif

/**
 * @brief Test cases for transpose with Zve32x extension.
 *
 * This test uses the transpose_e32 harness.
 */

#define TEST                                                                   \
  TRANSPOSE_E32_DEFAULTS, .steps = {                                           \
                              .init = transpose_e32_init,                      \
                              .warmup = NULL,                                  \
                              .execute = execute,                              \
                              .verify = transpose_e32_verify,                  \
                              .report = transpose_e32_report_test,             \
                              .cleanup = transpose_e32_cleanup,                \
  }

#define BENCH                                                                  \
  TRANSPOSE_E32_DEFAULTS, .steps = {                                           \
                              .init = transpose_e32_init,                      \
                              .warmup = execute,                               \
                              .execute = execute,                              \
                              .verify = NULL,                                  \
                              .report = transpose_e32_report_benchmark,        \
                              .cleanup = transpose_e32_cleanup,                \
  }

static void execute(skl_test_t *t);

// clang-format off
transpose_e32_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
  // Benchmark tests
  {BENCH, .m = 128, .n = 128},
#endif

#ifdef SKL_ENABLE_TESTS
  // Verification tests
  /* Edge cases */
  {TEST, .m =   0, .n =   0},
  {TEST, .m =   0, .n =   1},
  {TEST, .m =   1, .n =   0},
  {TEST, .m =   1, .n =   1},

  /* Almost square, dimensions near powers of 2 */
  {TEST, .m =  63, .n =  64},
  {TEST, .m =  64, .n =  63},
  {TEST, .m =  64, .n =  64},
  {TEST, .m =  64, .n =  65},
  {TEST, .m =  65, .n =  64},

  {TEST, .m = 127, .n = 128},
  {TEST, .m = 128, .n = 127},
  {TEST, .m = 128, .n = 128},
  {TEST, .m = 128, .n = 129},
  {TEST, .m = 129, .n = 128},

  {TEST, .m = 255, .n = 256},
  {TEST, .m = 256, .n = 255},
  {TEST, .m = 256, .n = 256},
  {TEST, .m = 256, .n = 257},
  {TEST, .m = 257, .n = 256},

  /* Wide/tall matrices */
  {TEST, .m =   1, .n = 128},
  {TEST, .m =   2, .n = 128},
  {TEST, .m = 128, .n =   1},
  {TEST, .m = 128, .n =   2},

  /* Nontrivial leading dimensions */
  {TEST, .m = 128, .n = 128, .rsa = 256, .rsat = 256},
  {TEST, .m = 256, .n = 128, .rsa = 256, .rsat = 256},
  {TEST, .m = 128, .n = 256, .rsa = 256, .rsat = 256},
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_transpose_e32_zve32x",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(transpose_e32_t),
                                 .tests = tests};

static void execute(skl_test_t *t) {
  const transpose_e32_t *h = (transpose_e32_t *)t->harness;

  skl_transpose_e32_zve32x(h->m, h->n, h->a.data, h->rsa, h->at.data, h->rsat);
}

int main(void) {
  // Set default row strides to row length
  for (size_t i = 0; i < suite.num_tests; i++) {
    tests[i].rsa = tests[i].rsa ? tests[i].rsa : tests[i].n;
    tests[i].rsat = tests[i].rsat ? tests[i].rsat : tests[i].m;
  }

  return skl_test_driver_run_suite(&suite);
}

// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "skl-test-driver.h"
#include "skl.h"
#include "transpose/transpose_e8.h"

#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv_xsfmmbase)
#error "This file requires the Xsfmmbase extension."
#endif

/**
 * @brief Test cases for transpose with Xsfmmbase extension.
 *
 * This test uses the transpose_e8 harness.
 */

#define TEST                                                                   \
  TRANSPOSE_E8_DEFAULTS, .steps = {                                            \
                             .init = transpose_e8_init,                        \
                             .warmup = NULL,                                   \
                             .execute = execute,                               \
                             .verify = transpose_e8_verify,                    \
                             .report = NULL,                                   \
                             .cleanup = transpose_e8_cleanup,                  \
  }

#define BENCH                                                                  \
  TRANSPOSE_E8_DEFAULTS, .steps = {                                            \
                             .init = transpose_e8_init,                        \
                             .warmup = execute,                                \
                             .execute = execute,                               \
                             .verify = NULL,                                   \
                             .report = transpose_e8_report,                    \
                             .cleanup = transpose_e8_cleanup,                  \
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
  {TEST, .m = 32, .n = 32},
#endif
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_transpose_e8_xsfmmbase",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(transpose_e8_t),
                                 .tests = tests};

static void execute(skl_test_t *t) {
  const transpose_e8_t *h = (transpose_e8_t *)t->harness;

  skl_transpose_e8_xsfmmbase(h->m, h->n, h->a.data, h->rsa, h->at.data,
                             h->rsat);
}

int main(void) {
  // Set default row strides to row length
  for (size_t i = 0; i < suite.num_tests; i++) {
    tests[i].rsa = tests[i].rsa ? tests[i].rsa : tests[i].n;
    tests[i].rsat = tests[i].rsat ? tests[i].rsat : tests[i].m;
  }

  return skl_test_driver_run_suite(&suite);
}

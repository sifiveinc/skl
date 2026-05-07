// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "cvt_f4_f8.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv_zvfofp4min)
#error This file requires the Zvfofp4min  extension
#endif

/**
 * @brief Test cases for Conversion from OFP8 to BF16 and F32 to OFP8.
 */

#define TEST_F4E2M1_F8E4M3                                                     \
  CVT_F4_F8_DEFAULTS, .steps = {                                               \
                          .init = cvt_f4_f8_init,                              \
                          .warmup = NULL,                                      \
                          .execute = execute_f4e2m1_f8e4m3,                    \
                          .verify = cvt_f4_f8_verify,                          \
                          .report = NULL,                                      \
                          .cleanup = cvt_f4_f8_cleanup,                        \
  }

#define BENCH_F4E2M1_F8E4M3                                                    \
  CVT_F4_F8_DEFAULTS, .steps = {                                               \
                          .init = cvt_f4_f8_init,                              \
                          .warmup = execute_f4e2m1_f8e4m3,                     \
                          .execute = execute_f4e2m1_f8e4m3,                    \
                          .verify = NULL,                                      \
                          .report = cvt_f4_f8_report,                          \
                          .cleanup = cvt_f4_f8_cleanup,                        \
  }

static void execute_f4e2m1_f8e4m3(skl_test_t *t);

// clang-format off
cvt_f4_f8_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
    // Benchmark tests
    {BENCH_F4E2M1_F8E4M3,   .len = 1024},
#endif // SKL_ENABLE_BENCHMARKS

#ifdef SKL_ENABLE_TESTS
    // Verification tests
    {TEST_F4E2M1_F8E4M3,    .len = 1024},
    {TEST_F4E2M1_F8E4M3,    .len = 1025},
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_cvt_f4_f8_zvfofp4min",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(cvt_f4_f8_t),
                                 .tests = tests};

static void execute_f4e2m1_f8e4m3(skl_test_t *t) {
  const cvt_f4_f8_t *h = (cvt_f4_f8_t *)t->harness;

  skl_cvt_f4e2m1_f8e4m3_zvfofp4min(h->out.data, h->in.data, h->len);
}

int main(void) { return skl_test_driver_run_suite(&suite); }

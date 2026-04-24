// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "cvt_f8_bf16.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv_zvfofp8min)
#error This file requires the Zvfofp8min  extension
#endif

/**
 * @brief Test cases for Conversion from OFP8 to BF16 and F32 to OFP8.
 */

#define TEST_F8E4M3_BF16                                                       \
  CVT_F8_BF16_DEFAULTS, .in_type = F8E4M3,                                     \
      .steps = {                                                               \
          .init = cvt_f8_bf16_init,                                            \
          .warmup = NULL,                                                      \
          .execute = execute_f8e4m3_bf16,                                      \
          .verify = cvt_f8_bf16_verify,                                        \
          .report = NULL,                                                      \
          .cleanup = cvt_f8_bf16_cleanup,                                      \
  }

#define TEST_F8E5M2_BF16                                                       \
  CVT_F8_BF16_DEFAULTS, .in_type = F8E5M2,                                     \
      .steps = {                                                               \
          .init = cvt_f8_bf16_init,                                            \
          .warmup = NULL,                                                      \
          .execute = execute_f8e5m2_bf16,                                      \
          .verify = cvt_f8_bf16_verify,                                        \
          .report = NULL,                                                      \
          .cleanup = cvt_f8_bf16_cleanup,                                      \
  }

#define BENCH_F8E4M3_BF16                                                      \
  CVT_F8_BF16_DEFAULTS, .in_type = F8E4M3,                                     \
      .steps = {                                                               \
          .init = cvt_f8_bf16_init,                                            \
          .warmup = execute_f8e4m3_bf16,                                       \
          .execute = execute_f8e4m3_bf16,                                      \
          .verify = NULL,                                                      \
          .report = cvt_f8_bf16_report,                                        \
          .cleanup = cvt_f8_bf16_cleanup,                                      \
  }

#define BENCH_F8E5M2_BF16                                                      \
  CVT_F8_BF16_DEFAULTS, .in_type = F8E5M2,                                     \
      .steps = {                                                               \
          .init = cvt_f8_bf16_init,                                            \
          .warmup = execute_f8e5m2_bf16,                                       \
          .execute = execute_f8e5m2_bf16,                                      \
          .verify = NULL,                                                      \
          .report = cvt_f8_bf16_report,                                        \
          .cleanup = cvt_f8_bf16_cleanup,                                      \
  }

static void execute_f8e4m3_bf16(skl_test_t *t);
static void execute_f8e5m2_bf16(skl_test_t *t);

// clang-format off
cvt_f8_bf16_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKING
    // Benchmark tests
    {BENCH_F8E4M3_BF16,    .len = 1024},
    {BENCH_F8E5M2_BF16,    .len = 1024,},
#endif // SKL_ENABLE_BENCHMARKING

#ifdef SKL_ENABLE_VALIDATION
    // Verification tests
    {TEST_F8E4M3_BF16,     .len = 1024},
    {TEST_F8E5M2_BF16,     .len = 1024},
#endif // SKL_ENABLE_VALIDATION
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_cvt_f8_bf16_zvfofp8min",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(cvt_f8_bf16_t),
                                 .tests = tests};

static void execute_f8e4m3_bf16(skl_test_t *t) {
  const cvt_f8_bf16_t *h = (cvt_f8_bf16_t *)t->harness;

  skl_cvt_f8e4m3_bf16_zvfofp8min(h->out.data, h->in.data, h->len);
}

static void execute_f8e5m2_bf16(skl_test_t *t) {
  const cvt_f8_bf16_t *h = (cvt_f8_bf16_t *)t->harness;

  skl_cvt_f8e5m2_bf16_zvfofp8min(h->out.data, h->in.data, h->len);
}

int main(void) { return skl_test_driver_run_suite(&suite); }

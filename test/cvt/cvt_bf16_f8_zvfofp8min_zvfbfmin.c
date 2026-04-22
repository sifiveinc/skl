// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "cvt_bf16_f8.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv_zvfofp8min) || !defined(__riscv_zvfbfmin)
#error This file requires the Zvfofp8min and Zvfbfmin extensions
#endif

/**
 * @brief Test cases for Conversion from BF16 to OFP8.
 */

#define TEST_BF16_F8E4M3                                                       \
  CVT_BF16_F8_DEFAULTS, .out_type = F8E4M3,                                    \
      .steps = {                                                               \
          .init = cvt_bf16_f8_init,                                            \
          .warmup = NULL,                                                      \
          .execute = execute_bf16_f8e4m3,                                      \
          .verify = cvt_bf16_f8_verify,                                        \
          .report = NULL,                                                      \
          .cleanup = cvt_bf16_f8_cleanup,                                      \
  }

#define TEST_BF16_F8E5M2                                                       \
  CVT_BF16_F8_DEFAULTS, .out_type = F8E5M2,                                    \
      .steps = {                                                               \
          .init = cvt_bf16_f8_init,                                            \
          .warmup = NULL,                                                      \
          .execute = execute_bf16_f8e5m2,                                      \
          .verify = cvt_bf16_f8_verify,                                        \
          .report = NULL,                                                      \
          .cleanup = cvt_bf16_f8_cleanup,                                      \
  }

#define BENCH_BF16_F8E4M3                                                      \
  CVT_BF16_F8_DEFAULTS, .out_type = F8E4M3,                                    \
      .steps = {                                                               \
          .init = cvt_bf16_f8_init,                                            \
          .warmup = execute_bf16_f8e4m3,                                       \
          .execute = execute_bf16_f8e4m3,                                      \
          .verify = NULL,                                                      \
          .report = cvt_bf16_f8_report,                                        \
          .cleanup = cvt_bf16_f8_cleanup,                                      \
  }

#define BENCH_BF16_F8E5M2                                                      \
  CVT_BF16_F8_DEFAULTS, .out_type = F8E5M2,                                    \
      .steps = {                                                               \
          .init = cvt_bf16_f8_init,                                            \
          .warmup = execute_bf16_f8e5m2,                                       \
          .execute = execute_bf16_f8e5m2,                                      \
          .verify = NULL,                                                      \
          .report = cvt_bf16_f8_report,                                        \
          .cleanup = cvt_bf16_f8_cleanup,                                      \
  }

static void execute_bf16_f8e4m3(skl_test_t *t);
static void execute_bf16_f8e5m2(skl_test_t *t);

// clang-format off
cvt_bf16_f8_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
    // Benchmark tests
    {BENCH_BF16_F8E4M3,  .saturation = false,  .len = 1024, .scale = 1.0f},
    {BENCH_BF16_F8E4M3,  .saturation = false,  .len = 1024, .scale = 1.5f},
    {BENCH_BF16_F8E4M3,  .saturation = true,   .len = 1024, .scale = 1.0f},
    {BENCH_BF16_F8E4M3,  .saturation = true,   .len = 1024, .scale = 1.5f},
    {BENCH_BF16_F8E5M2,  .saturation = false,  .len = 1024, .scale = 1.0f},
    {BENCH_BF16_F8E5M2,  .saturation = false,  .len = 1024, .scale = 1.5f},
    {BENCH_BF16_F8E5M2,  .saturation = true,   .len = 1024, .scale = 1.0f},
    {BENCH_BF16_F8E5M2,  .saturation = true,   .len = 1024, .scale = 1.5f},
#endif // SKL_ENABLE_BENCHMARKS

#ifdef SKL_ENABLE_TESTS
    // Verification tests
    {TEST_BF16_F8E4M3,  .saturation = false,  .len = 1024, .scale = 1.0f},
    {TEST_BF16_F8E4M3,  .saturation = false,  .len = 1024, .scale = 1.5f},
    {TEST_BF16_F8E4M3,  .saturation = true,   .len = 1024, .scale = 1.0f},
    {TEST_BF16_F8E4M3,  .saturation = true,   .len = 1024, .scale = 1.5f},
    {TEST_BF16_F8E5M2,  .saturation = false,  .len = 1024, .scale = 1.0f},
    {TEST_BF16_F8E5M2,  .saturation = false,  .len = 1024, .scale = 1.5f},
    {TEST_BF16_F8E5M2,  .saturation = true,   .len = 1024, .scale = 1.0f},
    {TEST_BF16_F8E5M2,  .saturation = true,   .len = 1024, .scale = 1.5f},
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_cvt_bf16_f8_zvfofp8min_zvfbfmin",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(cvt_bf16_f8_t),
                                 .tests = tests};

static void execute_bf16_f8e4m3(skl_test_t *t) {
  const cvt_bf16_f8_t *h = (cvt_bf16_f8_t *)t->harness;
  if (h->saturation) {
    skl_cvt_sat_bf16_f8e4m3_zvfofp8min_zvfbfmin(h->out.data, h->in.data,
                                                h->scale, h->len);
  } else {
    skl_cvt_bf16_f8e4m3_zvfofp8min_zvfbfmin(h->out.data, h->in.data, h->scale,
                                            h->len);
  }
}

static void execute_bf16_f8e5m2(skl_test_t *t) {
  const cvt_bf16_f8_t *h = (cvt_bf16_f8_t *)t->harness;
  if (h->saturation) {
    skl_cvt_sat_bf16_f8e5m2_zvfofp8min_zvfbfmin(h->out.data, h->in.data,
                                                h->scale, h->len);
  } else {
    skl_cvt_bf16_f8e5m2_zvfofp8min_zvfbfmin(h->out.data, h->in.data, h->scale,
                                            h->len);
  }
}

int main(void) { return skl_test_driver_run_suite(&suite); }

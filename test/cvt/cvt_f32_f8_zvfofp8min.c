// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "cvt_f32_f8.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv_zvfofp8min)
#error This file requires the Zvfofp8min  extension
#endif

/**
 * @brief Test cases for Conversion from F32 to OFP8.
 */

#define TEST_F32_F8E4M3                                                        \
  CVT_F32_F8_DEFAULTS, .out_type = F8E4M3,                                     \
      .steps = {                                                               \
          .init = cvt_f32_f8_init,                                             \
          .warmup = NULL,                                                      \
          .execute = execute_f32_f8e4m3,                                       \
          .verify = cvt_f32_f8_verify,                                         \
          .report = NULL,                                                      \
          .cleanup = cvt_f32_f8_cleanup,                                       \
  }

#define TEST_F32_F8E5M2                                                        \
  CVT_F32_F8_DEFAULTS, .out_type = F8E5M2,                                     \
      .steps = {                                                               \
          .init = cvt_f32_f8_init,                                             \
          .warmup = NULL,                                                      \
          .execute = execute_f32_f8e5m2,                                       \
          .verify = cvt_f32_f8_verify,                                         \
          .report = NULL,                                                      \
          .cleanup = cvt_f32_f8_cleanup,                                       \
  }

#define BENCH_F32_F8E4M3                                                       \
  CVT_F32_F8_DEFAULTS, .out_type = F8E4M3,                                     \
      .steps = {                                                               \
          .init = cvt_f32_f8_init,                                             \
          .warmup = execute_f32_f8e4m3,                                        \
          .execute = execute_f32_f8e4m3,                                       \
          .verify = NULL,                                                      \
          .report = cvt_f32_f8_report,                                         \
          .cleanup = cvt_f32_f8_cleanup,                                       \
  }

#define BENCH_F32_F8E5M2                                                       \
  CVT_F32_F8_DEFAULTS, .out_type = F8E5M2,                                     \
      .steps = {                                                               \
          .init = cvt_f32_f8_init,                                             \
          .warmup = execute_f32_f8e5m2,                                        \
          .execute = execute_f32_f8e5m2,                                       \
          .verify = NULL,                                                      \
          .report = cvt_f32_f8_report,                                         \
          .cleanup = cvt_f32_f8_cleanup,                                       \
  }

static void execute_f32_f8e4m3(skl_test_t *t);
static void execute_f32_f8e5m2(skl_test_t *t);

// clang-format off
cvt_f32_f8_t tests[] = {
#ifdef SKL_ENABLE_PROFILING
    // Benchmark tests
    {BENCH_F32_F8E4M3,  .saturation = false,  .len = 1024, .scale = 1.0f},
    {BENCH_F32_F8E4M3,  .saturation = false,  .len = 1024, .scale = 1.5f},
    {BENCH_F32_F8E4M3,  .saturation = true,   .len = 1024, .scale = 1.0f},
    {BENCH_F32_F8E4M3,  .saturation = true,   .len = 1024, .scale = 1.5f},
    {BENCH_F32_F8E5M2,  .saturation = false,  .len = 1024, .scale = 1.0f},
    {BENCH_F32_F8E5M2,  .saturation = false,  .len = 1024, .scale = 1.5f},
    {BENCH_F32_F8E5M2,  .saturation = true,   .len = 1024, .scale = 1.0f},
    {BENCH_F32_F8E5M2,  .saturation = true,   .len = 1024, .scale = 1.5f},
#endif // SKL_ENABLE_PROFILING

#ifdef SKL_ENABLE_VERIFICATION
    // Verification tests
    {TEST_F32_F8E4M3,   .saturation = false,  .len = 1024, .scale = 1.0f},
    {TEST_F32_F8E4M3,   .saturation = false,  .len = 1024, .scale = 1.5f},
    {TEST_F32_F8E4M3,   .saturation = true,   .len = 1024, .scale = 1.0f},
    {TEST_F32_F8E4M3,   .saturation = true,   .len = 1024, .scale = 1.5f},
    {TEST_F32_F8E5M2,   .saturation = false,  .len = 1024, .scale = 1.0f},
    {TEST_F32_F8E5M2,   .saturation = false,  .len = 1024, .scale = 1.5f},
    {TEST_F32_F8E5M2,   .saturation = true,   .len = 1024, .scale = 1.0f},
    {TEST_F32_F8E5M2,   .saturation = true,   .len = 1024, .scale = 1.5f},
#endif // SKL_ENABLE_VERIFICATION
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_cvt_f32_f8_zvfofp8min",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(cvt_f32_f8_t),
                                 .tests = tests};

static void execute_f32_f8e4m3(skl_test_t *t) {
  const cvt_f32_f8_t *h = (cvt_f32_f8_t *)t->harness;

  if (h->saturation) {
    skl_cvt_sat_f32_f8e4m3_zvfofp8min(h->out.data, h->in.data, h->scale,
                                      h->len);
  } else {
    skl_cvt_f32_f8e4m3_zvfofp8min(h->out.data, h->in.data, h->scale, h->len);
  }
}

static void execute_f32_f8e5m2(skl_test_t *t) {
  const cvt_f32_f8_t *h = (cvt_f32_f8_t *)t->harness;

  if (h->saturation) {
    skl_cvt_sat_f32_f8e5m2_zvfofp8min(h->out.data, h->in.data, h->scale,
                                      h->len);
  } else {
    skl_cvt_f32_f8e5m2_zvfofp8min(h->out.data, h->in.data, h->scale, h->len);
  }
}

int main(void) { return skl_test_driver_run_suite(&suite); }

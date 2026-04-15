// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "cvt_ofp.h"
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
  .in_type = BF16, .out_type = F8E4M3,                                         \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = NULL,                                                          \
      .execute = execute_bf16_f8e4m3,                                          \
      .verify = cvt_ofp_verify,                                                \
      .report = NULL,                                                          \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define TEST_SAT_BF16_F8E4M3                                                   \
  .in_type = BF16, .out_type = F8E4M3, .saturation = true,                     \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = NULL,                                                          \
      .execute = execute_sat_bf16_f8e4m3,                                      \
      .verify = cvt_ofp_verify,                                                \
      .report = NULL,                                                          \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define TEST_BF16_F8E5M2                                                       \
  .in_type = BF16, .out_type = F8E5M2,                                         \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = NULL,                                                          \
      .execute = execute_bf16_f8e5m2,                                          \
      .verify = cvt_ofp_verify,                                                \
      .report = NULL,                                                          \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define TEST_SAT_BF16_F8E5M2                                                   \
  .in_type = BF16, .out_type = F8E5M2, .saturation = true,                     \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = NULL,                                                          \
      .execute = execute_sat_bf16_f8e5m2,                                      \
      .verify = cvt_ofp_verify,                                                \
      .report = NULL,                                                          \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define BENCH_BF16_F8E4M3                                                      \
  .in_type = BF16, .out_type = F8E4M3,                                         \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = execute_bf16_f8e4m3,                                           \
      .execute = execute_bf16_f8e4m3,                                          \
      .verify = NULL,                                                          \
      .report = cvt_ofp_report,                                                \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define BENCH_SAT_BF16_F8E4M3                                                  \
  .in_type = BF16, .out_type = F8E4M3, .saturation = true,                     \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = execute_sat_bf16_f8e4m3,                                       \
      .execute = execute_sat_bf16_f8e4m3,                                      \
      .verify = NULL,                                                          \
      .report = cvt_ofp_report,                                                \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define BENCH_BF16_F8E5M2                                                      \
  .in_type = BF16, .out_type = F8E5M2, .saturation = false,                    \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = execute_sat_bf16_f8e5m2,                                       \
      .execute = execute_sat_bf16_f8e5m2,                                      \
      .verify = NULL,                                                          \
      .report = cvt_ofp_report,                                                \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define BENCH_SAT_BF16_F8E5M2                                                  \
  .in_type = BF16, .out_type = F8E5M2, .saturation = true,                     \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = execute_sat_bf16_f8e5m2,                                       \
      .execute = execute_sat_bf16_f8e5m2,                                      \
      .verify = NULL,                                                          \
      .report = cvt_ofp_report,                                                \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

static void execute_bf16_f8e4m3(skl_test_t *t);
static void execute_sat_bf16_f8e4m3(skl_test_t *t);
static void execute_bf16_f8e5m2(skl_test_t *t);
static void execute_sat_bf16_f8e5m2(skl_test_t *t);

// clang-format off
cvt_ofp_t tests[] = {
    // Benchmark tests
    {BENCH_BF16_F8E4M3,     .len = 1024, .scale = 1.0f},
    {BENCH_BF16_F8E4M3,     .len = 1024, .scale = 1.5f},
    {BENCH_SAT_BF16_F8E4M3, .len = 1024, .scale = 1.0f},
    {BENCH_SAT_BF16_F8E4M3, .len = 1024, .scale = 1.5f},
    {BENCH_BF16_F8E5M2,     .len = 1024, .scale = 1.0f},
    {BENCH_BF16_F8E5M2,     .len = 1024, .scale = 1.5f},
    {BENCH_SAT_BF16_F8E5M2, .len = 1024, .scale = 1.0f},
    {BENCH_SAT_BF16_F8E5M2, .len = 1024, .scale = 1.5f},

    // Verification tests
    {TEST_BF16_F8E4M3,      .len = 1024, .scale = 1.0f},
    {TEST_BF16_F8E4M3,      .len = 1024, .scale = 1.5f},
    {TEST_SAT_BF16_F8E4M3,  .len = 1024, .scale = 1.0f},
    {TEST_SAT_BF16_F8E4M3,  .len = 1024, .scale = 1.5f},
    {TEST_BF16_F8E5M2,      .len = 1024, .scale = 1.0f},
    {TEST_BF16_F8E5M2,      .len = 1024, .scale = 1.5f},
    {TEST_SAT_BF16_F8E5M2,  .len = 1024, .scale = 1.0f},
    {TEST_SAT_BF16_F8E5M2,  .len = 1024, .scale = 1.5f},
};
// clang-format on

static skl_test_suite_t suite = {.name = "cvt_ofp_zvfofp8min_zvfbfmin",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(cvt_ofp_t),
                                 .tests = tests};

static void execute_bf16_f8e4m3(skl_test_t *t) {
  const cvt_ofp_t *h = (cvt_ofp_t *)t->harness;

  skl_cvt_bf16_f8e4m3_zvfofp8min_zvfbfmin((uint8_t *)h->out, (__bf16 *)h->in,
                                          h->scale, h->len);
  t->status.execute_status = SKL_TEST_PASS;
}

static void execute_sat_bf16_f8e4m3(skl_test_t *t) {
  const cvt_ofp_t *h = (cvt_ofp_t *)t->harness;

  skl_cvt_sat_bf16_f8e4m3_zvfofp8min_zvfbfmin(
      (uint8_t *)h->out, (__bf16 *)h->in, h->scale, h->len);
  t->status.execute_status = SKL_TEST_PASS;
}

static void execute_bf16_f8e5m2(skl_test_t *t) {
  const cvt_ofp_t *h = (cvt_ofp_t *)t->harness;

  skl_cvt_bf16_f8e5m2_zvfofp8min_zvfbfmin((uint8_t *)h->out, (__bf16 *)h->in,
                                          h->scale, h->len);
  t->status.execute_status = SKL_TEST_PASS;
}

static void execute_sat_bf16_f8e5m2(skl_test_t *t) {
  const cvt_ofp_t *h = (cvt_ofp_t *)t->harness;

  skl_cvt_sat_bf16_f8e5m2_zvfofp8min_zvfbfmin(
      (uint8_t *)h->out, (__bf16 *)h->in, h->scale, h->len);
  t->status.execute_status = SKL_TEST_PASS;
}

int main(void) { return skl_test_driver_run_suite(&suite); }

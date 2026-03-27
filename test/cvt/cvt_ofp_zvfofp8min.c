// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "cvt_ofp.h"
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

#define TEST_F32_F8E4M3                                                        \
  .in_type = F32, .out_type = F8E4M3,                                          \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = NULL,                                                          \
      .execute = execute_f32_f8e4m3,                                           \
      .verify = cvt_ofp_verify,                                                \
      .report = NULL,                                                          \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define TEST_SAT_F32_F8E4M3                                                    \
  .in_type = F32, .out_type = F8E4M3, .saturation = true,                      \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = NULL,                                                          \
      .execute = execute_sat_f32_f8e4m3,                                       \
      .verify = cvt_ofp_verify,                                                \
      .report = NULL,                                                          \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define TEST_F32_F8E5M2                                                        \
  .in_type = F32, .out_type = F8E5M2,                                          \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = NULL,                                                          \
      .execute = execute_f32_f8e5m2,                                           \
      .verify = cvt_ofp_verify,                                                \
      .report = NULL,                                                          \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define TEST_SAT_F32_F8E5M2                                                    \
  .in_type = F32, .out_type = F8E5M2, .saturation = true,                      \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = NULL,                                                          \
      .execute = execute_sat_f32_f8e5m2,                                       \
      .verify = cvt_ofp_verify,                                                \
      .report = NULL,                                                          \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define TEST_F8E4M3_BF16                                                       \
  .in_type = F8E4M3, .out_type = BF16,                                         \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = NULL,                                                          \
      .execute = execute_f8e4m3_bf16,                                          \
      .verify = cvt_ofp_verify,                                                \
      .report = NULL,                                                          \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define TEST_F8E5M2_BF16                                                       \
  .in_type = F8E5M2, .out_type = BF16,                                         \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = NULL,                                                          \
      .execute = execute_f8e5m2_bf16,                                          \
      .verify = cvt_ofp_verify,                                                \
      .report = NULL,                                                          \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define BENCH_F32_F8E4M3                                                       \
  .in_type = F32, .out_type = F8E4M3,                                          \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = execute_f32_f8e4m3,                                            \
      .execute = execute_f32_f8e4m3,                                           \
      .verify = NULL,                                                          \
      .report = cvt_ofp_report,                                                \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define BENCH_SAT_F32_F8E4M3                                                   \
  .in_type = F32, .out_type = F8E4M3, .saturation = true,                      \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = execute_sat_f32_f8e4m3,                                        \
      .execute = execute_sat_f32_f8e4m3,                                       \
      .verify = NULL,                                                          \
      .report = cvt_ofp_report,                                                \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define BENCH_F32_F8E5M2                                                       \
  .in_type = F32, .out_type = F8E5M2, .saturation = false,                     \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = execute_sat_f32_f8e5m2,                                        \
      .execute = execute_sat_f32_f8e5m2,                                       \
      .verify = NULL,                                                          \
      .report = cvt_ofp_report,                                                \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define BENCH_SAT_F32_F8E5M2                                                   \
  .in_type = F32, .out_type = F8E5M2, .saturation = true,                      \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = execute_sat_f32_f8e5m2,                                        \
      .execute = execute_sat_f32_f8e5m2,                                       \
      .verify = NULL,                                                          \
      .report = cvt_ofp_report,                                                \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define BENCH_F8E4M3_BF16                                                      \
  .in_type = F8E4M3, .out_type = BF16,                                         \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = execute_f8e4m3_bf16,                                           \
      .execute = execute_f8e4m3_bf16,                                          \
      .verify = NULL,                                                          \
      .report = cvt_ofp_report,                                                \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

#define BENCH_F8E5M2_BF16                                                      \
  .in_type = F8E5M2, .out_type = BF16,                                         \
  .steps = {                                                                   \
      .init = cvt_ofp_init,                                                    \
      .warmup = execute_f8e5m2_bf16,                                           \
      .execute = execute_f8e5m2_bf16,                                          \
      .verify = NULL,                                                          \
      .report = cvt_ofp_report,                                                \
      .cleanup = cvt_ofp_cleanup,                                              \
  }

static int execute_f32_f8e4m3(skl_test_t *t);
static int execute_sat_f32_f8e4m3(skl_test_t *t);
static int execute_f32_f8e5m2(skl_test_t *t);
static int execute_sat_f32_f8e5m2(skl_test_t *t);
static int execute_f8e4m3_bf16(skl_test_t *t);
static int execute_f8e5m2_bf16(skl_test_t *t);

// clang-format off
cvt_ofp_t tests[] = {
    // Benchmark tests
    {BENCH_F32_F8E4M3,     .len = 1024, .scale = 1.0f},
    {BENCH_F32_F8E4M3,     .len = 1024, .scale = 1.5f},
    {BENCH_SAT_F32_F8E4M3, .len = 1024, .scale = 1.0f},
    {BENCH_SAT_F32_F8E4M3, .len = 1024, .scale = 1.5f},
    {BENCH_F32_F8E5M2,     .len = 1024, .scale = 1.0f},
    {BENCH_F32_F8E5M2,     .len = 1024, .scale = 1.5f},
    {BENCH_SAT_F32_F8E5M2, .len = 1024, .scale = 1.0f},
    {BENCH_SAT_F32_F8E5M2, .len = 1024, .scale = 1.5f},
    {BENCH_F8E4M3_BF16,    .len = 1024},
    {BENCH_F8E5M2_BF16,    .len = 1024,},

    // Verification tests
    {TEST_F32_F8E4M3,      .len = 1024, .scale = 1.0f},
    {TEST_F32_F8E4M3,      .len = 1024, .scale = 1.5f},
    {TEST_SAT_F32_F8E4M3,  .len = 1024, .scale = 1.0f},
    {TEST_SAT_F32_F8E4M3,  .len = 1024, .scale = 1.5f},
    {TEST_F32_F8E5M2,      .len = 1024, .scale = 1.0f},
    {TEST_F32_F8E5M2,      .len = 1024, .scale = 1.5f},
    {TEST_SAT_F32_F8E5M2,  .len = 1024, .scale = 1.0f},
    {TEST_SAT_F32_F8E5M2,  .len = 1024, .scale = 1.5f},
    {TEST_F8E4M3_BF16,     .len = 1024},
    {TEST_F8E5M2_BF16,     .len = 1024},
};
// clang-format on

static skl_test_suite_t suite = {.name = "cvt_ofp_zvfofp8min",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(cvt_ofp_t),
                                 .tests = tests};

static int execute_f32_f8e4m3(skl_test_t *t) {
  const cvt_ofp_t *h = (cvt_ofp_t *)t->harness;

  skl_cvt_f32_f8e4m3_zvfofp8min((uint8_t *)h->out, (float *)h->in, h->scale,
                                h->len);
  return (t->status == SKL_TEST_PASS) ? 0 : 1;
}

static int execute_sat_f32_f8e4m3(skl_test_t *t) {
  const cvt_ofp_t *h = (cvt_ofp_t *)t->harness;

  skl_cvt_sat_f32_f8e4m3_zvfofp8min((uint8_t *)h->out, (float *)h->in, h->scale,
                                    h->len);
  return (t->status == SKL_TEST_PASS) ? 0 : 1;
}

static int execute_f32_f8e5m2(skl_test_t *t) {
  const cvt_ofp_t *h = (cvt_ofp_t *)t->harness;

  skl_cvt_f32_f8e5m2_zvfofp8min((uint8_t *)h->out, (float *)h->in, h->scale,
                                h->len);
  return (t->status == SKL_TEST_PASS) ? 0 : 1;
}

static int execute_sat_f32_f8e5m2(skl_test_t *t) {
  const cvt_ofp_t *h = (cvt_ofp_t *)t->harness;

  skl_cvt_sat_f32_f8e5m2_zvfofp8min((uint8_t *)h->out, (float *)h->in, h->scale,
                                    h->len);
  return (t->status == SKL_TEST_PASS) ? 0 : 1;
}

static int execute_f8e4m3_bf16(skl_test_t *t) {
  const cvt_ofp_t *h = (cvt_ofp_t *)t->harness;

  skl_cvt_f8e4m3_bf16_zvfofp8min((__bf16 *)h->out, (uint8_t *)h->in, h->len);
  return (t->status == SKL_TEST_PASS) ? 0 : 1;
}

static int execute_f8e5m2_bf16(skl_test_t *t) {
  const cvt_ofp_t *h = (cvt_ofp_t *)t->harness;

  skl_cvt_f8e5m2_bf16_zvfofp8min((__bf16 *)h->out, (uint8_t *)h->in, h->len);
  return (t->status == SKL_TEST_PASS) ? 0 : 1;
}

int main(void) { return skl_test_driver_run_suite(&suite); }

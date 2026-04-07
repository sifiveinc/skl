// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfexpa)
#error This file requires the Xsfvfexpa extension
#endif

/**
 * @brief Test cases for Exponential with Xsfvfexpa extension.
 *
 * This test uses the exp_f32 harness.
 */

#include "math/unary_f32.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"

#define BASIC_STEPS                                                            \
  .steps = {                                                                   \
      .init = unary_f32_init,                                                  \
      .execute = execute,                                                      \
      .cleanup = unary_f32_cleanup,                                            \
      .report = unary_f32_report,                                              \
  }

#define MIN (-104.f)
#define MAX (+89.f)

#define TEST                                                                   \
  UNARY_F32_TEST_DEFAULTS, BASIC_STEPS, .steps.warmup = NULL,                  \
                                        .steps.verify = unary_f32_verify

#define BENCH                                                                  \
  UNARY_F32_BENCH_DEFAULTS, BASIC_STEPS, .steps.warmup = execute,              \
                                         .steps.verify = NULL

#define FUNCTION_TESTS(FUN, MIN, MAX, ULP)                                     \
  {TEST,           .ref_func = skl_exp_f32_ref, .func = (FUN), .a.min = (MIN), \
   .a.max = (MAX), .ctx.max_err = (ULP)}

#define FUNCTION_BENCHMARKS(FUN, MIN, MAX)                                     \
  {BENCH, .func = (FUN), .a.min = (MIN), .a.max = (MAX)}

static void execute(skl_test_t *t);

unary_f32_t tests[] = {
#if defined(__riscv_xsfvfexpa)
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_exp_1u_f32_xsfvfexpa, MIN, MAX),
    FUNCTION_BENCHMARKS(skl_exp_1p0002ugen5d639eP6s0_f32_xsfvfexpa,
                        -0x1.5d639ep6f, +89.f),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_exp_1u_f32_xsfvfexpa, MIN, MAX, 1.0f),
    FUNCTION_TESTS(skl_exp_1p0002ugen5d639eP6s0_f32_xsfvfexpa, -0x1.5d639ep6f,
                   +89.f, 1.0002f),
#endif
#endif
};

static skl_test_suite_t suite = {.name = "skl_exp_f32_xsfvfexpa",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f32_t),
                                 .tests = tests};

static void execute(skl_test_t *t) {
  unary_f32_t *h = (unary_f32_t *)t->harness;
  unary_func_t func = h->func;
  func(h->ctx.b, h->a.data, h->a.len);
}

int main(void) { return skl_test_driver_run_suite(&suite); }

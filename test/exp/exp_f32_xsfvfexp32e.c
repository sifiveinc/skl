// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

/**
 * @brief Test cases for Exponential with Xsfvfexp32e extension.
 *
 * This test uses the unary_f32 harness.
 */

#include "math/unary_f32.h"
#include "skl-ref.h" // NOLINT(misc-include-cleaner)
#include "skl-test-driver.h"
#include "skl.h"

#define FUNCTION_TESTS(FUN, MIN, MAX, ULP)                                     \
  {TEST,           .ref_func = skl_exp_f32_ref, .func = (FUN), .a.min = (MIN), \
   .a.max = (MAX), .ctx.max_err = (ULP)}

#define FUNCTION_BENCHMARKS(FUN, MIN, MAX)                                     \
  {BENCH, .func = (FUN), .a.min = (MIN), .a.max = (MAX)}

unary_f32_t tests[] = {
#if defined(__riscv_xsfvfexp32e)
#if defined(SKL_ENABLE_BENCHMARKS)
    FUNCTION_BENCHMARKS(skl_exp_2p398u0alt64ainf_f32_xsfvfexp32e,
                        -0x1.fffffep5f, 0x1.fffffep5f),
    FUNCTION_BENCHMARKS(skl_exp_5p32u_f32_xsfvfexp32e, MIN, MAX),
#endif
#if defined(SKL_ENABLE_TESTS)
    FUNCTION_TESTS(skl_exp_2p398u0alt64ainf_f32_xsfvfexp32e, -0x1.fffffep5f,
                   0x1.fffffep5f, 2.398f),
    FUNCTION_TESTS(skl_exp_5p32u_f32_xsfvfexp32e, MIN, MAX, 5.32f),
#endif
#endif
};

static skl_test_suite_t suite = {.name = "skl_exp_f32_xsfvfexp32e",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(unary_f32_t),
                                 .tests = tests};

static void execute(skl_test_t *t) {
  unary_f32_t *h = (unary_f32_t *)t->harness;
  unary_func_t func = h->func;
  func(h->ctx.b, h->a.data, h->a.len);
}

int main(void) { return skl_test_driver_run_suite(&suite); }
